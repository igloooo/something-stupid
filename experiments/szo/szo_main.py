import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,*([".."]*3))))
import argparse
import random
import time
import pickle
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cv2
from szo_factory import SZONowcastingFactory
from nowcasting.config import cfg, cfg_from_file, save_cfg
from nowcasting.my_module import MyModule
from nowcasting.encoder_forecaster import encoder_forecaster_build_networks, train_step, EncoderForecasterStates
from nowcasting.szo_evaluation import *
from nowcasting.utils import parse_ctx, logging_config, latest_iter_id
from nowcasting.szo_iterator import SZOIterator, save_png_sequence
from nowcasting.helpers.visualization import save_hko_gif
from collections import deque

# Uncomment to try different seeds

# random.seed(12345)
# mx.random.seed(930215)
# np.random.seed(921206)

# random.seed(1234)
# mx.random.seed(93021)
# np.random.seed(92120)

random.seed(123)
mx.random.seed(9302)
np.random.seed(9212)


def parse_args():
    parser = argparse.ArgumentParser(description='Train the SZO nowcasting model')
    parser.add_argument('--batch_size', dest='batch_size', help="batchsize of the training process",
                        default=None, type=int)
    parser.add_argument('--cfg', dest='cfg_file', help='Optional configuration file', default=None, type=str)
    parser.add_argument('--save_dir', help='The saving directory', required=True, type=str)
    parser.add_argument('--ctx', dest='ctx', help='Running Context. E.g `--ctx gpu` or `--ctx gpu0,gpu1` or `--ctx cpu`',
                        type=str, default='gpu')
    parser.add_argument('--resume', dest='resume', action='store_true', default=False)
    parser.add_argument('--resume_param_only', dest='resume_param_only', action='store_true', default=False)
    parser.add_argument('--lr', dest='lr', help='learning rate', default=None, type=float)
    parser.add_argument('--wd', dest='wd', help='weight decay', default=None, type=float)
    parser.add_argument('--grad_clip', dest='grad_clip', help='gradient clipping threshold',
                        default=None, type=float)
    args = parser.parse_args()
    args.ctx = parse_ctx(args.ctx)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file, target=cfg.MODEL)
    if args.batch_size is not None:
        cfg.MODEL.TRAIN.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        cfg.MODEL.TRAIN.LR = args.lr
    if args.wd is not None:
        cfg.MODEL.TRAIN.WD = args.wd
    if args.grad_clip is not None:
        cfg.MODEL.TRAIN.GRAD_CLIP = args.grad_clip
    if args.wd is not None:
        cfg.MODEL.TRAIN.WD = args.wd
    cfg.MODEL.SAVE_DIR = args.save_dir
    logging.info(args)
    return args


def get_base_dir(args):
    if args.save_dir is not None:
        return args.save_dir
    else:
        return "encoder_forecaster_hko"

def get_prediction(data_nd, states, encoder_net, forecaster_net):
    encoder_net.forward(is_train=False,
                        data_batch=mx.io.DataBatch(data=[data_nd]+states.get_encoder_states()))
    encoder_states = encoder_net.get_outputs()
    states.update(encoder_states)

    last_frame_nd = data_nd[data_nd.shape[0] - 1]
    if cfg.MODEL.ENCODER_FORECASTER.USE_SKIP:
        forecaster_net.forward(is_train=False,
                            data_batch=mx.io.DataBatch(data=states.get_forecaster_states()+[last_frame_nd]))
    else:
        forecaster_net.forward(is_train=False,
                            data_batch=mx.io.DataBatch(data=states.get_forecaster_states()))
    return forecaster_net.get_outputs()[0]

def save_prediction(data_nd, target_nd, pred_nd, path, default_as_0=False, mode='display', folder_name=None, gt_path=None, pred_path=None):
    """
    mx.nd.NDArray
    shape (seqlen, batch_size=1, channel=1, height, width)
    pixel value [0,255] np.float32
    if default_as_0 is true, default value will be 0, otherwise
    they will be 255
    """
    epsilon = cfg.MODEL.DISPLAY_EPSILON
    if cfg.MODEL.DATA_MODE == 'original':
        if not default_as_0:
        # remain 0-80, uint8
            data_np = data_nd.asnumpy().astype(np.uint8)
            target_np = target_nd.asnumpy().astype(np.uint8)
            pred_np = pred_nd.asnumpy().astype(np.uint8)
        else:
        # convert 0-80 to 0-255, uint8
            scale_fac = 255.0 / 80.0
            data_np = data_nd.asnumpy()
            target_np = target_nd.asnumpy()
            pred_np = pred_nd.asnumpy()
            data_np = data_np * (data_np<=80.0)
            target_np = target_np * (target_np<=80.0)
            pred_np = pred_np * (pred_np<=80.0)
            data_np = (data_np * scale_fac).astype(np.uint8)
            target_np = (target_np * scale_fac).astype(np.uint8)
            pred_np = (pred_np * scale_fac).astype(np.uint8)
    elif cfg.MODEL.DATA_MODE == 'rescaled':
        if not default_as_0:
            data_np = data_nd.asnumpy()
            target_np = target_nd.asnumpy()
            pred_np = pred_nd.asnumpy()
            for np_arr in (data_np, target_np, pred_np):
                mask = np_arr < epsilon
                np_arr *= (80/255)*(1-mask)
                np_arr += mask*255.0
            data_np = data_np.astype(np.uint8)
            target_np = target_np.astype(np.uint8)
            pred_np = pred_np.astype(np.uint8)
        else:
            data_np = data_nd.asnumpy().astype(np.uint8)
            target_np = target_nd.asnumpy().astype(np.uint8)
            pred_np = pred_nd.asnumpy().astype(np.uint8)
    else:
        raise NotImplementedError

    if mode == 'display':
        inputs = data_np
        ground_truth = target_np
        predictions = pred_np
        
        gif_true = np.concatenate([data_np, target_np], axis=0)
        gif_generated = np.concatenate([data_np, pred_np], axis=0)
        
        save_png_sequence([inputs, ground_truth, predictions], os.path.join(path, 'framewise_comp.png'))
        
        save_hko_gif(gif_true, os.path.join(path, "true.gif"), multiply_by_255=False)
        save_hko_gif(gif_generated, os.path.join(path, "generated.gif"), multiply_by_255=False)
    elif mode == 'save':
        pred_path = os.path.join(pred_path, folder_name)
        gt_path = os.path.join(gt_path, folder_name)
        save_pred_image_sequence(pred_np, pred_path)
        save_gt_image_sequence(target_np, gt_path)
    else:
        raise NotImplementedError

def save_pred_image_sequence(pred_np, dir_path):
    begin_index = 31
    for i in range(pred_np.shape[0]):
        cv2.imwrite(dir_path+"_f%03d"%(i+begin_index)+".png", pred_np[i])

def save_gt_image_sequence(gt_np, dir_path):
    begin_index = 31
    for i in range(gt_np.shape[0]):
        cv2.imwrite(dir_path+"_%03d"%(i+begin_index)+".png", gt_np[i])

def plot_loss_curve(path, losses):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MaxNLocator(nbins=10)
    ax.yaxis.set_major_locator(loc)
    plt.plot(losses)
    plt.savefig(path)
    plt.close('all')

def synchronize_kvstore(module):
    def updater_assign(key, inputs, stored):
        stored[:] = inputs
    module._kvstore._set_updater(updater_assign)
    for k, w in zip(module._exec_group.param_names, module._exec_group.param_arrays):
        module._kvstore.push(k, w[0].as_in_context(mx.cpu(0)))
    module._kvstore._set_updater(mx.optimizer.get_updater(module._optimizer))

def train(args):
    base_dir = get_base_dir(args)
    logging_config(folder=base_dir)
    save_cfg(dir_path=base_dir, source=cfg.MODEL)

    if cfg.MODEL.EXTEND_TO_FULL_OUTLEN:
        iter_outlen = 30
    else:  # onetime or none
        iter_outlen = cfg.MODEL.OUT_LEN
    model_outlen = cfg.MODEL.OUT_LEN  # training on 30 costs too much memory

    train_szo_iter = SZOIterator(rec_paths=cfg.SZO_TRAIN_DATA_PATHS,
                                in_len=cfg.MODEL.IN_LEN,
                                out_len=iter_outlen,  # iterator has to provide full output sequence as target if needed
                                batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
                                frame_skip=cfg.MODEL.FRAME_SKIP,
                                ctx=args.ctx)
    valid_szo_iter = SZOIterator(rec_paths=cfg.SZO_TEST_DATA_PATHS,
                                in_len=cfg.MODEL.IN_LEN,
                                out_len=iter_outlen,
                                batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
                                frame_skip=cfg.MODEL.FRAME_SKIP,
                                ctx=args.ctx)
        
    szo_nowcasting = SZONowcastingFactory(batch_size=cfg.MODEL.TRAIN.BATCH_SIZE // len(args.ctx),
                                          ctx_num=len(args.ctx),
                                          in_seq_len=cfg.MODEL.IN_LEN,
                                          out_seq_len=model_outlen,  # model still generate cfg.MODEL.OUT_LEN number of outputs at a time
                                          frame_stack=cfg.MODEL.FRAME_STACK)
    encoder_net, forecaster_net, loss_net, discrim_net, loss_D_net = \
        encoder_forecaster_build_networks(
            factory=szo_nowcasting,
            context=args.ctx)
    encoder_net.summary()
    forecaster_net.summary()
    loss_net.summary()
    discrim_net.summary()
    loss_D_net.summary()
    if cfg.MODEL.GAN_G_LAMBDA > 0:
        loss_types = ('mse','gdl','gan','dis')
    else:
        loss_types = ('mse', 'gdl')
    # try to load checkpoint
    if args.resume:
        start_iter_id = latest_iter_id(base_dir)
        encoder_net.load_params(os.path.join(base_dir, 'encoder_net'+'-%04d.params'%(start_iter_id)))
        forecaster_net.load_params(os.path.join(base_dir, 'forecaster_net'+'-%04d.params'%(start_iter_id)))
        if not args.resume_param_only:
            encoder_net.load_optimizer_states(os.path.join(base_dir, 'encoder_net'+'-%04d.states'%(start_iter_id)))
            forecaster_net.load_optimizer_states(os.path.join(base_dir, 'forecaster_net'+'-%04d.states'%(start_iter_id)))
        if cfg.MODEL.GAN_G_LAMBDA > 0:
            discrim_net.load_params(os.path.join(base_dir, 'discrim_net'+'-%04d.params'%(start_iter_id)))
            if not args.resume_param_only:
                discrim_net.load_optimizer_states(os.path.join(base_dir, 'discrim_net'+'-%04d.states'%(start_iter_id)))
        for module in (encoder_net, forecaster_net, discrim_net):
            synchronize_kvstore(module)
    else:
        start_iter_id = -1

    if args.resume and (not args.resume_param_only):
        with open(os.path.join(base_dir, 'train_loss_dicts.pkl'), 'rb') as f:
            train_loss_dicts = pickle.load(f)
        with open(os.path.join(base_dir, 'valid_loss_dicts.pkl'), 'rb') as f:
            valid_loss_dicts = pickle.load(f)
        for dicts in (train_loss_dicts, valid_loss_dicts):
            for k in dicts.keys():
                if k not in loss_types:
                    del dicts[k]
    else:
        train_loss_dicts = {}
        valid_loss_dicts = {}
        for dicts in (train_loss_dicts, valid_loss_dicts):
            for typ in loss_types:
                dicts[typ] = []

    states = EncoderForecasterStates(factory=szo_nowcasting, ctx=args.ctx[0])
    for info in szo_nowcasting.init_encoder_state_info:
        assert info["__layout__"].find('N') == 0, "Layout=%s is not supported!" %info["__layout__"]
    for info in szo_nowcasting.init_forecaster_state_info:
        assert info["__layout__"].find('N') == 0, "Layout=%s is not supported!" % info["__layout__"]

    cumulative_loss = {}
    for k in train_loss_dicts.keys():
        cumulative_loss[k] = 0.0

    iter_id = start_iter_id + 1
    if cfg.MODEL.EXTEND_TO_FULL_OUTLEN:
        second_period_flag = False
    buffers = {}
    buffers['fake'] = deque([], maxlen=cfg.MODEL.TRAIN.GEN_BUFFER_LEN)
    buffers['true'] = deque([], maxlen=cfg.MODEL.TRAIN.GEN_BUFFER_LEN)
    while iter_id < cfg.MODEL.TRAIN.MAX_ITER:
        if cfg.MODEL.EXTEND_TO_FULL_OUTLEN and second_period_flag:
            data_nd = data_nd_next
            target_nd = target_nd_next
        else:
            frame_dat = train_szo_iter.sample()
            data_nd = frame_dat[0:cfg.MODEL.IN_LEN,:,:,:,:] / 255.0  # scale to [0,1]
            target_nd = frame_dat[cfg.MODEL.IN_LEN:(cfg.MODEL.IN_LEN + cfg.MODEL.OUT_LEN),:,:,:,:] / 255.0
        states.reset_all()
        if cfg.MODEL.ENCODER_FORECASTER.HAS_MASK:
            mask_nd = target_nd < 1.0
        else:
            mask_nd = mx.nd.ones_like(target_nd)
        
        states, loss_dict, pred_nd, buffers = train_step(batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
                               encoder_net=encoder_net, forecaster_net=forecaster_net,
                               loss_net=loss_net, discrim_net=discrim_net, 
                               loss_D_net=loss_D_net, init_states=states,
                               data_nd=data_nd, gt_nd=target_nd, mask_nd=mask_nd,
                               iter_id=iter_id, buffers=buffers)
        for k in cumulative_loss.keys():
            loss = loss_dict[k+'_output']
            cumulative_loss[k] += loss
        if cfg.MODEL.EXTEND_TO_FULL_OUTLEN:
            if second_period_flag:
                second_period_flag = False
            else:
                second_period_flag = True
                data_nd_next = pred_nd[30-cfg.MODEL.IN_LEN-cfg.MODEL.OUT_LEN: 30-cfg.MODEL.OUT_LEN,:,:,:,:] / 255.0
                target_nd_next = frame_dat[30+cfg.MODEL.IN_LEN-cfg.MODEL.OUT_LEN:,:,:,:,:] / 255.0

        if (iter_id+1) % cfg.MODEL.VALID_ITER == 0:
            if cfg.MODEL.EXTEND_TO_FULL_OUTLEN:
                valid_second_period_flag = False
                iters = 2*cfg.MODEL.VALID_LOOP
            else:
                iters = cfg.MODEL.VALID_LOOP
            for i in range(iters):
                states.reset_all()
                if cfg.MODEL.EXTEND_TO_FULL_OUTLEN and valid_second_period_flag:
                    data_nd_v = data_nd_v_next
                    gt_nd_v = gt_nd_v_next
                else:
                    frame_dat_v = valid_szo_iter.sample()
                    data_nd_v = frame_dat_v[0:cfg.MODEL.IN_LEN,:,:,:,:] / 255.0
                    gt_nd_v = frame_dat_v[cfg.MODEL.IN_LEN:(cfg.MODEL.IN_LEN+cfg.MODEL.OUT_LEN),:,:,:,:] / 255.0
                if cfg.MODEL.ENCODER_FORECASTER.HAS_MASK:
                    mask_nd_v = gt_nd_v < 1.0
                else:
                    mask_nd_v = mx.nd.ones_like(gt_nd_v)
                states, new_valid_loss_dicts, pred_nd_v = valid_step(batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
                                encoder_net=encoder_net, forecaster_net=forecaster_net,
                                loss_net=loss_net, discrim_net=discrim_net,
                                loss_D_net=loss_D_net, init_states=states,
                                data_nd=data_nd_v, gt_nd=gt_nd_v, mask_nd=mask_nd_v,
                                valid_loss_dicts=valid_loss_dicts, iter_id=iter_id)
                if i == 0:
                    for k in valid_loss_dicts.keys():
                        valid_loss_dicts[k].append(new_valid_loss_dicts[k])
                else:
                    for k in valid_loss_dicts.keys():
                        valid_loss_dicts[k][-1] += new_valid_loss_dicts[k]
                if cfg.MODEL.EXTEND_TO_FULL_OUTLEN:
                    if valid_second_period_flag:
                        valid_second_period_flag = False
                    else:
                        valid_second_period_flag = True
                        data_nd_v_next = pred_nd_v[30-cfg.MODEL.IN_LEN-cfg.MODEL.OUT_LEN: 30-cfg.MODEL.OUT_LEN,:,:,:,:] / 255.0
                        gt_nd_v_next = frame_dat_v[30+cfg.MODEL.IN_LEN-cfg.MODEL.OUT_LEN:,:,:,:,:] / 255.0

            for k in valid_loss_dicts.keys():
                valid_loss_dicts[k][-1] /= (iters)
                plot_loss_curve(os.path.join(base_dir, 'valid_'+k+'_loss'), valid_loss_dicts[k])
            
        if (iter_id+1) % cfg.MODEL.DRAW_EVERY == 0:
            for k in train_loss_dicts.keys():
                avg_loss = cumulative_loss[k] / cfg.MODEL.DRAW_EVERY
                if k=='gan':
                    if avg_loss > 1.0:
                        avg_loss = 1.0
                elif k == 'dis':
                    if avg_loss > 1.0:
                        avg_loss = 1.0
                if avg_loss < 0:
                    avg_loss = 0
                train_loss_dicts[k].append(avg_loss)
                cumulative_loss[k] = 0.0
                plot_loss_curve(os.path.join(base_dir, 'train_'+k+'_loss'), train_loss_dicts[k])

        if (iter_id+1) % cfg.MODEL.DISPLAY_EVERY == 0:
            new_frame_dat = train_szo_iter.sample()
            data_nd_d = new_frame_dat[0:cfg.MODEL.IN_LEN,:,:,:,:] / 255.0
            target_nd_d = new_frame_dat[cfg.MODEL.IN_LEN:,:,:,:,:] / 255.0
            states.reset_all()
            pred_nd_d1 = get_prediction(data_nd_d, states, encoder_net, forecaster_net)
            if cfg.MODEL.EXTEND_TO_FULL_OUTLEN:
                states.reset_all()
                pred_nd_d2 = get_prediction(pred_nd_d1[30-cfg.MODEL.IN_LEN-cfg.MODEL.OUT_LEN: 30-cfg.MODEL.OUT_LEN,:,:,:,:], 
                                            states, encoder_net, forecaster_net)
                pred_nd_d = mx.nd.concat(pred_nd_d1[:30-cfg.MODEL.OUT_LEN,:,:,:,:],pred_nd_d2, dim=0)
            else:
                pred_nd_d = pred_nd_d1
            
            display_path1 = os.path.join(base_dir, 'display_'+str(iter_id))
            display_path2 = os.path.join(base_dir, 'display_'+str(iter_id)+'_')
            if not os.path.exists(display_path1):
                os.mkdir(display_path1)
            if not os.path.exists(display_path2):
                os.mkdir(display_path2)

            data_nd_d = (data_nd_d*255.0).clip(0, 255.0)
            target_nd_d = (target_nd_d*255.0).clip(0, 255.0)
            pred_nd_d = (pred_nd_d*255.0).clip(0, 255.0)
            save_prediction(data_nd_d[:,0,0,:,:], target_nd_d[:,0,0,:,:], pred_nd_d[:,0,0,:,:], display_path1, default_as_0=True)
            save_prediction(data_nd_d[:,0,0,:,:], target_nd_d[:,0,0,:,:], pred_nd_d[:,0,0,:,:], display_path2, default_as_0=False)
        if (iter_id+1) % cfg.MODEL.SAVE_ITER == 0:
            encoder_net.save_checkpoint(
                prefix=os.path.join(base_dir, "encoder_net",),
                epoch=iter_id,
                save_optimizer_states=True)
            forecaster_net.save_checkpoint(
                prefix=os.path.join(base_dir, "forecaster_net",),
                epoch=iter_id,
                save_optimizer_states=True)
            if cfg.MODEL.GAN_G_LAMBDA > 0:
                discrim_net.save_checkpoint(
                    prefix=os.path.join(base_dir, "discrim_net",),
                    epoch=iter_id,
                    save_optimizer_states=True)
            path1 = os.path.join(base_dir, 'train_loss_dicts.pkl')
            path2 = os.path.join(base_dir, 'valid_loss_dicts.pkl')
            with open(path1, 'wb') as f:
                pickle.dump(train_loss_dicts, f)
            with open(path2, 'wb') as f:
                pickle.dump(valid_loss_dicts, f)
        iter_id += 1
        
def valid_step(batch_size, encoder_net, forecaster_net,
               loss_net, discrim_net, loss_D_net, init_states,
               data_nd, gt_nd, mask_nd, valid_loss_dicts, iter_id=None):
    '''
    valid_loss_dicts: dict<list>
    '''
    init_states.reset_all()
    pred_nd = get_prediction(data_nd, init_states, encoder_net, forecaster_net)
    
    if cfg.MODEL.GAN_G_LAMBDA > 0:
        discrim_net.forward(data_batch=mx.io.DataBatch(data=[pred_nd]))
        discrim_output = discrim_net.get_outputs()[0]
    else:
        discrim_output = mx.nd.zeros((batch_size,))

    loss_net.forward(data_batch=mx.io.DataBatch(data=[pred_nd, discrim_output],
                                                label=[gt_nd, mask_nd]))

    if cfg.MODEL.GAN_G_LAMBDA > 0:                                         
        loss_D_net.forward(data_batch=mx.io.DataBatch(data=[discrim_output],
                                                label=[mx.nd.zeros_like(discrim_output)]))
        discrim_loss = mx.nd.mean(loss_D_net.get_outputs()[0]).asscalar()
        discrim_net.forward(data_batch=mx.io.DataBatch(data=[gt_nd]))
        discrim_output = discrim_net.get_outputs()[0]
        loss_D_net.forward(data_batch=mx.io.DataBatch(data=[discrim_output],
                                                    label=[mx.nd.ones_like(discrim_output)]))
        discrim_loss += mx.nd.mean(discrim_net.get_outputs()[0]).asscalar()
        discrim_loss = discrim_loss / 2
    new_valid_loss_dicts = {}
    for k in valid_loss_dicts.keys():
        if k == 'dis':
            if discrim_loss > 1.0:
                loss = 1.0
            else:        
                loss = discrim_loss
        elif k == 'gan':
            loss = mx.nd.mean(loss_net.get_output_dict()[k+'_output']).asscalar()
            if loss > 1.0:
                loss = 1.0
        else:
            loss = mx.nd.mean(loss_net.get_output_dict()[k+'_output']).asscalar()
        if loss < 0:
            loss = 0
        new_valid_loss_dicts[k] = loss
    loss_str = ", ".join(["%s=%g" %(k, v) for k, v in new_valid_loss_dicts.items()])
    logging.info("iter {}, validation, {}".format(iter_id, loss_str))

    return init_states, new_valid_loss_dicts, pred_nd


###
'''
2018.8.26 note
the change hasn't been complete
1 a new mode in parallel to "extend to full outlen" is needed
(or i should just abandon the flag cfg.MODEL.EXTEND_TO_FULL_OUTLEN )
2 when saving images, the naming should be considered(especially shift)
'''

def predict(args, num_samples, mode='display', extend='none', save_path=None):
    """
    mode can be either display or save
    under display mode, num_samples gifs and comparisons are saved.
    under save mode, num_samples sequence of pngs are saved in difference directories
    extend can be none, recursive or onetime
    """
    assert len(args.ctx) == 1
    base_dir = get_base_dir(args)
    if extend == 'recursive':
        iter_outlen = 30
        model_outlen = cfg.MODEL.OUT_LEN
    elif extend == 'onetime':
        iter_outlen = 30
        model_outlen = 30
    else:
        iter_outlen = cfg.MODEL.OUT_LEN
        model_outlen = cfg.MODEL.OUT_LEN
    szo_iterator = SZOIterator(rec_paths=cfg.SZO_TEST_DATA_PATHS,
                               in_len=cfg.MODEL.IN_LEN,
                               out_len=iter_outlen,
                               batch_size=1,
                               frame_skip=cfg.MODEL.FRAME_SKIP,
                               ctx=args.ctx)
    szo_nowcasting = SZONowcastingFactory(batch_size=1,
                                          ctx_num=1,
                                          in_seq_len=cfg.MODEL.IN_LEN,
                                          out_seq_len=model_outlen,
                                          frame_stack=cfg.MODEL.FRAME_STACK)

    encoder_net, forecaster_net, loss_net, discrim_net, loss_D_net = \
        encoder_forecaster_build_networks(
            factory=szo_nowcasting,
            context=args.ctx)
    encoder_net.summary()
    forecaster_net.summary()
    loss_net.summary()
    # load parameters
    # assume parameter files are available
    if args.resume:
        start_iter_id = latest_iter_id(base_dir)
        encoder_net.load_params(os.path.join(base_dir, 'encoder_net'+'-%04d.params'%(start_iter_id)))
        forecaster_net.load_params(os.path.join(base_dir, 'forecaster_net'+'-%04d.params'%(start_iter_id)))
    # initial states
    states = EncoderForecasterStates(factory=szo_nowcasting, ctx=args.ctx[0])
    # generate samples
    for i in range(num_samples):
        new_frame_dat, folder_names = szo_iterator.get_sample_name_pair(fix_shift=True)
        data_nd_d = new_frame_dat[0:cfg.MODEL.IN_LEN,:,:,:,:] / 255.0
        target_nd_d = new_frame_dat[cfg.MODEL.IN_LEN:,:,:,:,:] / 255.0
        states.reset_all()
        pred_nd_d1 = get_prediction(data_nd_d, states, encoder_net, forecaster_net)
        if extend == 'recursive':
            states.reset_all()
            pred_nd_d2 = get_prediction(pred_nd_d1[30-cfg.MODEL.IN_LEN-cfg.MODEL.OUT_LEN: 30-cfg.MODEL.OUT_LEN,:,:,:,:], 
                                        states, encoder_net, forecaster_net)
            pred_nd_d = mx.nd.concat(pred_nd_d1[:30-cfg.MODEL.OUT_LEN,:,:,:,:],pred_nd_d2, dim=0)
        else:
            pred_nd_d = pred_nd_d1
          
        data_nd_d = (data_nd_d*255.0).clip(0, 255.0)
        target_nd_d = (target_nd_d*255.0).clip(0, 255.0)
        pred_nd_d = (pred_nd_d*255.0).clip(0, 255.0)

        if mode == 'display':
            display_path1 = os.path.join(base_dir, 'prediction_'+str(i))
            display_path2 = os.path.join(base_dir, 'prediction_'+str(i)+'_')
            if not os.path.exists(display_path1):
                os.mkdir(display_path1)
            if not os.path.exists(display_path2):
                os.mkdir(display_path2)
            save_prediction(data_nd_d[:,0,0,:,:], target_nd_d[:,0,0,:,:], pred_nd_d[:,0,0,:,:], display_path1, default_as_0=True)
            save_prediction(data_nd_d[:,0,0,:,:], target_nd_d[:,0,0,:,:], pred_nd_d[:,0,0,:,:], display_path2, default_as_0=False)
            plt.hist(pred_nd_d.asnumpy().reshape([-1]), bins=100)
            plt.savefig(os.path.join(base_dir, 'hist'+str(i)))
            plt.close('all')
        elif mode == 'save':
            gt_path = os.path.join(save_path, 'groundtruth')
            pred_path = os.path.join(save_path, 'prediction')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if not os.path.exists(gt_path):
                os.mkdir(gt_path)
            if not os.path.exists(pred_path):
                os.mkdir(pred_path)
            folder_name = folder_names[0][-1]
            save_prediction(data_nd_d[:,0,0,:,:], target_nd_d[:,0,0,:,:], pred_nd_d[:,0,0,:,:], None, default_as_0=True, mode='save', folder_name=folder_name, gt_path=gt_path, pred_path=pred_path)
        else:
            raise NotImplementedError


def test(args, batches, checkpoint_id=None, on_train=False):
    if cfg.MODEL.EXTEND_TO_FULL_OUTLEN:
        outlen = 30
    else:
        outlen = cfg.MODEL.OUT_LEN
    evaluator = SZOEvaluation(outlen, False)
    base_dir = get_base_dir(args)
    logging.basicConfig(level=logging.INFO)
    save_cfg(dir_path=base_dir, source=cfg.MODEL)
    if on_train:
        szo_iter = SZOIterator(rec_paths=cfg.SZO_TRAIN_DATA_PATHS,
                                in_len=cfg.MODEL.IN_LEN,
                                out_len=outlen,
                                batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
                                frame_skip=cfg.MODEL.FRAME_SKIP,
                                ctx=args.ctx)
    else:
        szo_iter = SZOIterator(rec_paths=cfg.SZO_TEST_DATA_PATHS,
                                in_len=cfg.MODEL.IN_LEN,
                                out_len=outlen,
                                batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
                                frame_skip=cfg.MODEL.FRAME_SKIP,
                                ctx=args.ctx)

    szo_nowcasting = SZONowcastingFactory(batch_size=cfg.MODEL.TRAIN.BATCH_SIZE // len(args.ctx),
                                          ctx_num=len(args.ctx),
                                          in_seq_len=cfg.MODEL.IN_LEN,
                                          out_seq_len=cfg.MODEL.OUT_LEN,
                                          frame_stack=cfg.MODEL.FRAME_STACK)
    encoder_net, forecaster_net, loss_net, discrim_net, loss_D_net = \
        encoder_forecaster_build_networks(
            factory=szo_nowcasting,
            context=args.ctx)
    encoder_net.summary()
    forecaster_net.summary()
    loss_net.summary()
    # try to load checkpoint
    if checkpoint_id == None:
        start_iter_id = latest_iter_id(base_dir)
    else:
        start_iter_id = checkpoint_id
    encoder_net.load_params(os.path.join(base_dir, 'encoder_net'+'-%04d.params'%(start_iter_id)))
    forecaster_net.load_params(os.path.join(base_dir, 'forecaster_net'+'-%04d.params'%(start_iter_id)))
    states = EncoderForecasterStates(factory=szo_nowcasting, ctx=args.ctx[0])
    for i in range(batches):
        states.reset_all()
        frame_dat = szo_iter.sample()
        data_nd = frame_dat[0:cfg.MODEL.IN_LEN, :,:,:,:] / 255.0
        target_nd = frame_dat[cfg.MODEL.IN_LEN:(cfg.MODEL.IN_LEN + cfg.MODEL.OUT_LEN),:,:,:,:] / 255.0
        pred_nd1 = get_prediction(data_nd, states, encoder_net, forecaster_net)
        if cfg.MODEL.EXTEND_TO_FULL_OUTLEN:
            states.reset_all()
            pred_nd2 = get_prediction(pred_nd1[30-cfg.MODEL.IN_LEN-cfg.MODEL.OUT_LEN: 30-cfg.MODEL.OUT_LEN,:,:,:,:], 
                                        states, encoder_net, forecaster_net)
            pred_nd = mx.nd.concat(pred_nd1[:30-cfg.MODEL.OUT_LEN,:,:,:,:], pred_nd2, dim=0)
        else:
            pred_nd = pred_nd1

        # generate mask from target_nd
        if cfg.MODEL.ENCODER_FORECASTER.HAS_MASK:
            target_nd = target_nd * (255.0/80.0)
            pred_nd = pred_nd * (255.0/80.0)
        mask_nd = (target_nd == 0.0)
        evaluator.update(target_nd.asnumpy(), pred_nd.asnumpy(), mask_nd.asnumpy())
    evaluator.print_stat_readable()
    filename = 'test_result_%03d'%(start_iter_id)
    if on_train:
        filename += '_on_train.txt'
    else:
        filename += '.txt'
    evaluator.save_txt_readable(path=os.path.join(base_dir, filename))


if __name__ == "__main__":
    args = parse_args()
    train(args)
    #predict(args, 4, mode='save', extend='onetime', save_path='pred_result')
    #test(args, 200)    

