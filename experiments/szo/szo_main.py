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
from szo_factory import SZONowcastingFactory
from nowcasting.config import cfg, cfg_from_file, save_cfg
from nowcasting.my_module import MyModule
from nowcasting.encoder_forecaster import encoder_forecaster_build_networks, train_step, EncoderForecasterStates
from nowcasting.szo_evaluation import *
from nowcasting.utils import parse_ctx, logging_config, latest_iter_id
from nowcasting.szo_iterator import SZOIterator, save_png_sequence
from nowcasting.helpers.visualization import save_hko_gif


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

def save_prediction(data_nd, target_nd, pred_nd, path, convert_to_0_255=True):
    # shape (seqlen, batch_size=1, channel=1, height, width)
    # *255 and converted to numpy
    if not convert_to_0_255:
    # remain 0-80, uint8
        data_np = data_nd.asnumpy().astype(np.uint8)
        target_np = target_nd.asnumpy().astype(np.uint8)
        pred_np = pred_nd.asnumpy().astype(np.uint8)
    else:
    # convert 0-80 to 0-255, uint8
        scale_fac = 255.0 / cfg.SZO.DATA.RADAR_RANGE
        data_np = data_nd.asnumpy()
        target_np = target_nd.asnumpy()
        pred_np = pred_nd.asnumpy()
        data_np = data_np * (data_np<cfg.SZO.DATA.RADAR_RANGE)
        target_np = target_np * (target_np<cfg.SZO.DATA.RADAR_RANGE)
        pred_np = pred_np * (pred_np<cfg.SZO.DATA.RADAR_RANGE)
        data_np = (data_np * scale_fac).astype(np.uint8)
        target_np = (target_np * scale_fac).astype(np.uint8)
        pred_np = (pred_np * scale_fac).astype(np.uint8)
    
    inputs = data_np
    ground_truth = target_np
    predictions = pred_np
    
    gif_true = np.concatenate([data_np, target_np], axis=0)
    gif_generated = np.concatenate([data_np, pred_np], axis=0)
    
    save_png_sequence([inputs, ground_truth, predictions], os.path.join(path, 'framewise_comp.png'))
    
    save_hko_gif(gif_true, os.path.join(path, "true.gif"), multiply_by_255=False)
    save_hko_gif(gif_generated, os.path.join(path, "generated.gif"), multiply_by_255=False)
    

def plot_loss_curve(path, losses):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MaxNLocator(nbins=10)
    ax.yaxis.set_major_locator(loc)
    plt.plot(losses)
    plt.savefig(path)
    plt.close('all')

def train(args):
    base_dir = get_base_dir(args)
    logging_config(folder=base_dir)
    save_cfg(dir_path=base_dir, source=cfg.MODEL)
    if cfg.MODEL.TRAIN.TBPTT:
        # Create a set of sequent iterators with different starting point
        raise NotImplementedError
        '''
        train_szo_iters = []
        train_szo_iter_restart = []
        for _ in range(cfg.MODEL.TRAIN.BATCH_SIZE):
            ele_iter = SZOIterator(path=cfg.SZO_TRAIN_DATA_PATH,
                                   seq_len=cfg.SZO.ITERATOR.TRAIN_IN_LEN
                                          +cfg.SZO.ITERATOR.TRAIN_OUT_LEN)
            ele_iter.random_reset()
            train_szo_iter_restart.append(True)
            train_szo_iters.append(ele_iter)
        '''
    else:
        train_szo_iter = SZOIterator(rec_paths=cfg.SZO_TRAIN_DATA_PATHS,
                                   in_len=cfg.MODEL.IN_LEN,
                                   out_len=cfg.MODEL.OUT_LEN,
                                   batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
                                   ctx=args.ctx)
        valid_szo_iter = SZOIterator(rec_paths=cfg.SZO_TEST_DATA_PATHS,
                                     in_len=cfg.MODEL.IN_LEN,
                                     out_len=cfg.MODEL.OUT_LEN,
                                     batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
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
    discrim_net.summary()
    loss_D_net.summary()
    # try to load checkpoint
    if args.resume:
        start_iter_id = latest_iter_id(base_dir)
        '''
        encoder_net.load(prefix=os.path.join(base_dir, 'encoder_net'),
                         epoch=start_iter_id,
                         load_optimizer_states=True,
                         data_names=[ele.name for ele in szo_nowcasting.encoder_data_desc()],
                         label_names=[],
                         context=args.ctx[0])
        forecaster_net.load(prefix=os.path.join(base_dir, 'forecaster_net'),
                         epoch=start_iter_id,
                         load_optimizer_states=True,
                         data_names=[ele.name for ele in szo_nowcasting.forecaster_data_desc()],
                         label_names=[],
                         context=args.ctx[0])
        '''
        encoder_net.load_params(os.path.join(base_dir, 'encoder_net'+'-%04d.params'%(start_iter_id)))
        forecaster_net.load_params(os.path.join(base_dir, 'forecaster_net'+'-%04d.params'%(start_iter_id)))
        discrim_net.load_params(os.path.join(base_dir, 'discrim_net'+'-%04d.params'%(start_iter_id)))
        encoder_net.load_optimizer_states(os.path.join(base_dir, 'encoder_net'+'-%04d.states'%(start_iter_id)))
        forecaster_net.load_optimizer_states(os.path.join(base_dir, 'forecaster_net'+'-%04d.states'%(start_iter_id)))
        discrim_net.load_optimizer_states(os.path.join(base_dir, 'discrim_net'+'-%04d.states'%(start_iter_id)))
        with open(os.path.join(base_dir, 'train_loss_dicts.pkl'), 'rb') as f:
            train_loss_dicts = pickle.load(f)
        with open(os.path.join(base_dir, 'valid_loss_dicts.pkl'), 'rb') as f:
            valid_loss_dicts = pickle.load(f)
    else:
        start_iter_id = 0
        train_loss_dicts = {}
        valid_loss_dicts = {}
        for dicts in (train_loss_dicts, valid_loss_dicts):
            for typ in ('mse','gdl','gan','dis'):
                dicts[typ] = []

    states = EncoderForecasterStates(factory=szo_nowcasting, ctx=args.ctx[0])
    for info in szo_nowcasting.init_encoder_state_info:
        assert info["__layout__"].find('N') == 0, "Layout=%s is not supported!" %info["__layout__"]
    for info in szo_nowcasting.init_forecaster_state_info:
        assert info["__layout__"].find('N') == 0, "Layout=%s is not supported!" % info["__layout__"]

    cumulative_loss = {}
    for k in train_loss_dicts.keys():
        cumulative_loss[k] = 0.0

    iter_id = start_iter_id
    while iter_id < cfg.MODEL.TRAIN.MAX_ITER:
        if not cfg.MODEL.TRAIN.TBPTT:
            # We are not using TBPTT, we could directly sample a random minibatch
            frame_dat = train_szo_iter.sample()
            states.reset_all()
        else:
            # We are using TBPTT, we should sample minibatches from the iterators.
            raise NotImplementedError
            '''
            frame_dat_l = []
            mask_dat_l = []
            for i, ele_iter in enumerate(train_szo_iters):
                if ele_iter.use_up:
                    states.reset_batch(batch_id=i)
                    ele_iter.random_reset()
                    train_hko_iter_restart[i] = True
                if train_hko_iter_restart[i] == False and ele_iter.check_new_start():
                    states.reset_batch(batch_id=i)
                    ele_iter.random_reset()
                frame_dat, mask_dat, datetime_clips, _ = \
                    ele_iter.sample(batch_size=cfg.MODEL.TRAIN.BATCH_SIZE)
                train_hko_iter_restart[i] = False
                frame_dat_l.append(frame_dat)
                mask_dat_l.append(mask_dat)
            frame_dat = np.concatenate(frame_dat_l, axis=1)
            mask_dat = np.concatenate(mask_dat_l, axis=1)
            '''
        data_nd = frame_dat[0:cfg.MODEL.IN_LEN,:,:,:,:] / 255.0  # scale to [0,1]
        target_nd = frame_dat[cfg.MODEL.IN_LEN:(cfg.MODEL.IN_LEN + cfg.MODEL.OUT_LEN),:,:,:,:] / 255.0
        mask_nd = mx.nd.ones_like(target_nd)
        states, loss_dict = train_step(batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
                               encoder_net=encoder_net, forecaster_net=forecaster_net,
                               loss_net=loss_net, discrim_net=discrim_net, 
                               loss_D_net=loss_D_net, init_states=states,
                               data_nd=data_nd, gt_nd=target_nd, mask_nd=mask_nd,
                               iter_id=iter_id)
        for k in cumulative_loss.keys():
            loss = loss_dict[k+'_output']
            cumulative_loss[k] += loss

        if (iter_id+1) % cfg.MODEL.VALID_ITER == 0:
            frame_dat_v = valid_szo_iter.sample()
            data_nd_v = frame_dat_v[0:cfg.MODEL.IN_LEN,:,:,:,:] / 255.0
            gt_nd_v = frame_dat_v[cfg.MODEL.IN_LEN:(cfg.MODEL.IN_LEN+cfg.MODEL.OUT_LEN),:,:,:,:] / 255.0
            mask_nd_v = mx.nd.ones_like(gt_nd_v)
            states.reset_all()
            pred_nd_v = get_prediction(data_nd_v, states, encoder_net, forecaster_net)
            discrim_net.forward(data_batch=mx.io.DataBatch(data=[pred_nd_v]))
            discrim_output = discrim_net.get_outputs()[0]
            loss_net.forward(data_batch=mx.io.DataBatch(data=[pred_nd_v, discrim_output],
                                                        label=[gt_nd_v, mask_nd_v]))
            loss_D_net.forward(data_batch=mx.io.DataBatch(data=[discrim_output],
                                                        label=[mx.nd.zeros_like(discrim_output)]))
            discrim_loss = mx.nd.mean(loss_D_net.get_outputs()[0]).asscalar()
            discrim_net.forward(data_batch=mx.io.DataBatch(data=[gt_nd_v]))
            discrim_output = discrim_net.get_outputs()[0]
            loss_D_net.forward(data_batch=mx.io.DataBatch(data=[discrim_output],
                                                        label=[mx.nd.ones_like(discrim_output)]))
            discrim_loss += mx.nd.mean(discrim_net.get_outputs()[0]).asscalar()
            discrim_loss = discrim_loss / 2
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
                valid_loss_dicts[k].append(loss)
            loss_str = ", ".join(["%s=%g" %(k, v[-1]) for k, v in valid_loss_dicts.items()])
            logging.info("iter {}, validation, {}".format(iter_id, loss_str))
            for k in valid_loss_dicts.keys():
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
            data_nd = new_frame_dat[0:cfg.MODEL.IN_LEN,:,:,:,:] / 255.0
            target_nd = new_frame_dat[cfg.MODEL.IN_LEN:(cfg.MODEL.IN_LEN+cfg.MODEL.OUT_LEN),:,:,:,:] / 255.0
            states.reset_all()
            pred_nd = get_prediction(data_nd, states, encoder_net, forecaster_net)
            
            display_path1 = os.path.join(base_dir, 'display_'+str(iter_id))
            display_path2 = os.path.join(base_dir, 'display_'+str(iter_id)+'_')
            if not os.path.exists(display_path1):
                os.mkdir(display_path1)
            if not os.path.exists(display_path2):
                os.mkdir(display_path2)

            data_nd = (data_nd*255.0).clip(0, 255.0)
            target_nd = (target_nd*255.0).clip(0, 255.0)
            pred_nd = (pred_nd*255.0).clip(0, 255.0)
            save_prediction(data_nd[:,0,0,:,:], target_nd[:,0,0,:,:], pred_nd[:,0,0,:,:], display_path1)
            save_prediction(data_nd[:,0,0,:,:], target_nd[:,0,0,:,:], pred_nd[:,0,0,:,:], display_path2, convert_to_0_255=False)
        '''
        if (iter_id+1) % cfg.MODEL.TEMP_SAVE_ITER == 0:
            epoch = (iter_id//cfg.MODEL.SAVE_ITER)*cfg.MODEL.SAVE_ITER - 1
            encoder_net.save_checkpoint(
                prefix=os.path.join(base_dir, "encoder_net",),
                epoch=epoch,
                save_optimizer_states=True)
            forecaster_net.save_checkpoint(
                prefix=os.path.join(base_dir, "forecaster_net",),
                epoch=epoch,
                save_optimizer_states=True)
            path = os.path.join(base_dir, 'loss_dicts.pkl')
            with open(path, 'wb') as f:
                loss_dicts = {'train_mse':train_mse_losses, 'train_gdl':train_gdl_losses,
                              'valid_mse':valid_mse_losses, 'valid_gdl':valid_gdl_losses}
                pickle.dump(loss_dicts, f)
        '''
        if (iter_id+1) % cfg.MODEL.SAVE_ITER == 0:
            encoder_net.save_checkpoint(
                prefix=os.path.join(base_dir, "encoder_net",),
                epoch=iter_id,
                save_optimizer_states=True)
            forecaster_net.save_checkpoint(
                prefix=os.path.join(base_dir, "forecaster_net",),
                epoch=iter_id,
                save_optimizer_states=True)
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
        
def predict(args, num_samples):
    assert cfg.MODEL.FRAME_STACK == 1 and cfg.MODEL.FRAME_SKIP == 1
    assert len(args.ctx) == 1
    base_dir = get_base_dir(args)
    szo_iterator = SZOIterator(rec_paths=cfg.SZO_TRAIN_DATA_PATHS,
                               in_len=cfg.MODEL.IN_LEN,
                               out_len=cfg.MODEL.OUT_LEN,
                               batch_size=1,
                               ctx=args.ctx)
    szo_nowcasting = SZONowcastingFactory(batch_size=1,
                                          ctx_num=1,
                                          in_seq_len=cfg.MODEL.IN_LEN,
                                          out_seq_len=cfg.MODEL.OUT_LEN)

    encoder_net, forecaster_net, loss_net = \
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
        frame_dat = szo_iterator.sample()
        states.reset_all()
        data_nd = frame_dat[0:cfg.MODEL.IN_LEN, :,:,:,:] / 255.0
        target_nd = frame_dat[cfg.MODEL.IN_LEN:(cfg.MODEL.IN_LEN + cfg.MODEL.OUT_LEN),:,:,:,:] / 255.0
        pred_nd = get_prediction(data_nd, states, encoder_net, forecaster_net)

        display_path1 = os.path.join(base_dir, 'display_'+str(i))
        display_path2 = os.path.join(base_dir, 'display_'+str(i)+'_')
        if not os.path.exists(display_path1):
            os.mkdir(display_path1)
        if not os.path.exists(display_path2):
            os.mkdir(display_path2)
        
        data_nd = (data_nd*255.0).clip(0, 255.0)
        target_nd = (target_nd*255.0).clip(0, 255.0)
        pred_nd = (pred_nd*255.0).clip(0, 255.0)
        save_prediction(data_nd[:,0,0,:,:], target_nd[:,0,0,:,:], pred_nd[:,0,0,:,:], display_path1)
        save_prediction(data_nd[:,0,0,:,:], target_nd[:,0,0,:,:], pred_nd[:,0,0,:,:], display_path2, convert_to_0_255=False)
        plt.hist(pred_nd.asnumpy().reshape([-1]), bins=100)
        plt.savefig(os.path.join(base_dir, 'hist'+str(i)))
        plt.close()
        



if __name__ == "__main__":
    args = parse_args()
    train(args)
    #predict(args, 10)
