import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,*([".."]*2))))
import mxnet as mx
import mxnet.ndarray as nd
import nowcasting.config as cfg
from nowcasting.ops import reset_regs
from nowcasting.operators.common import grid_generator
from nowcasting.operators import *
from nowcasting.ops import *
from nowcasting.prediction_base_factory import PredictionBaseFactory
from nowcasting.operators.transformations import DFN
from nowcasting.my_module import MyModule


def get_encoder_forecaster_rnn_blocks(batch_size):
    encoder_rnn_blocks = []
    forecaster_rnn_blocks = []
    gan_rnn_blocks = []
    CONFIG = cfg.MODEL.ENCODER_FORECASTER.RNN_BLOCKS
    for vec, block_prefix in [(encoder_rnn_blocks, "ebrnn"),
                              (forecaster_rnn_blocks, "fbrnn"),
                              (gan_rnn_blocks, "dbrnn")]:
        for i in range(len(CONFIG.NUM_FILTER)):
            name = "%s%d" % (block_prefix, i + 1)
            if CONFIG.LAYER_TYPE[i] == "ConvGRU":
                rnn_block = BaseStackRNN(base_rnn_class=ConvGRU,
                                         stack_num=CONFIG.STACK_NUM[i],
                                         name=name,
                                         residual_connection=CONFIG.RES_CONNECTION,
                                         num_filter=CONFIG.NUM_FILTER[i],
                                         b_h_w=(batch_size,
                                                cfg.MODEL.ENCODER_FORECASTER.FEATMAP_SIZE[i],
                                                cfg.MODEL.ENCODER_FORECASTER.FEATMAP_SIZE[i]),
                                         h2h_kernel=CONFIG.H2H_KERNEL[i],
                                         h2h_dilate=CONFIG.H2H_DILATE[i],
                                         i2h_kernel=CONFIG.I2H_KERNEL[i],
                                         i2h_pad=CONFIG.I2H_PAD[i],
                                         act_type=cfg.MODEL.RNN_ACT_TYPE)
            elif CONFIG.LAYER_TYPE[i] == "TrajGRU":
                rnn_block = BaseStackRNN(base_rnn_class=TrajGRU,
                                         stack_num=CONFIG.STACK_NUM[i],
                                         name=name,
                                         L=CONFIG.L[i],
                                         residual_connection=CONFIG.RES_CONNECTION,
                                         num_filter=CONFIG.NUM_FILTER[i],
                                         b_h_w=(batch_size,
                                                cfg.MODEL.ENCODER_FORECASTER.FEATMAP_SIZE[i],
                                                cfg.MODEL.ENCODER_FORECASTER.FEATMAP_SIZE[i]),
                                         h2h_kernel=CONFIG.H2H_KERNEL[i],
                                         h2h_dilate=CONFIG.H2H_DILATE[i],
                                         i2h_kernel=CONFIG.I2H_KERNEL[i],
                                         i2h_pad=CONFIG.I2H_PAD[i],
                                         act_type=cfg.MODEL.RNN_ACT_TYPE)
            else:
                raise NotImplementedError
            vec.append(rnn_block)
    return encoder_rnn_blocks, forecaster_rnn_blocks, gan_rnn_blocks

class EncoderForecasterBaseFactory(PredictionBaseFactory):
    def __init__(self,
                 batch_size,
                 in_seq_len,
                 out_seq_len,
                 height,
                 width,
                 frame_stack=1,
                 ctx_num=1,
                 name="encoder_forecaster"):
        super(EncoderForecasterBaseFactory, self).__init__(batch_size=batch_size,
                                                           ctx_num=ctx_num,
                                                           in_seq_len=in_seq_len,
                                                           out_seq_len=out_seq_len,
                                                           frame_stack=frame_stack,
                                                           height=height,
                                                           width=width,
                                                           name=name)

    def _init_rnn(self):
        self._encoder_rnn_blocks, self._forecaster_rnn_blocks, self._gan_rnn_blocks =\
            get_encoder_forecaster_rnn_blocks(batch_size=self._batch_size)
        return self._encoder_rnn_blocks + self._forecaster_rnn_blocks + self._gan_rnn_blocks

    @property
    def init_encoder_state_info(self):
        init_state_info = []
        for block in self._encoder_rnn_blocks:
            for state in block.init_state_vars():
                init_state_info.append({'name': state.name,
                                        'shape': state.attr('__shape__'),
                                        '__layout__': state.list_attr()['__layout__']})
        return init_state_info

    @property
    def init_forecaster_state_info(self):
        init_state_info = []
        for block in self._forecaster_rnn_blocks:
            for state in block.init_state_vars():
                init_state_info.append({'name': state.name,
                                        'shape': state.attr('__shape__'),
                                        '__layout__': state.list_attr()['__layout__']})
        return init_state_info

    '''
    @property
    def init_gan_state_info(self):
        init_gan_state_info = []
        for block in self._gan_rnn_blocks:
            for state in block.init_state_vars():
                init_gan_state_info.append({'name': state.name,
                                            'shape': state.attr('__shape__'),
                                            '__layout__': state.list_attr()['__layout__']})
        return init_gan_state_info
    '''

    def stack_rnn_encode(self, data):
        assert self._in_seq_len % self._frame_stack == 0, 'frame_stack cannot devide seq_len'
        CONFIG = cfg.MODEL.ENCODER_FORECASTER
        pre_encoded_data = self._pre_encode_frame(frame_data=data, seqlen=self._in_seq_len, frame_stack=self._frame_stack)
        reshape_data = mx.sym.Reshape(pre_encoded_data, shape=(-1, 0, 0, 0), reverse=True)

        # Encoder Part
        conv1 = conv2d_act(data=reshape_data,
                           num_filter=CONFIG.FIRST_CONV1[0],
                           kernel=(CONFIG.FIRST_CONV1[1], CONFIG.FIRST_CONV1[1]),
                           stride=(CONFIG.FIRST_CONV1[2], CONFIG.FIRST_CONV1[2]),
                           pad=(CONFIG.FIRST_CONV1[3], CONFIG.FIRST_CONV1[3]),
                           act_type=cfg.MODEL.CNN_ACT_TYPE,
                           name="econv1")

        conv2 = conv2d_act(data=conv1,
                           num_filter=CONFIG.FIRST_CONV2[0],
                           kernel=(CONFIG.FIRST_CONV2[1], CONFIG.FIRST_CONV2[1]),
                           stride=(CONFIG.FIRST_CONV2[2], CONFIG.FIRST_CONV2[2]),
                           pad=(CONFIG.FIRST_CONV2[3], CONFIG.FIRST_CONV2[3]),
                           act_type=cfg.MODEL.CNN_ACT_TYPE,
                           name="econv2")
        
        rnn_block_num = len(CONFIG.RNN_BLOCKS.NUM_FILTER)
        encoder_rnn_block_states = []
        for i in range(rnn_block_num):
            if i == 0:
                inputs = conv2
            else:
                inputs = downsample
            rnn_out, states = self._encoder_rnn_blocks[i].unroll(
                length=self._in_seq_len//self._frame_stack,
                inputs=inputs,
                begin_states=None,
                ret_mid=False)
            encoder_rnn_block_states.append(states)
            if i < rnn_block_num - 1:
                downsample = downsample_module(data=rnn_out[-1],
                                               num_filter=CONFIG.RNN_BLOCKS.NUM_FILTER[i + 1],
                                               kernel=(CONFIG.DOWNSAMPLE[i][0],
                                                       CONFIG.DOWNSAMPLE[i][0]),
                                               stride=(CONFIG.DOWNSAMPLE[i][1],
                                                       CONFIG.DOWNSAMPLE[i][1]),
                                               pad=(CONFIG.DOWNSAMPLE[i][2],
                                                    CONFIG.DOWNSAMPLE[i][2]),
                                               b_h_w=(self._batch_size,
                                                      CONFIG.FEATMAP_SIZE[i + 1],
                                                      CONFIG.FEATMAP_SIZE[i + 1]),
                                               name="edown%d" %(i + 1))
        return encoder_rnn_block_states

    def stack_rnn_forecast(self, block_state_list, last_frame):
        assert self._out_seq_len % self._frame_stack == 0
        CONFIG = cfg.MODEL.ENCODER_FORECASTER
        block_state_list = [self._forecaster_rnn_blocks[i].to_split(block_state_list[i])
                            for i in range(len(self._forecaster_rnn_blocks))]
        rnn_block_num = len(CONFIG.RNN_BLOCKS.NUM_FILTER)
        rnn_block_outputs = []
        # RNN Forecaster Part
        curr_inputs = None
        for i in range(rnn_block_num - 1, -1, -1):
            rnn_out, rnn_state = self._forecaster_rnn_blocks[i].unroll(
                length=self._out_seq_len//self._frame_stack, inputs=curr_inputs,
                begin_states=block_state_list[i][::-1],  # Reverse the order of states for the forecaster
                ret_mid=False)
            rnn_block_outputs.append(rnn_out)
            if i > 0:
                upsample = upsample_module(data=rnn_out[-1],
                                           num_filter=CONFIG.RNN_BLOCKS.NUM_FILTER[i],
                                           kernel=(CONFIG.UPSAMPLE[i - 1][0],
                                                   CONFIG.UPSAMPLE[i - 1][0]),
                                           stride=(CONFIG.UPSAMPLE[i - 1][1],
                                                   CONFIG.UPSAMPLE[i - 1][1]),
                                           pad=(CONFIG.UPSAMPLE[i - 1][2],
                                                CONFIG.UPSAMPLE[i - 1][2]),
                                           b_h_w=(self._batch_size, CONFIG.FEATMAP_SIZE[i - 1]),
                                           name="fup%d" %i)
                curr_inputs = upsample
        # Output
        if cfg.MODEL.OUT_TYPE == "DFN":
            concat_fbrnn1_out = mx.sym.concat(*rnn_out[-1], dim=0)
            dynamic_filter = deconv2d(data=concat_fbrnn1_out,
                                      num_filter=121,
                                      kernel=(CONFIG.LAST_DECONV[1], CONFIG.LAST_DECONV[1]),
                                      stride=(CONFIG.LAST_DECONV[2], CONFIG.LAST_DECONV[2]),
                                      pad=(CONFIG.LAST_DECONV[3], CONFIG.LAST_DECONV[3]))
            flow = dynamic_filter
            dynamic_filter = mx.sym.SliceChannel(dynamic_filter, axis=0, num_outputs=self._out_seq_len//self._frame_stack)
            prev_frame = last_frame
            preds = []
            for i in range(self._out_seq_len//self._frame_stack):
                pred_ele = DFN(data=prev_frame, local_kernels=dynamic_filter[i], K=11, batch_size=self._batch_size)
                preds.append(pred_ele)
                prev_frame = pred_ele
            pred = mx.sym.concat(*preds, dim=0)
        elif cfg.MODEL.OUT_TYPE == "direct":
            flow = None
            deconv2 = deconv2d_act(data=mx.sym.concat(*rnn_out[-1], dim=0),
                                   num_filter=CONFIG.LAST_DECONV2[0],
                                   kernel=(CONFIG.LAST_DECONV2[1], CONFIG.LAST_DECONV2[1]),
                                   stride=(CONFIG.LAST_DECONV2[2], CONFIG.LAST_DECONV2[2]),
                                   pad=(CONFIG.LAST_DECONV2[3], CONFIG.LAST_DECONV2[3]),
                                   act_type=cfg.MODEL.CNN_ACT_TYPE,
                                   name="fdeconv2")
            deconv1 = deconv2d_act(data=deconv2,
                                   num_filter=CONFIG.LAST_DECONV1[0],
                                   kernel=(CONFIG.LAST_DECONV1[1], CONFIG.LAST_DECONV1[1]),
                                   stride=(CONFIG.LAST_DECONV1[2], CONFIG.LAST_DECONV1[2]),
                                   pad=(CONFIG.LAST_DECONV1[3], CONFIG.LAST_DECONV1[3]),
                                   act_type=cfg.MODEL.CNN_ACT_TYPE,
                                   name="fdeconv1")
            if cfg.MODEL.ENCODER_FORECASTER.USE_SKIP:
                last_frame = mx.sym.repeat(last_frame, repeats=self._frame_stack, axis=1)
                last_frame_repeat = mx.sym.repeat(last_frame, repeats=self._out_seq_len//self._frame_stack, axis=0)
                concated_layer = mx.sym.concat(deconv1, last_frame_repeat, dim=1)
            else:
                concated_layer = deconv1
            conv_final = conv2d_act(data=concated_layer,
                                    num_filter=CONFIG.LAST_DECONV1[0],
                                    kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                    act_type=cfg.MODEL.CNN_ACT_TYPE, name="conv_final")
            pred = conv2d(data=conv_final,
                          num_filter=self._frame_stack, kernel=(1, 1), name="out")

        else:
            raise NotImplementedError
        #print(pred.infer_shape(fbrnn1_begin_state_h=(2, 64, 83, 83),fbrnn2_begin_state_h=(2, 192, 41, 41),fbrnn3_begin_state_h=(2, 192, 20, 20)))
        pred = mx.sym.Reshape(pred,
                              shape=(self._out_seq_len//self._frame_stack, self._batch_size,
                                     self._frame_stack, self._height, self._width),
                              __layout__="TNCHW")
        #print(pred.infer_shape(fbrnn1_begin_state_h=(2, 64, 83, 83),fbrnn2_begin_state_h=(2, 192, 41, 41),fbrnn3_begin_state_h=(2, 192, 20, 20)))
        pred = pred.transpose([1,0,2,3,4])
        pred = pred.reshape([self._batch_size, self._out_seq_len, 1, self._height, self._width])
        pred = pred.transpose([1,0,2,3,4])
        # for safety
        pred = mx.sym.Reshape(pred,
                              shape=(self._out_seq_len, self._batch_size,
                                     1, self._height, self._width),
                              __layout__="TNCHW")
        

        return pred, flow   

    def encoder_sym(self):
        self.reset_all()
        data = mx.sym.Variable('data')  # Shape: (in_seq_len, batch_size, C, H, W)
        block_state_list = self.stack_rnn_encode(data=data)
        states = []
        for i, rnn_block in enumerate(self._encoder_rnn_blocks):
            states.extend(rnn_block.flatten_add_layout(block_state_list[i]))
        return mx.sym.Group(states)

    def encoder_data_desc(self):
        ret = list()
        ret.append(mx.io.DataDesc(name='data',
                                  shape=(self._in_seq_len,
                                         self._batch_size * self._ctx_num,
                                         1,
                                         self._height,
                                         self._width),
                                  layout="TNCHW"))
        for info in self.init_encoder_state_info:
            state_shape = safe_eval(info['shape'])
            assert info['__layout__'].find('N') == 0,\
                "Layout=%s is not supported!" %info["__layout__"]
            state_shape = (state_shape[0] * self._ctx_num, ) + state_shape[1:]
            ret.append(mx.io.DataDesc(name=info['name'],
                                      shape=state_shape,
                                      layout=info['__layout__']))
        return ret

    def forecaster_sym(self):
        self.reset_all()
        block_state_list = []
        for block in self._forecaster_rnn_blocks:
            block_state_list.append(block.init_state_vars())

        if cfg.MODEL.OUT_TYPE == "direct":
            last_frame = mx.sym.Variable('last_frame')
            pred, _ = self.stack_rnn_forecast(block_state_list=block_state_list,
                                            last_frame=last_frame)
            return mx.sym.Group([pred])
        else:
            last_frame = mx.sym.Variable('last_frame')  # Shape: (batch_size, C, H, W)
            pred, flow = self.stack_rnn_forecast(block_state_list=block_state_list,
                                                 last_frame=last_frame)
            return mx.sym.Group([pred, mx.sym.BlockGrad(flow)])

    def forecaster_data_desc(self):
        ret = list()
        for info in self.init_forecaster_state_info:
            state_shape = safe_eval(info['shape'])
            assert info['__layout__'].find('N') == 0, \
                "Layout=%s is not supported!" % info["__layout__"]
            state_shape = (state_shape[0] * self._ctx_num,) + state_shape[1:]
            ret.append(mx.io.DataDesc(name=info['name'],
                                      shape=state_shape,
                                      layout=info['__layout__']))
        if cfg.MODEL.ENCODER_FORECASTER.USE_SKIP:
            ret.append(mx.io.DataDesc(name="last_frame",
                                    shape=(self._ctx_num * self._batch_size,
                                            1, self._height, self._width),
                                    layout="NCHW"))
        return ret



    def discriminator(self):
        raise NotImplementedError

    def discrim_sym(self):
        return self.discriminator()

    def discrim_data_desc(self):
        ret = list()
        ret.append(mx.io.DataDesc(name='video',
                                  shape=(self._out_seq_len,
                                         self._ctx_num * self._batch_size,
                                         1,
                                         self._height,
                                         self._width),
                                  layout="TNCHW"))
        return ret

    def loss_sym(self):
        raise NotImplementedError

    def loss_data_desc(self):
        ret = list()
        ret.append(mx.io.DataDesc(name='pred',
                                  shape=(self._out_seq_len,
                                         self._ctx_num * self._batch_size,
                                         1,
                                         self._height,
                                         self._width),
                                  layout="TNCHW"))

        ret.append(mx.io.DataDesc(name='discrim_out',
                                  shape=(self._ctx_num * self._batch_size,),
                                  layout="N"))
        return ret

    def loss_label_desc(self):
        ret = list()
        ret.append(mx.io.DataDesc(name='target',
                                  shape=(self._out_seq_len,
                                         self._ctx_num * self._batch_size,
                                         1,
                                         self._height,
                                         self._width),
                                  layout="TNCHW"))
        if cfg.MODEL.ENCODER_FORECASTER.HAS_MASK:
            ret.append(mx.io.DataDesc(name='mask',
                                      shape=(self._out_seq_len,
                                             self._ctx_num * self._batch_size,
                                             1,
                                             self._height,
                                             self._width),
                                      layout="TNCHW"))

        return ret

    def loss_D_sym(self):
        raise NotImplementedError

    def loss_D_data_desc(self):
        ret = list()
        ret.append(mx.io.DataDesc(name='discrim_out',
                                  shape=(self._ctx_num * self._batch_size,),
                                  layout="N"))
        return ret
    
    def loss_D_label_desc(self):
        ret = list()
        ret.append(mx.io.DataDesc(name='discrim_label',
                                  shape=(self._ctx_num * self._batch_size,),
                                  layout="N"))
        return ret
    


def init_optimizer_using_cfg(net, for_finetune, lr=cfg.MODEL.TRAIN.LR, min_lr=cfg.MODEL.TRAIN.MIN_LR, lr_decay_iter=cfg.MODEL.TRAIN.LR_DECAY_ITER,  lr_decay_factor=cfg.MODEL.TRAIN.LR_DECAY_FACTOR, optimizer_type=None):
    if optimizer_type is None:
        optimizer_type = cfg.MODEL.TRAIN.OPTIMIZER.lower()
    if not for_finetune:
        lr_scheduler = mx.lr_scheduler.FactorScheduler(step=lr_decay_iter,
                                                       factor=lr_decay_factor,
                                                       stop_factor_lr=min_lr)
        if optimizer_type == "adam":
            net.init_optimizer(optimizer="adam",
                               optimizer_params={'learning_rate': lr,
                                                 'beta1': cfg.MODEL.TRAIN.BETA1,
                                                 'rescale_grad': 1.0,
                                                 'epsilon': cfg.MODEL.TRAIN.EPS,
                                                 'lr_scheduler': lr_scheduler,
                                                 'wd': cfg.MODEL.TRAIN.WD})
        elif optimizer_type == "rmsprop":
            net.init_optimizer(optimizer="rmsprop",
                               optimizer_params={'learning_rate': lr,
                                                 'gamma1': cfg.MODEL.TRAIN.GAMMA1,
                                                 'rescale_grad': 1.0,
                                                 'epsilon': cfg.MODEL.TRAIN.EPS,
                                                 'lr_scheduler': lr_scheduler,
                                                 'wd': cfg.MODEL.TRAIN.WD})
        elif optimizer_type == "sgd":
            net.init_optimizer(optimizer="sgd",
                               optimizer_params={'learning_rate': lr,
                                                 'momentum': 0.0,
                                                 'rescale_grad': 1.0,
                                                 'lr_scheduler': lr_scheduler,
                                                 'wd': cfg.MODEL.TRAIN.WD})
        elif optimizer_type == "adagrad":
            net.init_optimizer(optimizer="adagrad",
                               optimizer_params={'learning_rate': lr,
                                                 'eps': cfg.MODEL.TRAIN.EPS,
                                                 'rescale_grad': 1.0,
                                                 'wd': cfg.MODEL.TRAIN.WD})
        else:
            raise NotImplementedError
    else:
        if optimizer_type == "adam":
            net.init_optimizer(optimizer="adam",
                               optimizer_params={'learning_rate': lr,
                                                 'beta1': cfg.MODEL.TEST.ONLINE.BETA1,
                                                 'rescale_grad': 1.0,
                                                 'epsilon': cfg.MODEL.TEST.ONLINE.EPS,
                                                 'wd': cfg.MODEL.TEST.ONLINE.WD})
        elif optimizer_type == "rmsprop":
            net.init_optimizer(optimizer="rmsprop",
                               optimizer_params={'learning_rate': lr,
                                                 'gamma1': cfg.MODEL.TEST.ONLINE.GAMMA1,
                                                 'rescale_grad': 1.0,
                                                 'epsilon': cfg.MODEL.TEST.ONLINE.EPS,
                                                 'wd': cfg.MODEL.TEST.ONLINE.WD})
        elif optimizer_type == "sgd":
            net.init_optimizer(optimizer="sgd",
                               optimizer_params={'learning_rate': lr,
                                                 'momentum': 0.0,
                                                 'rescale_grad': 1.0,
                                                 'wd': cfg.MODEL.TEST.ONLINE.WD})
        elif optimizer_type == "adagrad":
            net.init_optimizer(optimizer="adagrad",
                               optimizer_params={'learning_rate': lr,
                                                 'eps': cfg.MODEL.TRAIN.EPS,
                                                 'rescale_grad': 1.0,
                                                 'wd': cfg.MODEL.TEST.ONLINE.WD})
    return net


def encoder_forecaster_build_networks(factory, context,
                                      shared_encoder_net=None,
                                      shared_forecaster_net=None,
                                      shared_loss_net=None,
                                      for_finetune=False):
    """
    
    Parameters
    ----------
    factory : EncoderForecasterBaseFactory
    context : list
    shared_encoder_net : MyModule or None
    shared_forecaster_net : MyModule or None
    shared_loss_net : MyModule or None
    for_finetune : bool

    Returns
    -------

    """
    encoder_net = MyModule(factory.encoder_sym(),
                           data_names=[ele.name for ele in factory.encoder_data_desc()],
                           label_names=[],
                           context=context,
                           name="encoder_net")
    encoder_net.bind(data_shapes=factory.encoder_data_desc(),
                     label_shapes=None,
                     inputs_need_grad=True,
                     shared_module=shared_encoder_net)
    if shared_encoder_net is None:
        encoder_net.init_params(mx.init.MSRAPrelu(slope=0.2))
        init_optimizer_using_cfg(encoder_net, for_finetune=for_finetune)
    forecaster_net = MyModule(factory.forecaster_sym(),
                                   data_names=[ele.name for ele in
                                               factory.forecaster_data_desc()],
                                   label_names=[],
                                   context=context,
                                   name="forecaster_net")
    forecaster_net.bind(data_shapes=factory.forecaster_data_desc(),
                        label_shapes=None,
                        inputs_need_grad=True,
                        shared_module=shared_forecaster_net)
    if shared_forecaster_net is None:
        forecaster_net.init_params(mx.init.MSRAPrelu(slope=0.2))
        init_optimizer_using_cfg(forecaster_net, for_finetune=for_finetune)

    loss_net = MyModule(factory.loss_sym(),
                        data_names=[ele.name for ele in
                                    factory.loss_data_desc()],
                        label_names=[ele.name for ele in
                                     factory.loss_label_desc()],
                        context=context,
                        name="loss_net")
    loss_net.bind(data_shapes=factory.loss_data_desc(),
                  label_shapes=factory.loss_label_desc(),
                  inputs_need_grad=True,
                  shared_module=shared_loss_net)
    if shared_loss_net is None:
        loss_net.init_params()
    
    discrim_net = MyModule(factory.discrim_sym(),
                        data_names=[ele.name for ele in
                                    factory.discrim_data_desc()],
                        label_names=[],
                        context=context,
                        name="discrim_net")
    discrim_net.bind(data_shapes=factory.discrim_data_desc(),
                    label_shapes=None,
                    inputs_need_grad=True,
                    shared_module=None)
    discrim_net.init_params(mx.init.MSRAPrelu(slope=0.2))
    init_optimizer_using_cfg(discrim_net, for_finetune=for_finetune,
                            lr=cfg.MODEL.TRAIN.LR_DIS,
                            min_lr=cfg.MODEL.TRAIN.MIN_LR_DIS,
                            lr_decay_iter=cfg.MODEL.TRAIN.LR_DECAY_ITER_DIS, 
                            lr_decay_factor=cfg.MODEL.TRAIN.LR_DECAY_FACTOR_DIS,
                            optimizer_type=cfg.MODEL.TRAIN.OPTIMIZER_DIS.lower())

    loss_D_net = MyModule(factory.loss_D_sym(),
                        data_names=[ele.name for ele in
                                    factory.loss_D_data_desc()],
                        label_names=[ele.name for ele in
                                     factory.loss_D_label_desc()],
                        context=context,
                        name="loss_D_net")
    loss_D_net.bind(data_shapes=factory.loss_D_data_desc(),
                    label_shapes=factory.loss_D_label_desc(),
                    inputs_need_grad=True,
                    shared_module=None)
    loss_D_net.init_params()

    return encoder_net, forecaster_net, loss_net, discrim_net, loss_D_net


class EncoderForecasterStates(object):
    def __init__(self, factory, ctx):
        self._factory = factory
        self._ctx = ctx
        self._encoder_state_info = factory.init_encoder_state_info
        self._forecaster_state_info = factory.init_forecaster_state_info
        self._states_nd = []
        for info in self._encoder_state_info:
            state_shape = safe_eval(info['shape'])
            state_shape = (state_shape[0] * factory._ctx_num, ) + state_shape[1:]
            self._states_nd.append(mx.nd.zeros(shape=state_shape, ctx=ctx))

    def reset_all(self):
        for ele, info in zip(self._states_nd, self._encoder_state_info):
            ele[:] = 0

    def reset_batch(self, batch_id):
        for ele, info in zip(self._states_nd, self._encoder_state_info):
            ele[batch_id][:] = 0

    def update(self, states_nd):
        for target, src in zip(self._states_nd, states_nd):
            target[:] = src

    def get_encoder_states(self):
        return self._states_nd

    def get_forecaster_states(self):
        return self._states_nd


def train_step(batch_size, encoder_net, forecaster_net,
               loss_net, discrim_net, loss_D_net, init_states,
               data_nd, gt_nd, mask_nd, iter_id=None):
    """Finetune the encoder, forecaster and GAN for one step

    Parameters
    ----------
    batch_size : int
    encoder_net : MyModule
    forecaster_net : MyModule
    loss_net : MyModule
    discrim_net: MyModule
    loss_D_net: MyModule
    init_states : EncoderForecasterStates
    data_nd : mx.nd.ndarray
    gt_nd : mx.nd.ndarray
    mask_nd : mx.nd.ndarray
    iter_id : int

    Returns
    -------
    init_states: EncoderForecasterStates
    loss_dict: dict
    """
    # Forward Encoder
    encoder_net.forward(is_train=True,
                        data_batch=mx.io.DataBatch(data=[data_nd] + init_states.get_encoder_states()))
    encoder_states_nd = encoder_net.get_outputs()
    init_states.update(encoder_states_nd)
    # Forward Forecaster
    #if cfg.MODEL.OUT_TYPE == "direct":
    #    forecaster_net.forward(is_train=True,
    #                           data_batch=mx.io.DataBatch(data=init_states.get_forecaster_states()))
    #else:
    last_frame_nd = data_nd[data_nd.shape[0] - 1]
    if cfg.MODEL.ENCODER_FORECASTER.USE_SKIP:
        forecaster_net.forward(is_train=True,
                            data_batch=mx.io.DataBatch(data=init_states.get_forecaster_states() +
                                                                    [last_frame_nd]))
    else:
        forecaster_net.forward(is_train=True,
                            data_batch=mx.io.DataBatch(data=init_states.get_forecaster_states()))
    forecaster_outputs = forecaster_net.get_outputs()
    pred_nd = forecaster_outputs[0]
    
    discrim_net.forward(is_train=True,
                        data_batch=mx.io.DataBatch(data=[pred_nd]))
    discrim_output = discrim_net.get_outputs()[0]
    
    # Calculate the gradient of the loss functions
    if cfg.MODEL.ENCODER_FORECASTER.HAS_MASK:
        loss_net.forward_backward(data_batch=mx.io.DataBatch(data=[pred_nd, discrim_output],
                                                             label=[gt_nd, mask_nd]))
    else:
        loss_net.forward_backward(data_batch=mx.io.DataBatch(data=[pred_nd, discrim_output],
                                                             label=[gt_nd]))
    pred_grad_ordinary = loss_net.get_input_grads()[0]
    discrim_net.backward(out_grads=[loss_net.get_input_grads()[1]])
    pred_grad_gan = discrim_net.get_input_grads()[0]
    grad_ratio = pred_grad_gan.norm().asscalar() / pred_grad_ordinary.norm().asscalar()
    #print(pred_grad_gan)
    pred_grad = pred_grad_ordinary + pred_grad_gan  # add up gradients computed from different path with respect to pred

    loss_dict = loss_net.get_output_dict()
    for k in loss_dict:
        loss_dict[k] = nd.mean(loss_dict[k]).asscalar()
    # Backward Forecaster
    forecaster_net.backward(out_grads=[pred_grad])
    encoder_states_grad_nd = forecaster_net.get_input_grads()
    # Backward Encoder
    encoder_net.backward(out_grads=encoder_states_grad_nd)
    # Update forecaster and encoder
    forecaster_grad_norm = forecaster_net.clip_by_global_norm(max_norm=cfg.MODEL.TRAIN.GRAD_CLIP)
    encoder_grad_norm = encoder_net.clip_by_global_norm(max_norm=cfg.MODEL.TRAIN.GRAD_CLIP)
    forecaster_net.update()
    encoder_net.update()
    
    # train the discriminator
    dis_loss = 0.0
    label = mx.nd.zeros_like(discrim_output)
    # fake data
    loss_D_net.forward_backward(data_batch=mx.io.DataBatch(data=[discrim_output],
                                                            label=[label]))
    dis_loss += mx.nd.mean(loss_D_net.get_output_dict()['dis_output']).asscalar()
    #print('fake loss:',mx.nd.mean(loss_D_net.get_output_dict()['dis_output']).asscalar())
    fake_grad = loss_D_net.get_input_grads()[0]
    #print('fake output:',mx.nd.mean(discrim_output).asscalar())
    discrim_net.backward(out_grads=[fake_grad])
    temp_grad = [[grad.copyto(grad.context) for grad in grads] for grads in discrim_net._exec_group.grad_arrays]
    # true data
    label[:] = 1.0
    discrim_net.forward(data_batch=mx.io.DataBatch(data=[gt_nd]))
    discrim_output = discrim_net.get_outputs()[0]
    loss_D_net.forward_backward(data_batch=mx.io.DataBatch(data=[discrim_output],
                                                            label=[label]))
    dis_loss += mx.nd.mean(loss_D_net.get_output_dict()['dis_output']).asscalar()
    #print('true loss:',mx.nd.mean(loss_D_net.get_output_dict()['dis_output']).asscalar())
    #print('true output:',mx.nd.mean(discrim_output).asscalar())
    true_grad = loss_D_net.get_input_grads()[0]
    discrim_net.backward(out_grads=[true_grad])
    # add them up
  
    for gradsr, gradsf in zip(discrim_net._exec_group.grad_arrays, temp_grad):
        for gradr, gradf in zip(gradsr, gradsf):
            gradr += gradf 
    discriminator_grad_norm = discrim_net.clip_by_global_norm(max_norm=cfg.MODEL.TRAIN.GRAD_CLIP_DIS)
    #print(discrim_net._exec_group.grad_arrays)
    discrim_net.update()
    discrim_net.spectral_normalize()
    dis_loss = dis_loss / 2
    loss_dict['dis_output'] = dis_loss
    loss_str = ", ".join(["%s=%g" %(k, v) for k, v in loss_dict.items()])
    if iter_id is not None:
        logging.info("Iter:%d, %s,\n e_gnorm=%g, f_gnorm=%g, d_gnorm=%g, gan:other=%g"
                     % (iter_id, loss_str, encoder_grad_norm, forecaster_grad_norm, discriminator_grad_norm, grad_ratio))

    return init_states, loss_dict

'''
def load_encoder_forecaster_params(load_dir, load_iter, encoder_net, forecaster_net):
    logging.info("Loading parameters from {}, Iter = {}"
                 .format(os.path.realpath(load_dir), load_iter))
    encoder_arg_params, encoder_aux_params = load_params(prefix=os.path.join(load_dir,
                                                                             "encoder_net"),
                                                         epoch=load_iter)
    encoder_net.init_params(arg_params=encoder_arg_params, aux_params=encoder_aux_params,
                            allow_missing=False, force_init=True)
    forecaster_arg_params, forecaster_aux_params = load_params(prefix=os.path.join(load_dir,
                                                                             "forecaster_net"),
                                                               epoch=load_iter)
    forecaster_net.init_params(arg_params=forecaster_arg_params,
                               aux_params=forecaster_aux_params,
                               allow_missing=False,
                               force_init=True)
    logging.info("Loading Complete!")
'''
