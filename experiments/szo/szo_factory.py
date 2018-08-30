import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,*([".."]*3))))
import mxnet as mx
from nowcasting.config import cfg
from nowcasting.szo_evaluation import rainfall_to_pixel
from nowcasting.encoder_forecaster import EncoderForecasterBaseFactory
from nowcasting.operators import *
from nowcasting.ops import *

def get_temporal_weight_symbol(seq_len):
    if cfg.MODEL.TEMPORAL_WEIGHT_TYPE == "same":
        return mx.sym.ones((seq_len,))
    elif cfg.MODEL.TEMPORAL_WEIGHT_TYPE == "linear":
        upper = cfg.MODEL.TEMPORAL_WEIGHT_UPPER
        assert upper >= 1.0
        temporal_mult = 1 + \
                        mx.sym.arange(start=0, stop=seq_len) * (upper - 1.0) / (seq_len - 1.0)
        temporal_mult = mx.sym.reshape(temporal_mult, shape=(seq_len, 1, 1, 1, 1))
        return temporal_mult
    elif cfg.MODEL.TEMPORAL_WEIGHT_TYPE == "exponential":
        upper = cfg.MODEL.TEMPORAL_WEIGHT_UPPER
        assert upper >= 1.0
        base_factor = np.log(upper) / (seq_len - 1.0)
        temporal_mult = mx.sym.exp(mx.sym.arange(start=0, stop=seq_len) * base_factor)
        temporal_mult = mx.sym.reshape(temporal_mult, shape=(seq_len, 1, 1, 1, 1))
        return temporal_mult
    else:
        raise NotImplementedError

def get_loss_weight_symbol(data, mask, seq_len):
    """
    data, mask, seq_len are symbols, pixel values [0,255] np.float32
    return weights have the same shape as data and mask, np.float32
    """
    # use symbol here, since it's part of the network!!!
    if cfg.MODEL.USE_BALANCED_LOSS:
        if cfg.MODEL.DATA_MODE == 'rescaled':
            balancing_weights = cfg.MODEL.BALANCING_WEIGHTS
            weights = mx.sym.ones_like(data) * balancing_weights[0]
            thresholds = [ele / 255.0 for ele in cfg.MODEL.THRESHOLDS]
            for i, threshold in enumerate(thresholds):
                weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (data >= threshold)
            weights = weights * mask
        elif cfg.MODEL.DATA_MODE == 'original':
            epsilon = 10**(-3)
            assert cfg.SZO.ITERATOR.DOWN_RATIO == 1.0
            thresh_mat = data<1.0
            data_size = cfg.MODEL.OUT_LEN*cfg.MODEL.TRAIN.BATCH_SIZE*cfg.SZO.DATA.SIZE**2
            ratio = mx.symbol.sum(thresh_mat)/data_size
            assert (ratio < 1.0) and (ratio > 0.0)
            weights = (mx.sym.broadcast_mul(thresh_mat, (cfg.MODEL.BALANCE_FACTOR/(ratio+epsilon))) + mx.sym.broadcast_mul(1-thresh_mat, (1-cfg.MODEL.BALANCE_FACTOR)/(1-ratio+epsilon))) / 2 #* mask
            weights = weights * mask
        else:
            raise NotImplementedError
    else:
        weights = mask
    temporal_mult = get_temporal_weight_symbol(seq_len)
    weights = mx.sym.broadcast_mul(weights, temporal_mult)
    return weights


class SZONowcastingFactory(EncoderForecasterBaseFactory):
    def __init__(self,
                 batch_size,
                 in_seq_len,
                 out_seq_len,
                 frame_stack=1,
                 ctx_num=1,
                 name="szo_nowcasting"):
        super(SZONowcastingFactory, self).__init__(batch_size=batch_size,
                                                   in_seq_len=in_seq_len,
                                                   out_seq_len=out_seq_len,
                                                   frame_stack=frame_stack,
                                                   ctx_num=ctx_num,
                                                   height=cfg.SZO.ITERATOR.RESIZED_SIZE,
                                                   width=cfg.SZO.ITERATOR.RESIZED_SIZE,
                                                   name=name)
    
    def discriminator(self, video=mx.sym.Variable('video')):
        """
        args:
        video - symbol of shape (seq_len, batch_size, channels, height, width)
        return:
        output - symbol of shape (batch_size, 1)
        """
        if not cfg.MODEL.DISCRIMINATOR.USE_2D:
            # transform the layout to (batch_size, channels, seq_len, height, width)
            video = mx.symbol.transpose(video, axes=(1, 2, 0, 3, 4))
            num_layers = len(cfg.MODEL.DISCRIMINATOR.DISCRIM_CONV) 
            pool_layer = mx.gluon.nn.MaxPool3D(strides=cfg.MODEL.DISCRIMINATOR.DOWNSAMPLE_VIDEO, padding=1)
            video = pool_layer(video)

            for i in range(num_layers):
                if i==0:
                    inputs = video
                else:
                    inputs = output
                output = conv3d_bn_act(data=inputs, 
                                    height=cfg.MODEL.DISCRIMINATOR.FEATMAP_SIZE[i][1],
                                    width=cfg.MODEL.DISCRIMINATOR.FEATMAP_SIZE[i][1],
                                    num_filter=cfg.MODEL.DISCRIMINATOR.DISCRIM_CONV[i]['num_filter'],
                                    kernel=cfg.MODEL.DISCRIMINATOR.DISCRIM_CONV[i]['kernel'],
                                    stride=cfg.MODEL.DISCRIMINATOR.DISCRIM_CONV[i]['stride'],
                                    pad=cfg.MODEL.DISCRIMINATOR.DISCRIM_CONV[i]['padding'],
                                    act_type=cfg.MODEL.CNN_ACT_TYPE,
                                    name='discrim_conv'+str(i))
                # another conv3d will serve as pooling layer
                if i<num_layers-1:
                    output = conv3d_bn_act(data=output,
                                        height=cfg.MODEL.DISCRIMINATOR.FEATMAP_SIZE[i+1][1],
                                        width=cfg.MODEL.DISCRIMINATOR.FEATMAP_SIZE[i+1][1],
                                        num_filter=cfg.MODEL.DISCRIMINATOR.DISCRIM_POOL[i]['num_filter'],
                                        kernel=cfg.MODEL.DISCRIMINATOR.DISCRIM_POOL[i]['kernel'],
                                        stride=cfg.MODEL.DISCRIMINATOR.DISCRIM_POOL[i]['stride'],
                                        pad=cfg.MODEL.DISCRIMINATOR.DISCRIM_POOL[i]['padding'],
                                        act_type=cfg.MODEL.CNN_ACT_TYPE,
                                        name='discrim_pool'+str(i))
            output = fc_layer(data=output.reshape([self._batch_size, -1]),
                            num_hidden=1,
                            name='discrim_fc',
                            no_bias=True)
            output = output.reshape([self._batch_size,])
            return output
        else:
            # transform to shape (seqlen*batch_size, channel, height, width)
            video = video.reshape(shape=(-1,0,0,0), reverse=True)
            num_layers = len(cfg.MODEL.DISCRIMINATOR.DISCRIM_CONV)
            pool_layer = mx.gluon.nn.MaxPool2D(strides=cfg.MODEL.DISCRIMINATOR.DOWNSAMPLE_VIDEO, padding=1)
            video = pool_layer(video)
            for i in range(num_layers):
                if i == 0:
                    inputs = video
                else:
                    inputs = output
                output = conv2d_bn_act(data=inputs, 
                                    num_filter=cfg.MODEL.DISCRIMINATOR.DISCRIM_CONV[i]['num_filter'],
                                    kernel=cfg.MODEL.DISCRIMINATOR.DISCRIM_CONV[i]['kernel'],
                                    stride=cfg.MODEL.DISCRIMINATOR.DISCRIM_CONV[i]['stride'],
                                    pad=cfg.MODEL.DISCRIMINATOR.DISCRIM_CONV[i]['padding'],
                                    act_type=cfg.MODEL.CNN_ACT_TYPE,
                                    name='discrim_conv'+str(i))
                # another conv3d will serve as pooling layer
                if i<num_layers-1:
                    output = conv2d_bn_act(data=output,
                                        num_filter=cfg.MODEL.DISCRIMINATOR.DISCRIM_POOL[i]['num_filter'],
                                        kernel=cfg.MODEL.DISCRIMINATOR.DISCRIM_POOL[i]['kernel'],
                                        stride=cfg.MODEL.DISCRIMINATOR.DISCRIM_POOL[i]['stride'],
                                        pad=cfg.MODEL.DISCRIMINATOR.DISCRIM_POOL[i]['padding'],
                                        act_type=cfg.MODEL.CNN_ACT_TYPE,
                                        name='discrim_pool'+str(i))
            output = fc_layer(data=output.reshape([self._batch_size*self._out_seq_len, -1]),
                            num_hidden=1,
                            name='discrim_fc',
                            no_bias=True)
            output = output.reshape([self._out_seq_len, self._batch_size], __layout__="TN")
            return output

    
    def loss_sym(self,
                 pred=mx.sym.Variable('pred'),
                 mask=mx.sym.Variable('mask'),
                 target=mx.sym.Variable('target'),
                 discrim_output=mx.sym.Variable('discrim_out')):
        """Construct loss symbol.

        Optional args:
            pred: Shape (out_seq_len, batch_size, C, H, W)
            mask: Shape (out_seq_len, batch_size, C, H, W)
            target: Shape (out_seq_len, batch_size, C, H, W)
            discrim_output Shape (batch_size,) if 3d discrim, (out_seq_len, batch_size,) if 2d discrim
        """
        self.reset_all()
        weights = get_loss_weight_symbol(data=target, mask=mask, seq_len=self._out_seq_len)
        mse = weighted_mse(pred=pred, gt=target, weight=weights)
        mae = weighted_mae(pred=pred, gt=target, weight=weights)
        gdl = masked_gdl_loss(pred=pred, gt=target, mask=mask)
        avg_mse = mx.sym.mean(mse)
        avg_mae = mx.sym.mean(mae)
        avg_gdl = mx.sym.mean(gdl)
        # gan loss is for a whole sequence, and isn't influenced by weights
        if not cfg.MODEL.DISCRIMINATOR.USE_2D:
            gan = mx.sym.abs(discrim_output - mx.sym.ones_like(discrim_output))
        else:
            gan = mx.sym.square(discrim_output - mx.sym.ones_like(discrim_output))
            temporal_weights = get_temporal_weight_symbol(self._out_seq_len)
            gan = mx.sym.broadcast_mul(mx.sym.sum(mx.sym.broadcast_mul(gan, temporal_weights), axis=0), 1/mx.sym.sum(temporal_weights))  # normalize to 0-1
        avg_gan = mx.sym.mean(gan)  # average over batches and frames
        
        global_grad_scale = cfg.MODEL.NORMAL_LOSS_GLOBAL_SCALE
        if cfg.MODEL.L2_LAMBDA > 0:
            avg_mse = mx.sym.MakeLoss(avg_mse,
                                      grad_scale=global_grad_scale * cfg.MODEL.L2_LAMBDA,
                                      name="mse")
        else:
            avg_mse = mx.sym.BlockGrad(avg_mse, name="mse")
        if cfg.MODEL.L1_LAMBDA > 0:
            avg_mae = mx.sym.MakeLoss(avg_mae,
                                      grad_scale=global_grad_scale * cfg.MODEL.L1_LAMBDA,
                                      name="mae")
        else:
            avg_mae = mx.sym.BlockGrad(avg_mae, name="mae")
        if cfg.MODEL.GDL_LAMBDA > 0:
            avg_gdl = mx.sym.MakeLoss(avg_gdl,
                                      grad_scale=global_grad_scale * cfg.MODEL.GDL_LAMBDA,
                                      name="gdl")
        else:
            avg_gdl = mx.sym.BlockGrad(avg_gdl, name="gdl")       
        if cfg.MODEL.GAN_G_LAMBDA > 0:
            avg_gan = mx.sym.MakeLoss(avg_gan, 
                                    grad_scale=global_grad_scale*cfg.MODEL.GAN_G_LAMBDA,
                                    name='gan')
        else:
            avg_gan = mx.sym.BlockGrad(avg_gan, name='gan')
        
        loss = mx.sym.Group([avg_mse, avg_mae, avg_gdl, avg_gan])
        return loss

    def loss_D_sym(self, 
                discrim_output=mx.sym.Variable('discrim_out'), 
                label=mx.sym.Variable('discrim_label')):
        if not cfg.MODEL.DISCRIMINATOR.USE_2D:
            discrim_loss = mx.sym.abs(discrim_output -label)
        else:
            discrim_loss = mx.sym.square(discrim_output - label)
        avg_discrim_loss = mx.sym.mean(discrim_loss)  # here we don't need to have temporal weights
        global_grad_scale = cfg.MODEL.NORMAL_LOSS_GLOBAL_SCALE
        if cfg.MODEL.GAN_D_LAMBDA > 0:
            avg_discrim_loss = mx.sym.MakeLoss(avg_discrim_loss,
                                                grad_scale=global_grad_scale*cfg.MODEL.GAN_D_LAMBDA,
                                                name='dis')
        else:
            avg_discrim_loss = mx.sym.BlockGrad(avg_discrim_loss, name='dis')

        loss = mx.sym.Group([avg_discrim_loss])
        return loss

if __name__ == '__main__':
    a = mx.nd.uniform(0,2,shape=(25,2,1,500,500))
    print(get_loss_weight_symbol(a, None, None))
