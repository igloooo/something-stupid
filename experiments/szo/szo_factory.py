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

def get_gradient_norm_symbold(data):
    """
    assume data has layout TNCHW
    """
    x_diff = one_step_diff(data, axis=3)
    y_diff = one_step_diff(data, axis=4)
    x_diff = x_diff.slice_axis(axis=4, begin=0, end=-1)
    y_diff = y_diff.slice_axis(axis=3, begin=0, end=-1)
    g_norm = mx.sym.sqrt(mx.sym.square(x_diff)+mx.sym.square(y_diff))
    g_norm = mx.sym.pad(g_norm, mode='edge', pad_width=(0,0,0,0,0,0,0,1,0,1))
    g_norm = g_norm.reshape(shape=[-1,0,0,0], reverse=True)
    kernel_size = cfg.MODEL.GRAD_BLUR_KERNEL_SIZE
    blur_kernel = mx.sym.ones([kernel_size, kernel_size])/(kernel_size**2)
    blurred_gnorm = mx.sym.Convolution(data=data, num_filter=1, kernel=kernel_size, stride=1,
                                weight=blur_kernel, dilate=1, no_bias=True,
                                pad=1, name='blur_loss', workspace=256)
    blurred_gnorm = blurred_gnorm.reshape([cfg.MODEL.OUT_LEN, -1,0,0,0], reverse=True, __layout__='TNCHW')
    return blurred_gnorm

def get_loss_weight_symbol(data, mask, seq_len):
    """
    data, mask, seq_len are symbols, pixel values [0,255] np.float32
    return weights have the same shape as data and mask, np.float32
    """
    if cfg.MODEL.PROBLEM_FORM == 'regression':
        if cfg.MODEL.DATA_MODE == 'rescaled':
            balancing_weights = cfg.MODEL.BALANCING_WEIGHTS
            weights = mx.sym.ones_like(data) * balancing_weights[0]
            thresholds = [ele / 80.0 for ele in cfg.MODEL.THRESHOLDS]
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
        if cfg.MODEL.USE_GWEIGHTS:
            gnorm = get_gradient_norm_symbold(data)
            bwg = cfg.MODEL.BALANCING_WEIGHTS_GRADIENT
            tg = cfg.MODEL.THRESHOLD_GRADIENT
            g_weights = mx.sym.ones_like(data) * bwg[0]
            for i, threshold in enumerate(tg):
                g_weights = g_weights + (bwg[i+1] - bwg[i]) * (data >= tg[i])
            weights = weights*g_weights
    elif cfg.MODEL.PROBLEM_FORM == 'classification':
        # assume target along channel dimension is a one hot vector
        for i in range(len(cfg.MODEL.BINS)):
            if i==0:
                weights = data.slice_axis(axis=2, begin=i, end=i+1) * cfg.MODEL.BIN_WEIGHTS[i]
            else:
                weights = weights + data.slice_axis(axis=2, begin=i, end=i+1) * cfg.MODEL.BIN_WEIGHTS[i]
        weights = weights*mask
    else:
        raise NotImplementedError

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
                if (i == num_layers - 1) and cfg.MODEL.DISCRIMINATOR.PIXEL: # this is a stupid and temporary chagne
                    output = conv2d_act(data=inputs,
                                        num_filter=cfg.MODEL.DISCRIMINATOR.DISCRIM_CONV[i]['num_filter'],
                                        kernel=cfg.MODEL.DISCRIMINATOR.DISCRIM_CONV[i]['kernel'],
                                        stride=cfg.MODEL.DISCRIMINATOR.DISCRIM_CONV[i]['stride'],
                                        pad=cfg.MODEL.DISCRIMINATOR.DISCRIM_CONV[i]['padding'],
                                        act_type=cfg.MODEL.CNN_ACT_TYPE,
                                        name='discrim_conv'+str(i))
                else:
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
            if not cfg.MODEL.DISCRIMINATOR.PIXEL:
                output = fc_layer(data=output.reshape([self._batch_size*self._out_seq_len, -1]),
                                num_hidden=1,
                                name='discrim_fc',
                                no_bias=True)
                output = output.reshape([self._out_seq_len, self._batch_size], __layout__="TN")
            else:
                #output = conv2d_act(data=output,
                #                num_filter=1,
                #                name='discrim_1x1',
                #                act_type='relu')
                output = output.reshape([self._out_seq_len, self._batch_size,0,0,0], reverse=True, __layout__="TNCHW")
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
        global_grad_scale = cfg.MODEL.NORMAL_LOSS_GLOBAL_SCALE
        if cfg.MODEL.PROBLEM_FORM == 'regression':
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
                if cfg.MODEL.DISCRIMINATOR.PIXEL:
                    gan = mx.sym.mean(gan, axis=(2,3,4))
                temporal_weights = get_temporal_weight_symbol(self._out_seq_len)
                gan = mx.sym.broadcast_mul(mx.sym.sum(mx.sym.broadcast_mul(gan, temporal_weights), axis=0), 1/mx.sym.sum(temporal_weights))  # normalize to 0-1
            avg_gan = mx.sym.mean(gan)  # average over batches and frames
            
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
        elif cfg.MODEL.PROBLEM_FORM == 'classification':
            scale = cfg.SZO.DATA.SIZE // cfg.MODEL.TARGET_TRAIN_SIZE
            mask = mask.reshape([-1,0,0,0], reverse=True)
            mask = mx.sym.Pooling(data=mask, kernel=(scale, scale), stride=(scale, scale), pool_type='max')
            mask = mask.reshape([self._out_seq_len, self._batch_size, 0,0,0], reverse=True)
            target = target.reshape([-1,0,0,0], reverse=True)
            target = mx.sym.Pooling(data=target, kernel=(scale, scale), stride=(scale, scale), pool_type='max')
            target = target.reshape([self._out_seq_len, self._batch_size, 0,0,0], reverse=True)
            target_class_list = []
            # generate classes graph from target sequence; note that the range of radar reflexity is 0-1
            for i in range(len(cfg.MODEL.BINS)):
                lower = cfg.MODEL.BINS[i]
                if i < len(cfg.MODEL.BINS)-1:
                    upper = cfg.MODEL.BINS[i+1]
                else:
                    upper = 1
                target_class_list.append(((target>=lower)*(target<upper)))
            target_class = mx.sym.concat(*target_class_list, dim=2)

            pred = mx.sym.softmax(pred, axis=2)
            
            eps = cfg.MODEL.CE_EPSILON
            ce_loss = -mx.sym.sum(target_class*mx.sym.log(pred+eps), axis=2).expand_dims(axis=2)
            #pred = pred.transpose((0,1,3,4,2))
            #pred = pred.reshape([-1,0], reverse=True)
            #target_class = target_class.transpose((0,1,3,4,2))
            #target_class = target_class.reshape([-1,0], reverse=True)
            #ce_loss = ce_loss.reshape([self._out_seq_len, self._batch_size, self._height, self._width,0],
            #                            __layout__='TNHWC')
            #ce_loss = ce_loss.transpose((0,1,4,2,3))
            ce_loss = ce_loss * get_loss_weight_symbol(target_class, mask, self._out_seq_len)
            ce_loss = mx.sym.sum(ce_loss, axis=(2,3,4))
            ce_loss = mx.sym.mean(ce_loss)
            loss = mx.sym.MakeLoss(ce_loss, grad_scale=global_grad_scale, name='ce')
            #log_pred = mx.sym.MakeLoss(mx.sym.min(pred), name='log_pred')
            return loss  # mx.sym.Group([loss, log_pred])
        else:
            raise NotImplementedError


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
