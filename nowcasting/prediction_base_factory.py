import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,*([".."]*2))))
import mxnet as mx
from nowcasting.config import cfg
from nowcasting.ops import reset_regs
from nowcasting.operators.common import grid_generator


class PredictionBaseFactory(object):
    def __init__(self, batch_size, in_seq_len, out_seq_len, height, width, frame_stack=1, ctx_num=1,  name="forecaster"):
        self._out_typ = cfg.MODEL.OUT_TYPE
        self._batch_size = batch_size
        self._ctx_num = ctx_num
        self._in_seq_len = in_seq_len
        self._out_seq_len = out_seq_len
        self._frame_stack = frame_stack
        self._height = height
        self._width = width
        self._name = name
        self._spatial_grid = grid_generator(batch_size=batch_size, height=height, width=width)
        self.rnn_list = self._init_rnn()
        self._reset_rnn()

    def _pre_encode_frame(self, frame_data, seqlen, frame_stack=1):
        # suppose layout (T, N, C, H, W)
        #frame_data = frame_data.transpose([1,0,2,3,4])
        #frame_data = frame_data.reshape([0, seqlen//frame_stack, frame_stack, 0, 0])
        #frame_data = frame_data.transpose([1,0,2,3,4])
        if frame_stack > 1:
            frame_data =  mx.sym.concat(mx.sym.broadcast_to(frame_data.slice_axis(axis=0, begin=0, end=1), 
                                                            shape=(frame_stack-1, self._batch_size,
                                                                1, self._height, self._width)),
                                        frame_data, 
                                        dim=0)
        shifted_frame_datas = [frame_data.slice_axis(axis=0, begin=i, end=(seqlen+i)) for i in range(frame_stack)]
        frame_data = mx.sym.concat(*shifted_frame_datas,dim=2)
        ret = mx.sym.Concat(frame_data,
                             mx.sym.broadcast_to(mx.sym.expand_dims(self._spatial_grid, axis=0),
                                                 shape=(seqlen, self._batch_size,
                                                        2, self._height, self._width)),
                             mx.sym.ones(shape=(seqlen, self._batch_size, 1,
                                                self._height, self._width)),
                             num_args=3, dim=2)
        return ret

    def _init_rnn(self):
        raise NotImplementedError

    def _reset_rnn(self):
        for rnn in self.rnn_list:
            rnn.reset()

    def reset_all(self):
        reset_regs()
        self._reset_rnn()


class RecursiveOneStepBaseFactory(PredictionBaseFactory):
    def __init__(self, batch_size, in_seq_len, out_seq_len, height, width, use_ss=False,
                 name="forecaster"):
        super(RecursiveOneStepBaseFactory, self).__init__(batch_size=batch_size,
                                                          in_seq_len=in_seq_len,
                                                          out_seq_len=out_seq_len,
                                                          height=height,
                                                          width=width,
                                                          name=name)
        self._use_ss = False

