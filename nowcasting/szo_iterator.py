import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,*([".."]*3))))
import numpy as np
from nowcasting import image
from nowcasting.mask import *
from nowcasting.config import cfg
from nowcasting.utils import *
import os
import random
import mxnet as mx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time

class SZOIterator:
    def __init__(self, rec_paths, in_len, out_len, batch_size, ctx):
        if isinstance(rec_paths, list):
            self.rec_paths = rec_paths
        else:
            self.rec_paths = [rec_paths]
        assert in_len + out_len < cfg.SZO.DATA.TOTAL_LEN, 'require a sequence that is too long'
        self.in_len = in_len
        self.out_len = out_len
        self.batch_size = batch_size
        self.height = cfg.SZO.DATA.SIZE // cfg.SZO.ITERATOR.DOWN_RATIO
        self.width = cfg.SZO.DATA.SIZE // cfg.SZO.ITERATOR.DOWN_RATIO
        self.dataset_ind = 0
        self.image_iterator = mx.io.ImageRecordIter(
                                    path_imgrec=self.rec_paths[self.dataset_ind],
                                    data_shape=(1, cfg.SZO.DATA.SIZE, cfg.SZO.DATA.SIZE),
                                    batch_size=cfg.SZO.DATA.TOTAL_LEN*self.batch_size,
                                    prefectch_buffer=4,
                                    preprocess_threads=4
                                    )
        if isinstance(ctx, list):
            self.ctx = ctx[0]
        else:
            self.ctx = ctx
    
    def sample(self):    
        try:
            batch = self.image_iterator.next()
        except StopIteration:
            self.reset()
            batch = self.image_iterator.next()
        finally:
            frames = batch.data[0].reshape([self.batch_size, cfg.SZO.DATA.TOTAL_LEN, 1, cfg.SZO.DATA.SIZE, cfg.SZO.DATA.SIZE])
            frames = frames.transpose([1,0,2,3,4]) # to make frames in a video appear consecutively
        shift = random.randint(0, cfg.SZO.DATA.TOTAL_LEN - self.in_len - self.out_len)
        frames = frames[shift:(shift+self.in_len+self.out_len),:,:,:,:]
        #masks = (frames <= cfg.SZO.DATA.RADAR_RANGE).astype(np.float32)       
        return frames.as_in_context(self.ctx)#, masks.as_in_context(self.ctx)

    def reset(self):
        self.dataset_ind = (self.dataset_ind + 1) % len(self.rec_paths)
        self.image_iterator = mx.io.ImageRecordIter(
                                    path_imgrec=self.rec_paths[self.dataset_ind],
                                    data_shape=(1, cfg.SZO.DATA.SIZE, cfg.SZO.DATA.SIZE),
                                    batch_size = self.batch_size*cfg.SZO.DATA.TOTAL_LEN,
                                    )
        self.image_iterator.reset()

    def resize(self, arr_nd):
        # the array should be of shape (seqlen*batch_size, channel, height, width)
        arr_nd = mx.nd.contrib.BilinearResize2D(arr_nd, self.height, self.width)
        arr_nd = arr_nd.reshape([cfg.SZO.DATA.TOTAL_LEN, 1, self.height, self.width])
        return arr_nd

    def get_num_frames(self):
        return self.in_len + self.out_len

def show_histogram(batch_size, repeat_times):
    train_iterator = SZOIterator(rec_paths=cfg.SZO_TRAIN_DATA_PATHS,
                                 in_len=cfg.MODEL.IN_LEN,
                                 out_len=cfg.MODEL.OUT_LEN,
                                 batch_size=batch_size,
                                 ctx=mx.gpu())
    for i in range(repeat_times):
        train_batch = train_iterator.sample()
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.hist(train_batch.asnumpy().reshape([-1]), bins=50)
        plt.savefig('data_display/histogram{}'.format(i))


def test_speed(batch_size, repeat_times):
    train_iterator = SZOIterator(rec_paths=cfg.SZO_TRAIN_DATA_PATHS,
                                 in_len=cfg.MODEL.IN_LEN,
                                 out_len=cfg.MODEL.OUT_LEN,
                                 batch_size=batch_size,
                                 ctx=mx.gpu())

    begin = time.time()
    for i in range(repeat_times):
        sample_sequence = train_iterator.sample()
    end = time.time()
    print("Train Data Sample FPS: %f" % (batch_size * train_iterator.get_num_frames()
                                        * repeat_times / float(end - begin)))

def save_png_sequence(np_seqs, path):
    # np_seq of shape (seqlen, H, W)
    if not isinstance(np_seqs, list):
        np_seqs = [np_seqs]
    num_seqs = len(np_seqs)
    seq_len = max([np_seq.shape[0] for np_seq in np_seqs])
    plt.figure(figsize=(seq_len, num_seqs))
    for n,np_seq in enumerate(np_seqs):
        for i in range(np_seq.shape[0]):
            plt.subplot(num_seqs, seq_len, n*seq_len+i+1)
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            frame = np_seq[i]
            frame = np.expand_dims(frame, axis=2)
            frame = np.concatenate([frame, frame, frame], axis=2)
            plt.imshow(frame)
    plt.savefig(path)
    plt.close('all')

def save_gif_examples(num):
    train_iterator = SZOIterator(rec_paths=cfg.SZO_TRAIN_DATA_PATHS,
                                 in_len=cfg.MODEL.IN_LEN,
                                 out_len=cfg.MODEL.OUT_LEN,
                                 batch_size=1,
                                 ctx=mx.gpu()) 

    for i in range(num):
        train_batch = train_iterator.sample()
        
        examples = train_batch[:,0,0,:,:].asnumpy().astype(np.uint8)
        scaled_examples = ((examples*(examples<255))*(255/80)).astype(np.uint8)

        if not os.path.exists('data_display'):
            os.makedirs('data_display')

        original_name_str = os.path.join('data_display', 'original')
        scaled_name_str = os.path.join('data_display', 'scaled')

        save_hko_gif(examples, save_path=original_name_str +str(i) + '.gif', multiply_by_255=False)
        save_hko_gif(scaled_examples, save_path=scaled_name_str + str(i) + '.gif', multiply_by_255=False)
        #save_hko_gif((train_batch[:, 0, 0, :, :]*train_mask[:, 0, 0, :, :]).asnumpy().astype(np.uint8),
        #save_hko_gif(train_mask[:, 0, 0, :, :].asnumpy().astype(np.uint8) * 255,
        #             save_path=name_str + '_mask.gif')
        save_png_sequence(examples, path=original_name_str+'_'+str(i)+'.png')
        save_png_sequence(scaled_examples, path=scaled_name_str+'_'+str(i)+'.png')


# Simple test for the performance of the HKO iterator.
if __name__ == '__main__':
    np.random.seed(123)
    import time
    from nowcasting.config import cfg
    from nowcasting.helpers.visualization import save_hko_gif, save_hko_movie

    save_gif_examples(100)
    #show_histogram(64, 100)
     

