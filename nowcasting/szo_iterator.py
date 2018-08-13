import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,*([".."]*3))))
import numpy as np
from nowcasting import image
from nowcasting.mask import *
from nowcasting.config import cfg
from nowcasting.utils import *
from nowcasting.helpers.visualization import save_hko_gif
import os
import random
import mxnet as mx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time
import re

class SZOIterator:
    def __init__(self, rec_paths, in_len, out_len, batch_size, ctx, frame_skip=1, auto_add_file=True):
        if isinstance(rec_paths, list):
            self.rec_paths = rec_paths
        else:
            self.rec_paths = [rec_paths]
        assert (in_len + out_len-1)*frame_skip+1<= cfg.SZO.DATA.TOTAL_LEN, 'require a sequence that is too long'
        self.in_len = in_len
        self.out_len = out_len
        self.batch_size = batch_size
        self.frame_skip = frame_skip
        #self.height = cfg.SZO.DATA.SIZE // cfg.SZO.ITERATOR.DOWN_RATIO
        #self.width = cfg.SZO.DATA.SIZE // cfg.SZO.ITERATOR.DOWN_RATIO
        self.dataset_ind = 0
        self.image_iterator = mx.io.ImageRecordIter(
                                    path_imgrec=self.rec_paths[self.dataset_ind],
                                    data_shape=(1, cfg.SZO.DATA.SIZE, cfg.SZO.DATA.SIZE),
                                    batch_size=cfg.SZO.DATA.TOTAL_LEN*self.batch_size,
                                    prefectch_buffer=4,
                                    preprocess_threads=4
                                    )
        self.auto_add_file= auto_add_file
        if isinstance(ctx, list):
            self.ctx = ctx[0]
        else:
            self.ctx = ctx    

        self.folders=[]
        prefix = self.rec_paths[0].split('.')[0]
        self.load_lst(prefix+'.lst')
        self.seq_ind = 0
        
    def load_lst(self, lst_path):
        with open(lst_path, 'r') as f:
            f.seek(0)
            for i, line in enumerate(f):
                if i%cfg.SZO.DATA.TOTAL_LEN == 0:
                    self.folders.append(re.split(r'\s+', line)[-2].split('/')[-2])

    def sample(self):    
        """
        return tensor of 
        shape [self.in_len+self.out_len, self.batch_size, 1, cfg.SZO.DATA.SIZE, cfg.SZO.DATA.SIZE]
        pixel value [0,255] np.float32
        context self.ctx
        """
        try:
            batch = self.image_iterator.next()
        except StopIteration:
            self.reset()
            batch = self.image_iterator.next()
        finally:
            self.seq_ind += self.batch_size
            frames = batch.data[0].reshape([self.batch_size, cfg.SZO.DATA.TOTAL_LEN, 1, cfg.SZO.DATA.SIZE, cfg.SZO.DATA.SIZE])
            frames = frames.transpose([1,0,2,3,4]) # to make frames in a video appear consecutively
        ret_len = self.in_len + self.out_len
        shift = random.randint(0, cfg.SZO.DATA.TOTAL_LEN - ret_len*self.frame_skip)
        frames = frames[shift:(shift+(ret_len-1)*self.frame_skip+1):self.frame_skip,:,:,:,:]
        assert frames.shape[0] == ret_len
        return frames.as_in_context(self.ctx)

    def reset(self):
        # first check if file list has been updated
        if self.auto_add_file:
            parent_dir = os.path.join(*self.rec_paths[self.dataset_ind].split('/')[0:-1])
            parent_dir = '/'+parent_dir
            all_files = os.listdir(parent_dir)
            all_rec_files = [filename for filename in all_files if filename[-3:]=='rec']
            if set(all_rec_files) != set(self.rec_paths):
                print('find new files:{}'.format(set(all_rec_files)-set(self.rec_paths)))
                self.dataset_ind = 0
                self.rec_paths = [os.path.join(parent_dir, filename) for filename in all_rec_files]
            else:
                self.dataset_ind = (self.dataset_ind + 1) % len(self.rec_paths)
        else:
            self.dataset_ind = (self.dataset_ind + 1) % len(self.rec_paths)
        next_file = self.rec_paths[self.dataset_ind]
        print('switching to {}'.format(next_file))    
        self.image_iterator = mx.io.ImageRecordIter(
                                    path_imgrec=next_file,
                                    data_shape=(1, cfg.SZO.DATA.SIZE, cfg.SZO.DATA.SIZE),
                                    batch_size = self.batch_size*cfg.SZO.DATA.TOTAL_LEN,
                                    )
        self.image_iterator.reset()
        new_lst = next_file.split('.')[0] + '.lst'
        self.load_lst(new_lst)
        self.seq_ind = 0
    '''
    def resize(self, arr_nd):
        # the array should be of shape (seqlen*batch_size, channel, height, width)
        arr_nd = mx.nd.contrib.BilinearResize2D(arr_nd, self.height, self.width)
        arr_nd = arr_nd.reshape([cfg.SZO.DATA.TOTAL_LEN, 1, self.height, self.width])
        return arr_nd
    '''

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
    """
    np_seqs has shape (seqlen, H, W), pixel value [0,255], np.uint8
    path is save path
    """
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

def save_gif_examples(num, train_iterator=None, save_path=None):
    if train_iterator is None:
        train_iterator = SZOIterator(rec_paths=cfg.SZO_TRAIN_DATA_PATHS,
                                     in_len=cfg.MODEL.IN_LEN,
                                     out_len=cfg.MODEL.OUT_LEN,
                                     batch_size=1,
                                     frame_skip=1,
                                     ctx=mx.gpu()) 
    if save_path is None:
        save_path = 'data_display'

    for i in range(num):
        train_batch = train_iterator.sample()
        if cfg.MODEL.DATA_MODE == 'original':
            wb_examples = train_batch[:,0,0,:,:].asnumpy().astype(np.uint8)
            bb_examples = ((wb_examples*(wb_examples<255))*(255/80)).astype(np.uint8)
        elif cfg.MODEL.DATA_MODE == 'rescaled':
            epsilon = cfg.MODEL.DISPLAY_EPSILON
            bb_examples = train_batch[:,0,0,:,:].asnumpy()
            bb_examples = bb_examples.astype(np.uint8)
            mask = bb_examples<epsilon
            wb_examples = (bb_examples*(1-mask)*(80/255) + mask*255).astype(np.uint8)
        else:
            raise NotImplementedError

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        wb_name_str = os.path.join(save_path, 'white_back')
        bb_name_str = os.path.join(save_path, 'black_back')

        save_hko_gif(wb_examples, save_path=wb_name_str +str(i) + '.gif', multiply_by_255=False)
        save_hko_gif(bb_examples, save_path=bb_name_str + str(i) + '.gif', multiply_by_255=False)

        save_png_sequence(wb_examples, path=wb_name_str+'_'+str(i)+'.png')
        save_png_sequence(bb_examples, path=bb_name_str+'_'+str(i)+'.png')

# Simple test for the performance of the HKO iterator.
if __name__ == '__main__':
    np.random.seed(123)
    '''
    rec_path = os.path.join(cfg.ROOT_DIR, 'szo_data', 'train', 'TRAIN_1_2.rec')
    save_path = 'data_display'
    train_iterator = SZOIterator(rec_paths=rec_path,
                                 in_len=5,
                                 out_len=5,
                                 batch_size=32,
                                 frame_skip=6,  # 
                                 ctx=mx.gpu())
    save_gif_examples(20, train_iterator=train_iterator, save_path=save_path)
    '''
    save_gif_examples(20)
