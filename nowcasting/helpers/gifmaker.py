import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,*([".."]*3))))
from PIL import Image
import numpy as np
import logging

def save_gif(single_seq, fname, multiply_by_255=True):
    """Save a single gif consisting of image sequence in single_seq to fname."""
    if multiply_by_255:
        logging.info('multiplying pixel values by 255...')
        img_seq = [Image.fromarray(img.astype(np.float32) * 255, 'F').convert("L") for img in single_seq]
    else:
        img_seq = [Image.fromarray(img.astype(np.float32), 'F').convert('L') for img in single_seq]
    img = img_seq[0]
    img.save(fname, save_all=True, append_images=img_seq[1:])

def save_gifs(seq, prefix):
    """Save several gifs.

    Args:
      seq: Shape (num_gifs, IMG_SIZE, IMG_SIZE)
      prefix: prefix-idx.gif will be the final filename.
    """

    for idx, single_seq in enumerate(seq):
        save_gif(single_seq, "{}-{}.gif".format(prefix, idx))
