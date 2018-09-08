import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,*([".."]*2))))
import numpy as np
import os
import yaml
import logging
from collections import OrderedDict
from .helpers.ordered_easydict import OrderedEasyDict as edict

__C = edict()
cfg = __C  # type: edict()

# Random seed
__C.SEED = None

# Dataset name
# Used by symbols factories who need to adjust for different
# inputs based on dataset used. Should be set by the script.
__C.DATASET = None

# Project directory, since config.py is supposed to be in $ROOT_DIR/nowcasting
__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

__C.MNIST_PATH = os.path.join(__C.ROOT_DIR, 'mnist_data')
if not os.path.exists(__C.MNIST_PATH):
    os.makedirs(__C.MNIST_PATH)
__C.HKO_DATA_BASE_PATH = os.path.join(__C.ROOT_DIR, 'hko_data')

__C.SZO_DATA_BASE_PATH = os.path.join(__C.ROOT_DIR, 'szo_data')
train_path = os.path.join(__C.SZO_DATA_BASE_PATH, 'train')
test_path = os.path.join(__C.SZO_DATA_BASE_PATH, 'test')
__C.SZO_TRAIN_DATA_PATHS = [os.path.join(train_path, filename) for filename in os.listdir(train_path) if filename[-4:]=='.rec']
__C.SZO_TEST_DATA_PATHS = [os.path.join(test_path, filename) for filename in os.listdir(test_path) if filename[-4:]=='.rec']

# Append your path to the possible paths
possible_hko_png_paths = [os.path.join('E:\\datasets\\HKO-data\\radarPNG\\radarPNG'),
                          os.path.join(__C.HKO_DATA_BASE_PATH, 'radarPNG')]
possible_hko_mask_paths = [os.path.join('E:\\datasets\\HKO-data\\radarPNG\\radarPNG_mask'),
                           os.path.join(__C.HKO_DATA_BASE_PATH, 'radarPNG_mask')]
# Search for the radarPNG
find_hko_png_path = False
for ele in possible_hko_png_paths:
    if os.path.exists(ele):
        find_hko_png_path = True
        __C.HKO_PNG_PATH = ele
        break
if not find_hko_png_path:
    pass
    #raise RuntimeError("radarPNG is not found! You can download the radarPNG using" 
    #" `bash download_radar_png.bash`")
# Search for the radarPNG_mask
find_hko_mask_path = False
for ele in possible_hko_mask_paths:
    if os.path.exists(ele):
        find_hko_mask_path = True
        __C.HKO_MASK_PATH = ele
        break
if not find_hko_mask_path:
    pass
    #raise RuntimeError("radarPNG_mask is not found! You can download the radarPNG_mask using"
    #                   " `bash download_radar_png.bash`")
if not os.path.exists(__C.HKO_DATA_BASE_PATH):
    os.makedirs(__C.HKO_DATA_BASE_PATH)
__C.HKO_PD_BASE_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'pd')
if not os.path.exists(__C.HKO_PD_BASE_PATH):
    os.makedirs(__C.HKO_PD_BASE_PATH)
__C.HKO_VALID_DATETIME_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'valid_datetime.pkl')
__C.HKO_SORTED_DAYS_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'sorted_day.pkl')
__C.HKO_RAINY_TRAIN_DAYS_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'hko7_rainy_train_days.txt')
__C.HKO_RAINY_VALID_DAYS_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'hko7_rainy_valid_days.txt')
__C.HKO_RAINY_TEST_DAYS_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'hko7_rainy_test_days.txt')

__C.HKO_PD = edict()
__C.HKO_PD.ALL = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_all.pkl')
__C.HKO_PD.ALL_09_14 = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_all_09_14.pkl')
__C.HKO_PD.ALL_15 = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_all_15.pkl')
__C.HKO_PD.RAINY_TRAIN = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_rainy_train.pkl')
__C.HKO_PD.RAINY_VALID = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_rainy_valid.pkl')
__C.HKO_PD.RAINY_TEST = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_rainy_test.pkl')

__C.HKO = edict()
__C.HKO.ITERATOR = edict()
__C.HKO.ITERATOR.WIDTH = 480
__C.HKO.ITERATOR.HEIGHT = 480
__C.HKO.ITERATOR.FILTER_RAINFALL = True           # Whether to discard part of the rainfall, has a denoising effect
__C.HKO.ITERATOR.FILTER_RAINFALL_THRESHOLD = 0.28 # All the pixel values that are smaller than round(threshold * 255) will be discarded


# The Benchmark parameters
__C.HKO.BENCHMARK = edict()
__C.HKO.BENCHMARK.STAT_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'benchmark_stat')
if not os.path.exists(__C.HKO.BENCHMARK.STAT_PATH):
    os.makedirs(__C.HKO.BENCHMARK.STAT_PATH)
__C.HKO.BENCHMARK.VISUALIZE_SEQ_NUM = 10  # Number of sequences that will be plotted and saved to the benchmark directory
__C.HKO.BENCHMARK.IN_LEN = 5   # The maximum input length to ensure that all models are tested on the same set of input data
__C.HKO.BENCHMARK.OUT_LEN = 20  # The maximum output length to ensure that all models are tested on the same set of input data
__C.HKO.BENCHMARK.STRIDE = 5   # The stride


__C.HKO.EVALUATION = edict()
__C.HKO.EVALUATION.ZR = edict()
__C.HKO.EVALUATION.ZR.a = 58.53  # The a factor in the Z-R relationship
__C.HKO.EVALUATION.ZR.b = 1.56  # The b factor in the Z-R relationship
__C.HKO.EVALUATION.THRESHOLDS = (0.5, 2.0, 5.0, 10.0, 30.0)
__C.HKO.EVALUATION.BALANCING_WEIGHTS = (1.0, 1.0, 2.0, 5.0, 10.0, 30.0)  # The corresponding balancing weights
__C.HKO.EVALUATION.CENTRAL_REGION = (120, 120, 360, 360)



__C.SZO = edict()
__C.SZO.DATA = edict()
__C.SZO.DATA.SIZE = 500
__C.SZO.DATA.TOTAL_LEN = 61
__C.SZO.DATA.IMAGE_CHANNEL = 2

__C.SZO.ITERATOR = edict()
__C.SZO.ITERATOR.DOWN_RATIO = 1
assert __C.SZO.DATA.SIZE % __C.SZO.ITERATOR.DOWN_RATIO == 0
__C.SZO.ITERATOR.RESIZED_SIZE = __C.SZO.DATA.SIZE // __C.SZO.ITERATOR.DOWN_RATIO
#__C.SZO.ITERATOR.FILTER_RAINFALL = True           # Whether to discard part of the rainfall, has a denoising effect
#__C.SZO.ITERATOR.FILTER_RAINFALL_THRESHOLD = 0.28 # All the pixel values that are smaller than round(threshold * 255) will be discarded



__C.SZO.EVALUATION = edict()
__C.SZO.EVALUATION.THRESHOLD_WEIGHTS = (1, 2, 3, 4, 5, 6) #  (1, 1, 2, 5, 10, 30)  # The corresponding balancing weights
__C.SZO.EVALUATION.THRESHOLDS = (10, 20, 30, 50, 60, 70)  # (0.5, 2, 5, 10, 30)
__C.SZO.EVALUATION.TEMPORAL_WEIGHT_SLOPE = 0.1  # start with 1
__C.SZO.EVALUATION.ZR = edict()
__C.SZO.EVALUATION.ZR.a = 58.53  # The a factor in the Z-R relationship
__C.SZO.EVALUATION.ZR.b = 1.56  # The b factor in the Z-R relationship
__C.SZO.EVALUATION.CENTRAL_REGION = (120, 120, 360, 360)

__C.MOVINGMNIST = edict()
__C.MOVINGMNIST.DISTRACTOR_NUM = 0
__C.MOVINGMNIST.VELOCITY_LOWER = 0.0
__C.MOVINGMNIST.VELOCITY_UPPER = 3.6
__C.MOVINGMNIST.SCALE_VARIATION_LOWER = 1/1.1
__C.MOVINGMNIST.SCALE_VARIATION_UPPER = 1.1
__C.MOVINGMNIST.ROTATION_LOWER = -30
__C.MOVINGMNIST.ROTATION_UPPER = 30
__C.MOVINGMNIST.ILLUMINATION_LOWER = 0.6
__C.MOVINGMNIST.ILLUMINATION_UPPER = 1.0
__C.MOVINGMNIST.DIGIT_NUM = 3
__C.MOVINGMNIST.IN_LEN = 10
__C.MOVINGMNIST.OUT_LEN = 10
__C.MOVINGMNIST.TESTING_LEN = 20
__C.MOVINGMNIST.IMG_SIZE = 64
__C.MOVINGMNIST.TEST_FILE = os.path.join(__C.MNIST_PATH, "movingmnist_10000_nodistr.npz")


__C.MODEL = edict()
__C.MODEL.RESUME = False  # If True, load checkpoint
__C.MODEL.TESTING = False # If True, run in Testing mode
__C.MODEL.LOAD_DIR = "" # The directory to load the pre-trained parameters
                        # Could be like `D:\\HKUST\\3-2\\NIPS2017\\hko_0502\\bal_loss_direct`
__C.MODEL.LOAD_ITER = 79999           # Only applicable when LOAD_DIR is non-empty
__C.MODEL.SAVE_DIR = ""
__C.MODEL.CNN_ACT_TYPE = "leaky"
__C.MODEL.RNN_ACT_TYPE = "leaky"
__C.MODEL.OPTFLOW_AS_INPUT = False
__C.MODEL.FRAME_STACK = 1          # Stack multiple frames as the input
__C.MODEL.FRAME_SKIP_IN = 1           # The frame skip size of input
__C.MODEL.FRAME_SKIP_OUT = 1
__C.MODEL.IN_LEN = 30               # Size of the input
__C.MODEL.OUT_LEN = 30             # Size of the output
__C.MODEL.OUT_TYPE = "direct"      # Can be "direct", or "DFN"
__C.MODEL.NORMAL_LOSS_GLOBAL_SCALE = 0.00005
__C.MODEL.USE_BALANCED_LOSS = True
__C.MODEL.TEMPORAL_WEIGHT_TYPE = "same"  # Can be "same", "linear" or "exponential"
__C.MODEL.TEMPORAL_WEIGHT_UPPER = 5      # Only applicable when temporal_weights_type is "linear" or "exponential"
                                         # If linear
                                         #   the weights will be increased following (1 + i * (upper - 1) / (T - 1))
                                         # If exponential
                                         #   the weights will be increased following exp^{i * \ln(upper) / (T-1)}
__C.MODEL.L1_LAMBDA = 1.0
__C.MODEL.L2_LAMBDA = 1.0
__C.MODEL.GDL_LAMBDA = 0.0
__C.MODEL.USE_SEASONALITY = False          # Whether to use seasonality
__C.MODEL.GAN_G_LAMBDA = 0.2
__C.MODEL.GAN_D_LAMBDA = 0.2
__C.MODEL.EXTEND_TO_FULL_OUTLEN = False

__C.MODEL.DATA_MODE = 'original'  # 'rescaled' or 'original'
__C.MODEL.BALANCE_FACTOR = 0.15  # under 'original' mode
# the following two don't have to be the same with SZO.EVALUATION._
__C.MODEL.THRESHOLDS = (10, 20, 30, 50)  #(0.5, 2, 5, 10, 30)  # under 'rescaled mode', in dbz
__C.MODEL.BALANCING_WEIGHTS = (1, 2, 4, 7, 11)  #(1, 1, 2, 5, 10, 30)  # The corresponding balancing weights
__C.MODEL.DISPLAY_EPSILON = 0.0  # under 'rescaled mode'

__C.MODEL.TRAJRNN = edict()
__C.MODEL.TRAJRNN.INIT_GRID = True
__C.MODEL.TRAJRNN.FLOW_LR_MULT = 1.0
__C.MODEL.TRAJRNN.SAVE_MID_RESULTS = False

__C.MODEL.ENCODER_FORECASTER = edict()
__C.MODEL.ENCODER_FORECASTER.HAS_MASK = False
__C.MODEL.ENCODER_FORECASTER.FEATMAP_SIZE = [123, 41, 20]
__C.MODEL.ENCODER_FORECASTER.USE_SKIP = False
__C.MODEL.ENCODER_FORECASTER.FIRST_CONV1 = (8, 8, 2, 0) #(8, 7, 5, 1)  # Num filter, kernel, stride, pad
__C.MODEL.ENCODER_FORECASTER.FIRST_CONV2 = (16, 6, 2, 3)
__C.MODEL.ENCODER_FORECASTER.LAST_DECONV1 = (8, 8, 2, 0)  # Num filter, kernel, stride, pad
__C.MODEL.ENCODER_FORECASTER.LAST_DECONV2 = (16, 6, 2, 3)
__C.MODEL.ENCODER_FORECASTER.DOWNSAMPLE = [(5, 3, 1),
                                           (3, 2, 1)]  # (kernel, stride, pad) for conv2d
__C.MODEL.ENCODER_FORECASTER.UPSAMPLE = [(5, 3, 1),
                                         (3, 2, 1)]  # (kernel, stride, pad) for deconv2d

__C.MODEL.ENCODER_FORECASTER.RNN_BLOCKS = edict()    # Define the RNN blocks for the encoder RNN
                                                     # In our network, the forecaster RNN will always have the reverse structure of encoder RNN
__C.MODEL.ENCODER_FORECASTER.RNN_BLOCKS.RES_CONNECTION = True
__C.MODEL.ENCODER_FORECASTER.RNN_BLOCKS.LAYER_TYPE = ["ConvGRU", "ConvGRU", "ConvGRU"]
__C.MODEL.ENCODER_FORECASTER.RNN_BLOCKS.STACK_NUM = [2, 3, 3]
# These features are used for both ConvGRU
__C.MODEL.ENCODER_FORECASTER.RNN_BLOCKS.NUM_FILTER = [32, 64, 64]
__C.MODEL.ENCODER_FORECASTER.RNN_BLOCKS.H2H_KERNEL = [(5, 5), (5, 5), (3, 3)]
__C.MODEL.ENCODER_FORECASTER.RNN_BLOCKS.H2H_DILATE = [(1, 1), (1, 1), (1, 1)]
__C.MODEL.ENCODER_FORECASTER.RNN_BLOCKS.I2H_KERNEL = [(3, 3), (3, 3), (3, 3)]
__C.MODEL.ENCODER_FORECASTER.RNN_BLOCKS.I2H_PAD = [(1, 1), (1, 1), (1, 1)]
# These features are only used in TrajGRU
__C.MODEL.ENCODER_FORECASTER.RNN_BLOCKS.L = [5, 5, 5]

__C.MODEL.DISCRIMINATOR = edict()
__C.MODEL.DISCRIMINATOR.USE_2D = False
__C.MODEL.DISCRIMINATOR.PIXEL = False
__C.MODEL.DISCRIMINATOR.DOWNSAMPLE_VIDEO = [1, 5, 5]
__C.MODEL.DISCRIMINATOR.FEATMAP_SIZE = [[20, 500], [10, 250], [5, 125], [2, 62]]
__C.MODEL.DISCRIMINATOR.DISCRIM_CONV = [edict({'num_filter':32, 'kernel':[3, 3, 3], 'stride':[1, 1, 1],'padding':[1, 1, 1]}),
                                        edict({'num_filter':64, 'kernel':[3, 3, 3], 'stride':[1, 1, 1],'padding':[1, 1, 1]}),
                                        edict({'num_filter':128, 'kernel':[3, 3, 3], 'stride':[1, 1, 1],'padding':[1, 1, 1]}),
                                        edict({'num_filter':192, 'kernel':[3, 3, 3], 'stride':[1, 1, 1],'padding':[1, 1, 1]})]
__C.MODEL.DISCRIMINATOR.DISCRIM_POOL = [edict({'num_filter':32, 'kernel':[4, 4, 4], 'stride':[2, 2, 2],'padding':[2, 2, 2]}),
                                       edict({'num_filter':64, 'kernel':[4, 4, 4], 'stride':[2, 2, 2],'padding':[2, 2, 2]}),
                                       edict({'num_filter':128, 'kernel':[3, 3, 3], 'stride':[2, 2, 2],'padding':[0, 0, 0]})]
__C.MODEL.DISCRIMINATOR.SPECTRAL_NORMALIZE_FACTOR = 0.1


__C.MODEL.DECONVBASELINE = edict()
__C.MODEL.DECONVBASELINE.BASE_NUM_FILTER = 16
__C.MODEL.DECONVBASELINE.USE_3D = True
__C.MODEL.DECONVBASELINE.ENCODER = "separate"
__C.MODEL.DECONVBASELINE.BN = True
__C.MODEL.DECONVBASELINE.BN_GLOBAL_STATS = False
__C.MODEL.DECONVBASELINE.COMPAT = edict() # Compatibility flags to recover behavior of previous versions
__C.MODEL.DECONVBASELINE.COMPAT.CONV_INSTEADOF_FC_IN_ENCODER = False # Until 6th May 2017
__C.MODEL.DECONVBASELINE.FC_BETWEEN_ENCDEC = 0

__C.MODEL.TRAIN = edict()
__C.MODEL.TRAIN.BATCH_SIZE = 4
__C.MODEL.TRAIN.DISCRIM_LOOP = 3
__C.MODEL.TRAIN.GEN_BUFFER_LEN = 4
__C.MODEL.TRAIN.TBPTT = False
__C.MODEL.TRAIN.OPTIMIZER = "adam"
__C.MODEL.TRAIN.LR = 1E-4
__C.MODEL.TRAIN.GAMMA1 = 0.9   # Used in RMSProp
__C.MODEL.TRAIN.BETA1 = 0.5    # When using ADAM, momentum is called beta1
__C.MODEL.TRAIN.EPS = 1E-8
__C.MODEL.TRAIN.MIN_LR = 1E-6
__C.MODEL.TRAIN.GRAD_CLIP = 50.0
__C.MODEL.TRAIN.GRAD_CLIP_DIS = 1.0
__C.MODEL.TRAIN.WD = 0
__C.MODEL.TRAIN.MAX_ITER = 180000
__C.MODEL.VALID_ITER = 5000
__C.MODEL.VALID_LOOP = 10
__C.MODEL.SAVE_ITER = 15000
__C.MODEL.TEMP_SAVE_ITER = 500
__C.MODEL.TRAIN.LR_DECAY_ITER = 10000
__C.MODEL.TRAIN.LR_DECAY_FACTOR = 0.7
__C.MODEL.TRAIN.LR_DIS = 1E-4
__C.MODEL.TRAIN.MIN_LR_DIS = 1E-8
__C.MODEL.TRAIN.LR_DECAY_ITER_DIS = 800
__C.MODEL.TRAIN.LR_DECAY_FACTOR_DIS = 0.7
__C.MODEL.TRAIN.OPTIMIZER_DIS = 'rmsprop'

__C.MODEL.DRAW_EVERY = 100
__C.MODEL.DISPLAY_EVERY = 100

__C.MODEL.TEST = edict()
__C.MODEL.TEST.FINETUNE = True
__C.MODEL.TEST.MAX_ITER = 1      # Number of samples to generate in testing mode
__C.MODEL.TEST.MODE = "online"    # Can be `online` or `fixed`
__C.MODEL.TEST.DISABLE_TBPTT = True
__C.MODEL.TEST.ONLINE = edict()
__C.MODEL.TEST.ONLINE.OPTIMIZER = "adagrad"
__C.MODEL.TEST.ONLINE.LR = 1E-4
__C.MODEL.TEST.ONLINE.FINETUNE_MIN_MSE = 0.0
__C.MODEL.TEST.ONLINE.GAMMA1 = 0.9    # Used in RMSProp
__C.MODEL.TEST.ONLINE.BETA1 = 0.5     # Used in ADAM!
__C.MODEL.TEST.ONLINE.EPS = 1E-6
__C.MODEL.TEST.ONLINE.GRAD_CLIP = 50.0
__C.MODEL.TEST.ONLINE.WD = 0


def _merge_two_config(user_cfg, default_cfg):
    """ Merge user's config into default config dictionary, clobbering the
        options in b whenever they are also specified in a.
        Need to ensure the type of two val under same key are the same
        Do recursive merge when encounter hierarchical dictionary
    """
    if type(user_cfg) is not edict:
        return
    for key, val in user_cfg.items():
        # Since user_cfg is a sub-file of default_cfg
        if not key in default_cfg:
            raise KeyError('{} is not a valid config key'.format(key))

        if (type(default_cfg[key]) is not type(val) and
                default_cfg[key] is not None):
            if isinstance(default_cfg[key], np.ndarray):
                val = np.array(val, dtype=default_cfg[key].dtype)
            else:
                raise ValueError(
                     'Type mismatch ({} vs. {}) '
                     'for config key: {}'.format(type(default_cfg[key]),
                                                 type(val), key))
        # Recursive merge config
        if type(val) is edict:
            try:
                _merge_two_config(user_cfg[key], default_cfg[key])
            except:
                print('Error under config key: {}'.format(key))
                raise
        else:
            default_cfg[key] = val


def cfg_from_file(file_name, target=__C):
    """ Load a config file and merge it into the default options.
    """
    import yaml
    with open(file_name, 'r') as f:
        print('Loading YAML config file from %s' %f)
        yaml_cfg = edict(yaml.load(f))

    _merge_two_config(yaml_cfg, target)


def ordered_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items(), flow_style=False)

    def _ndarray_representer(dumper, data):
        return dumper.represent_list(data.tolist())

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    OrderedDumper.add_representer(edict, _dict_representer)
    OrderedDumper.add_representer(np.ndarray, _ndarray_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def save_cfg(dir_path, source=__C):
    cfg_count = 0
    file_path = os.path.join(dir_path, 'cfg%d.yml' %cfg_count)
    while os.path.exists(file_path):
        cfg_count += 1
        file_path = os.path.join(dir_path, 'cfg%d.yml' % cfg_count)
    with open(file_path, 'w') as f:
        logging.info("Save YAML config file to %s" %file_path)
        ordered_dump(source, f, yaml.SafeDumper, default_flow_style=None)


def load_latest_cfg(dir_path, target=__C):
    import re
    cfg_count = None
    source_cfg_path = None
    for fname in os.listdir(dir_path):
        ret = re.search('cfg(\d+)\.yml', fname)
        if ret != None:
            if cfg_count is None or (int(re.group(1)) > cfg_count):
                cfg_count = int(re.group(1))
                source_cfg_path = os.path.join(dir_path, ret.group(0))
    cfg_from_file(file_name=source_cfg_path, target=target)
