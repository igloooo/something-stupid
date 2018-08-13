import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,*([".."]*2))))
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
import logging
import os
from collections import namedtuple
from nowcasting.config import cfg
from nowcasting.helpers.msssim import _SSIMForMultiScale
from numba import jit, float32, boolean, int32, float64


def pixel_to_rainfall(img, a=None, b=None):
    """Convert the pixel values to real rainfall intensity

    Parameters
    ----------
    img : np.ndarray, between 0 and 1
    a : float32, optional
    b : float32, optional

    Returns
    -------
    rainfall_intensity : np.ndarray
    """
    '''
    if a is None:
        a = cfg.SZO.EVALUATION.ZR.a
    if b is None:
        b = cfg.SZO.EVALUATION.ZR.b
    dBZ = img*80.0
    dBR = (dBZ - 10.0 * np.log10(a)) / b
    rainfall_intensity = np.power(10, dBR / 10.0)
    return rainfall_intensity 
    '''
    return img * 80


def rainfall_to_pixel(rainfall_intensity, a=None, b=None):
    """Convert the rainfall intensity to pixel values

    Parameters
    ----------
    rainfall_intensity : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    pixel_vals : np.ndarray
    """
    '''
    if a is None:
        a =cfg.SZO.EVALUATION.ZR.a
    if b is None:
        b = cfg.SZO.EVALUATION.ZR.b
    dBR = np.log10(rainfall_intensity) * 10.0
    dBZ = dBR * b + 10.0 * np.log10(a)
    return dBZ / 80
    '''
    return rainfall_intensity / 80


def get_hit_miss_counts(prediction, truth, mask=None, thresholds=None):
    """This function calculates the overall hits and misses for the prediction, which could be used
    to get the skill scores and threat scores:


    This function assumes the input, i.e, prediction and truth are 3-dim tensors, (timestep, row, col)
    and all inputs should be between 0~1

    Parameters
    ----------
    prediction : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    truth : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    mask : np.ndarray or None
        Shape: (seq_len, batch_size, 1, height, width)
        0 --> not use
        1 --> use
    thresholds : list or tuple

    Returns
    -------
    hits : np.ndarray
        (seq_len, batch_size, len(thresholds))
        TP
    misses : np.ndarray
        (seq_len, batch_size, len(thresholds))
        FN
    false_alarms : np.ndarray
        (seq_len, batch_size, len(thresholds))
        FP
    correct_negatives : np.ndarray
        (seq_len, batch_size, len(thresholds))
        TN
    """
    if thresholds is None:
        thresholds = cfg.SZO.EVALUATION.THRESHOLDS
    assert 5 == prediction.ndim
    assert 5 == truth.ndim
    assert prediction.shape == truth.shape
    assert prediction.shape[2] == 1
    thresholds = rainfall_to_pixel(np.array(thresholds,
                                            dtype=np.float32)
                                   .reshape((1, 1, len(thresholds), 1, 1)))
    bpred = (prediction >= thresholds)
    btruth = (truth >= thresholds)
    bpred_n = np.logical_not(bpred)
    btruth_n = np.logical_not(btruth)
    summation_axis = (3, 4)
    if mask is None:
        hits = np.logical_and(bpred, btruth).sum(axis=summation_axis)
        misses = np.logical_and(bpred_n, btruth).sum(axis=summation_axis)
        false_alarms = np.logical_and(bpred, btruth_n).sum(axis=summation_axis)
        correct_negatives = np.logical_and(bpred_n, btruth_n).sum(axis=summation_axis)
    else:
        hits = np.logical_and(np.logical_and(bpred, btruth), mask)\
            .sum(axis=summation_axis)
        misses = np.logical_and(np.logical_and(bpred_n, btruth), mask)\
            .sum(axis=summation_axis)
        false_alarms = np.logical_and(np.logical_and(bpred, btruth_n), mask)\
            .sum(axis=summation_axis)
        correct_negatives = np.logical_and(np.logical_and(bpred_n, btruth_n), mask)\
            .sum(axis=summation_axis)
    return hits, misses, false_alarms, correct_negatives


def get_hit_miss_counts_numba(prediction, truth, mask, thresholds=None):
    """This function calculates the overall hits and misses for the prediction, which could be used
    to get the skill scores and threat scores:


    This function assumes the input, i.e, prediction and truth are 3-dim tensors, (timestep, row, col)
    and all inputs should be between 0~1

    Parameters
    ----------
    prediction : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    truth : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    mask : np.ndarray or None
        Shape: (seq_len, batch_size, 1, height, width)
        0 --> not use
        1 --> use
    thresholds : list or tuple

    Returns
    -------
    hits : np.ndarray
        (seq_len, batch_size, len(thresholds))
        TP
    misses : np.ndarray
        (seq_len, batch_size, len(thresholds))
        FN
    false_alarms : np.ndarray
        (seq_len, batch_size, len(thresholds))
        FP
    correct_negatives : np.ndarray
        (seq_len, batch_size, len(thresholds))
        TN
    """
    if thresholds is None:
        thresholds = cfg.SZO.EVALUATION.THRESHOLDS
    assert 5 == prediction.ndim
    assert 5 == truth.ndim
    assert prediction.shape == truth.shape
    assert prediction.shape[2] == 1
    thresholds = [rainfall_to_pixel(thresholds[i]) for i in range(len(thresholds))]
    thresholds = sorted(thresholds)
    ret = _get_hit_miss_counts_numba(prediction=prediction,
                                     truth=truth,
                                     mask=mask,
                                     thresholds=thresholds)
    return ret[:, :, :, 0], ret[:, :, :, 1], ret[:, :, :, 2], ret[:, :, :, 3]


@jit(int32(float32, float32, boolean, float32))
def _get_hit_miss_counts_numba(prediction, truth, mask, thresholds):
    seqlen, batch_size, _, height, width = prediction.shape
    threshold_num = len(thresholds)
    ret = np.zeros(shape=(seqlen, batch_size, threshold_num, 4), dtype=np.int32)

    for i in range(seqlen):
        for j in range(batch_size):
            for m in range(height):
                for n in range(width):
                    if mask[i][j][0][m][n]:
                        for k in range(threshold_num):
                            bpred = prediction[i][j][0][m][n] >= thresholds[k]
                            btruth = truth[i][j][0][m][n] >= thresholds[k]
                            ind = (1 - btruth) * 2 + (1 - bpred)
                            ret[i][j][k][ind] += 1
                            # The above code is the same as:
                            # ret[i][j][k][0] += bpred * btruth
                            # ret[i][j][k][1] += (1 - bpred) * btruth
                            # ret[i][j][k][2] += bpred * (1 - btruth)
                            # ret[i][j][k][3] += (1 - bpred) * (1- btruth)
    return ret


def get_correlation(prediction, truth):
    """

    Parameters
    ----------
    prediction : np.ndarray
    truth : np.ndarray

    Returns
    -------

    """
    assert truth.shape == prediction.shape
    assert 5 == prediction.ndim
    assert prediction.shape[2] == 1
    eps = 1E-12
    ret = (prediction * truth).sum(axis=(3, 4)) / (
        np.sqrt(np.square(prediction).sum(axis=(3, 4))) * np.sqrt(np.square(truth).sum(axis=(3, 4))) + eps)
    ret = ret.sum(axis=(1, 2))
    return ret


def get_PSNR(prediction, truth):
    """Peak Signal Noise Ratio

    Parameters
    ----------
    prediction : np.ndarray
    truth : np.ndarray

    Returns
    -------
    ret : np.ndarray
    """
    mse = np.square(prediction - truth).mean(axis=(2, 3, 4))
    ret = 10.0 * np.log10(1.0 / mse)
    ret = ret.sum(axis=1)
    return ret


def get_SSIM(prediction, truth):
    """Calculate the SSIM score following
    [TIP2004] Image Quality Assessment: From Error Visibility to Structural Similarity

    Same functionality as
    https://github.com/coupriec/VideoPredictionICLR2016/blob/master/image_error_measures.lua#L50-L75

    We use nowcasting.helpers.msssim, which is borrowed from Tensorflow to do the evaluation

    Parameters
    ----------
    prediction : np.ndarray
    truth : np.ndarray

    Returns
    -------
    ret : np.ndarray
    """
    assert truth.shape == prediction.shape
    assert 5 == prediction.ndim
    assert prediction.shape[2] == 1
    seq_len = prediction.shape[0]
    batch_size = prediction.shape[1]
    prediction = prediction.reshape((prediction.shape[0] * prediction.shape[1],
                                     prediction.shape[3], prediction.shape[4], 1))
    truth = truth.reshape((truth.shape[0] * truth.shape[1],
                           truth.shape[3], truth.shape[4], 1))
    ssim, cs = _SSIMForMultiScale(img1=prediction, img2=truth, max_val=1.0)
    print(ssim.shape)
    ret = ssim.reshape((seq_len, batch_size)).sum(axis=1)
    return ret


class SZOEvaluation(object):
    def __init__(self, seq_len, use_central, no_ssim=True, threholds=None,
                 central_region=None):
        if central_region is None:
            central_region = cfg.SZO.EVALUATION.CENTRAL_REGION
        self._thresholds = cfg.SZO.EVALUATION.THRESHOLDS if threholds is None else threholds
        self._seq_len = seq_len
        self._no_ssim = no_ssim
        self._use_central = use_central
        self._central_region = central_region
        self.begin()

    def begin(self):
        self._total_hits = np.zeros((self._seq_len, len(self._thresholds)), dtype=np.int)
        self._total_misses = np.zeros((self._seq_len, len(self._thresholds)),  dtype=np.int)
        self._total_false_alarms = np.zeros((self._seq_len, len(self._thresholds)), dtype=np.int)
        self._total_correct_negatives = np.zeros((self._seq_len, len(self._thresholds)),
                                                 dtype=np.int)
        self._ssim = np.zeros((self._seq_len,), dtype=np.float32)
        self._total_batch_num = 0

    def clear_all(self):
        self._total_hits[:] = 0
        self._total_misses[:] = 0
        self._total_false_alarms[:] = 0
        self._total_correct_negatives[:] = 0
        self._ssim[:] = 0
        self._total_batch_num = 0

    def update(self, gt, pred, mask):
        """

        Parameters
        ----------
        gt : np.ndarray
        pred : np.ndarray
        mask : np.ndarray
            0 indicates not use and 1 indicates that the location will be taken into account

        Returns
        -------

        """
        batch_size = gt.shape[1]
        assert gt.shape[0] == self._seq_len
        assert gt.shape == pred.shape
        assert gt.shape == mask.shape

        if self._use_central:
            # Crop the central regions for evaluation
            pred = pred[:, :, :,
                        self._central_region[1]:self._central_region[3],
                        self._central_region[0]:self._central_region[2]]
            gt = gt[:, :, :,
                    self._central_region[1]:self._central_region[3],
                    self._central_region[0]:self._central_region[2]]
            mask = mask[:, :, :,
                        self._central_region[1]:self._central_region[3],
                        self._central_region[0]:self._central_region[2]]
        self._total_batch_num += batch_size
        #TODO Save all the hits, misses, false_alarms and correct_negatives
        if not self._no_ssim:
            raise NotImplementedError
            # self._ssim += get_SSIM(prediction=pred, truth=gt)
        hits, misses, false_alarms, correct_negatives = \
            get_hit_miss_counts_numba(prediction=pred, truth=gt, mask=mask,
                                      thresholds=self._thresholds)
        self._total_hits += hits.sum(axis=1)
        self._total_misses += misses.sum(axis=1)
        self._total_false_alarms += false_alarms.sum(axis=1)
        self._total_correct_negatives += correct_negatives.sum(axis=1)

    def calculate_stat(self):
        """The following measurements will be used to measure the score of the forecaster

        See Also
        [Weather and Forecasting 2010] Equitability Revisited: Why the "Equitable Threat Score" Is Not Equitable
        http://www.wxonline.info/topics/verif2.html

        We will denote
        (a b    (hits       false alarms
         c d) =  misses   correct negatives)

        We will report the
        POD = a / (a + c)
        FAR = b / (a + b)
        CSI = a / (a + b + c)
        Heidke Skill Score (HSS) = 2(ad - bc) / ((a+c) (c+d) + (a+b)(b+d))
        Gilbert Skill Score (GSS) = HSS / (2 - HSS), also known as the Equitable Threat Score
            HSS = 2 * GSS / (GSS + 1)
        MSE = mask * (pred - gt) **2
        MAE = mask * abs(pred - gt)
        GDL = valid_mask_h * abs(gd_h(pred) - gd_h(gt)) + valid_mask_w * abs(gd_w(pred) - gd_w(gt))
        Returns
        -------

        """
        a = self._total_hits.astype(np.float64)
        b = self._total_false_alarms.astype(np.float64)
        c = self._total_misses.astype(np.float64)
        d = self._total_correct_negatives.astype(np.float64)
        pod = a / (a + c)
        far = b / (a + b)
        csi = a / (a + b + c)
        n = a + b + c + d
        aref = (a + b) / n * (a + c)
        gss = (a - aref) / (a + b + c - aref)
        hss = 2 * gss / (gss + 1)
        
        temporal_weights = [1+i*cfg.SZO.EVALUATION.TEMPORAL_WEIGHT_SLOPE for i in range(self._seq_len)]
        threshold_weights = cfg.SZO.EVALUATION.THRESHOLD_WEIGHTS
        temporal_weights = np.array(temporal_weights).reshape((self._seq_len,1))
        threshold_weights = np.array(threshold_weights).reshape((1,len(self._thresholds)))
        weighted_hss = np.sum(hss*temporal_weights*threshold_weights) / np.sum(temporal_weights*threshold_weights)
        if not self._no_ssim:
            raise NotImplementedError
            # ssim = self._ssim / self._total_batch_num
        # return pod, far, csi, hss, gss, mse, mae, gdl
        return pod, far, csi, hss, gss, weighted_hss

    def print_stat_readable(self, prefix=""):
        logging.info("%sTotal Sequence Number: %d, Use Central: %d"
                     %(prefix, self._total_batch_num, self._use_central))
        pod, far, csi, hss, gss, weighted_hss = self.calculate_stat()
        # pod, far, csi, hss, gss, mse, mae, gdl = self.calculate_stat()
        logging.info("   Hits: " + ', '.join([">%g:%g/%g" % (threshold,
                                                             self._total_hits[:, i].mean(),
                                                             self._total_hits[-1, i])
                                             for i, threshold in enumerate(self._thresholds)]))
        logging.info("   POD: " + ', '.join([">%g:%g/%g" % (threshold, pod[:, i].mean(), pod[-1, i])
                                  for i, threshold in enumerate(self._thresholds)]))
        logging.info("   FAR: " + ', '.join([">%g:%g/%g" % (threshold, far[:, i].mean(), far[-1, i])
                                  for i, threshold in enumerate(self._thresholds)]))
        logging.info("   CSI: " + ', '.join([">%g:%g/%g" % (threshold, csi[:, i].mean(), csi[-1, i])
                                  for i, threshold in enumerate(self._thresholds)]))
        logging.info("   GSS: " + ', '.join([">%g:%g/%g" % (threshold, gss[:, i].mean(), gss[-1, i])
                                             for i, threshold in enumerate(self._thresholds)]))
        logging.info("   HSS: " + ', '.join([">%g:%g/%g" % (threshold, hss[:, i].mean(), hss[-1, i])
                                             for i, threshold in enumerate(self._thresholds)]))
        logging.info("   weighted_HSS: %g"%(weighted_hss))
        if not self._no_ssim:
            raise NotImplementedError

    def save_pkl(self, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        f = open(path, 'wb')
        logging.info("Saving SZOEvaluation to %s" %path)
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def save_txt_readable(self, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        pod, far, csi, hss, gss, weighted_hss = self.calculate_stat()
        # pod, far, csi, hss, gss, mse, mae, gdl = self.calculate_stat()
        f = open(path, 'w')
        logging.info("Saving readable txt of SZOEvaluation to %s" % path)
        f.write("Total Sequence Num: %d, Out Seq Len: %d, Use Central: %d\n"
                %(self._total_batch_num,
                  self._seq_len,
                  self._use_central))
        for (i, threshold) in enumerate(self._thresholds):
            f.write("Threshold = %g:\n" %threshold)
            f.write("   POD: %s\n" %str(list(pod[:, i])))
            f.write("   FAR: %s\n" % str(list(far[:, i])))
            f.write("   CSI: %s\n" % str(list(csi[:, i])))
            f.write("   GSS: %s\n" % str(list(gss[:, i])))
            f.write("   HSS: %s\n" % str(list(hss[:, i])))
            f.write("   POD stat: avg %g/final %g\n" %(pod[:, i].mean(), pod[-1, i]))
            f.write("   FAR stat: avg %g/final %g\n" %(far[:, i].mean(), far[-1, i]))
            f.write("   CSI stat: avg %g/final %g\n" %(csi[:, i].mean(), csi[-1, i]))
            f.write("   GSS stat: avg %g/final %g\n" %(gss[:, i].mean(), gss[-1, i]))
            f.write("   HSS stat: avg %g/final %g\n" % (hss[:, i].mean(), hss[-1, i]))
        f.write("Weigthed HSS: %g\n"%(weighted_hss))
        f.close()

    def save(self, prefix):
        self.save_txt_readable(prefix + ".txt")
        self.save_pkl(prefix + ".pkl")

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    evaluator = SZOEvaluation(10,False)

    # case1, identical, HSS supposed to be 1
    print('case 1')
    pred_np = np.random.uniform(size=[10,4,1,500,500])
    gt_np = pred_np
    mask_np = np.ones([10,4,1,500,500])
    evaluator.update(gt_np, pred_np, mask_np)
    evaluator.print_stat_readable()
    evaluator.clear_all()

    # case2, independent, HSS supposed to be 0
    print('case 2')
    pred_np = np.random.uniform(size=[10,4,1,500,500])
    gt_np = np.random.uniform(size=[10,4,1,500,500])
    evaluator.update(gt_np, pred_np, mask_np)
    evaluator.print_stat_readable()
    evaluator.clear_all()

    # test mask, nan every where
    print('test mask')
    mask_np = np.zeros_like(mask_np)
    evaluator.update(gt_np, pred_np, mask_np)
    evaluator.print_stat_readable()
    evaluator.clear_all()

    # case3, avoiding, HSS supposed to be 0 on level 3, nan on level 1 and 2, 4
    print('case 3')
    thresholds = [rainfall_to_pixel(cfg.SZO.EVALUATION.THRESHOLDS[i]) for i in range(len(cfg.SZO.EVALUATION.THRESHOLDS))]
    pred_np = np.random.uniform(low=thresholds[2], high=thresholds[3], size=[10,4,1,500,500])
    gt_np = np.random.uniform(low=thresholds[1], high=thresholds[2], size=[10,4,1,500,500])
    mask_np = np.ones_like(mask_np)
    evaluator.update(gt_np, pred_np, mask_np)
    evaluator.print_stat_readable()
    evaluator.clear_all()

    # test if save information over time
    print('continual test')
    for i in range(10):
        pred_np = np.random.uniform(size=[10,4,1,500,500])
        if np.random.uniform()>0.5:
            gt_np = pred_np
        else:
            gt_np = np.random.uniform(size=[10,4,1,500,500])
        evaluator.update(gt_np, pred_np, mask_np)
        evaluator.print_stat_readable()
    evaluator.save('temp_test/test_evaluator')
    evaluator.clear_all()

    # test saved pickle
    with open('temp_test/test_evaluator.pkl', 'rb') as f:
        obj = pickle.load(f)
        print('loading complete')
        obj.print_stat_readable()  # supposed to be same with last printed record

    