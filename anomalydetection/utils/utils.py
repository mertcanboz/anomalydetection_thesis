import csv

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn import functional as F

Q = 3
M = 5
WINDOW_SIZE = 1440


def spectral_residual(values):
    """
    This method transform a time series into spectral residual series
    :param values: list.
        a list of float values.
    :return: mag: list.
        a list of float values as the spectral residual values
    """
    EPS = 1e-8
    trans = np.fft.fft(values)
    mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)

    maglog = [np.log(item) if abs(item) > EPS else 0 for item in mag]

    spectral = np.exp(maglog - average_filter(maglog, n=Q))

    trans.real = [ireal * ispectral / imag if abs(imag) > EPS else 0
                  for ireal, ispectral, imag in zip(trans.real, spectral, mag)]
    trans.imag = [iimag * ispectral / imag if abs(imag) > EPS else 0
                  for iimag, ispectral, imag in zip(trans.imag, spectral, mag)]

    wave_r = np.fft.ifft(trans)
    mag = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)

    return mag


def average_filter(values, n=3):
    """
    Calculate the sliding window average for the give time series.
    Mathematically, res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)
    :param values: list.
        a list of float numbers
    :param n: int, default 3.
        window size.
    :return res: list.
        a list of value after the average_filter process.
    """

    if n >= len(values):
        n = len(values)

    res = np.cumsum(values, dtype=float)
    res[n:] = res[n:] - res[:-n]
    res[n:] = res[n:] / n

    for i in range(1, n):
        res[i] /= (i + 1)

    return res


def predict_next(values):
    """
    Predicts the next value by sum up the slope of the last value with previous values.
    Mathematically, g = 1/m * sum_{i=1}^{m} g(x_n, x_{n-i}), x_{n+1} = x_{n-m+1} + g * m,
    where g(x_i,x_j) = (x_i - x_j) / (i - j)
    :param values: list.
        a list of float numbers.
    :return : float.
        the predicted next value.
    """

    if len(values) <= 1:
        raise ValueError(f'data should contain at least 2 numbers')

    v_last = values[-1]
    n = len(values)

    slopes = [(v_last - v) / (n - 1 - i) for i, v in enumerate(values[:-1])]

    return values[1] + sum(slopes)


def extend_series(values, extend_num=M, look_ahead=M):
    """
    extend the array data by the predicted next value
    :param values: list.
        a list of float numbers.
    :param extend_num: int, default 5.
        number of values added to the back of data.
    :param look_ahead: int, default 5.
        number of previous values used in prediction.
    :return: list.
        The result array.
    """

    if look_ahead < 1:
        raise ValueError('look_ahead must be at least 1')

    extension = [predict_next(values[-look_ahead - 2:-1])] * extend_num
    return np.concatenate((values, extension), axis=0)


class DataGenerator:
    """
    (optional) to inject anomalous points according to the formula in the paper:
    """

    def __init__(self, win_siz, step, nums):
        self.control = 0
        self.win_siz = win_siz
        self.step = step
        self.number = nums

    def generate_train_data(self, value, back_k=0, insert_anomaly=True):
        def normalize(a):
            amin = np.min(a)
            amax = np.max(a)
            a = (a - amin) / (amax - amin + 1e-5)
            return 3 * a

        if back_k <= 5:
            back = back_k
        else:
            back = 5
        length = len(value)
        tmp = []
        for pt in range(self.win_siz, length - back, self.step):
            head = max(0, pt - self.win_siz)
            tail = min(length - back, pt)
            data = np.array(value[head:tail])
            data = data.astype(np.float64)
            data = normalize(data)
            num = np.random.randint(1, self.number)
            ids = np.random.choice(self.win_siz, num, replace=False)
            lbs = np.zeros(self.win_siz, dtype=np.int64)
            if insert_anomaly:
                if (self.win_siz - 6) not in ids:
                    self.control += np.random.random()
                else:
                    self.control = 0
                if self.control > 100:
                    ids[0] = self.win_siz - 6
                    self.control = 0
                mean = np.mean(data)
                dataavg = average_filter(data)
                var = np.var(data)
                for id in ids:
                    data[id] += (dataavg[id] + mean) * np.random.randn() * min((1 + var), 10)
                    lbs[id] = 1
            tmp.append([data.tolist(), lbs.tolist()])
        return tmp


def cuda_if_available(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x.cpu()


def adjust_lr(optimizer, epoch, lr):
    cur_lr = lr * (0.5 ** ((epoch + 10) // 10))
    for param in optimizer.param_groups:
        param['lr'] = cur_lr


def Var(x):
    return Variable(cuda_if_available(x))


def loss_function(x, lb, model, weight_decay, win_size=WINDOW_SIZE):
    l2_reg = 0.
    for W in model.parameters():
        l2_reg = l2_reg + W.norm(2)
    kpiweight = torch.ones(lb.shape)
    kpiweight[lb == 1] = win_size // 100
    kpiweight = cuda_if_available(kpiweight)
    BCE = F.binary_cross_entropy(x, lb, weight=kpiweight, reduction='sum')
    return l2_reg * weight_decay + BCE


def calc(pred, true):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for pre, gt in zip(pred, true):
        if gt == 1:
            if pre == 1:
                TP += 1
            else:
                FN += 1
        if gt == 0:
            if pre == 1:
                FP += 1
            else:
                TN += 1
    print('TP=%d FP=%d TN=%d FN=%d' % (TP, FP, TN, FN))
    return TP, FP, TN, FN


class SrDataset(Dataset):
    '''
    Dataset implementation for transforming data to spectral residual scores on the fly.
    '''

    def __init__(self, width, data):
        self.genlen = 0
        self.len = self.genlen
        self.width = width
        self.kpinegraw = data
        self.negrawlen = len(self.kpinegraw)
        print('length :', len(self.kpinegraw))
        self.len += self.negrawlen
        self.kpineglen = 0
        self.control = 0.

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        idx = index % self.negrawlen
        datas = self.kpinegraw[idx]
        datas = np.array(datas)
        data = datas[0, :].astype(np.float64)
        lbs = datas[1, :].astype(np.float64)
        wave = spectral_residual(data)
        waveavg = average_filter(wave)
        for i in range(self.width):
            if wave[i] < 0.001 and waveavg[i] < 0.001:
                lbs[i] = 0
                continue
            ratio = wave[i] / waveavg[i]
            if ratio < 1.0 and lbs[i] == 1:
                lbs[i] = 0
            if ratio > 5.0:
                lbs[i] = 1
        srscore = abs(wave - waveavg) / (waveavg + 0.01)
        sortid = np.argsort(srscore)
        for idx in sortid[-2:]:
            if srscore[idx] > 3:
                lbs[idx] = 1
        resdata = torch.from_numpy(100 * wave)
        reslb = torch.from_numpy(lbs)
        return resdata, reslb


def load_kpi(csv_path):
    kpis = {}
    anomalies = 0
    with open(csv_path) as f:
        input = csv.reader(f, delimiter=',')
        cnt = 0
        for row in input:
            if cnt == 0:
                cnt += 1
                continue
            kpi = kpis.get(str(row[3]), [[], [], []])
            kpi[0].append(int(row[0]))  # timestamp
            kpi[1].append(float(row[1]))  # value
            kpi[2].append(int(row[2]))  # label
            kpis[str(row[3])] = kpi
            cnt += 1
            if int(row[2]) == 1:
                anomalies += 1
        print("Training data loaded. Total length: {}, number of anomalies: {}".format(cnt, anomalies))
        f.close()
    return kpis
