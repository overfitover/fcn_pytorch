import os
import numpy as np
import PIL.Image as Image

def getPalette():
    '''
    http://blog.csdn.net/yhl_leo/article/details/52185581
    '''
    pal = np.array([[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]], dtype='uint8').flatten()
    return pal


def colorize_mask(mask):
    """
    :param mask: 图片大小的数值，代表不同的颜色
    :return:
    """
    new_mask = Image.fromarray(mask.astype(np.uint8), 'P')  # 将二维数组转为图像

    pal = getPalette()
    new_mask.putpalette(pal)
    # print(new_mask.show())
    return new_mask

# m = np.array([[1,2], [3,4]])
# colorize_mask(m)


def getFileName(file_path):
    '''
    get file_path name from path+name+'test.jpg'
    return test
    '''
    full_name = file_path.split('/')[-1]
    name = os.path.splitext(full_name)[0]

    return name


def labelTopng(label, img_name):
    '''
    convert tensor cpu label to png and save
    '''
    label = label.numpy()             # 320 320
    label_pil = colorize_mask(label)
    label_pil.save(img_name)

def labelToimg(label):
    label = label.numpy()
    label_pil = colorize_mask(label)
    return label_pil


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def accuracy_score(label_trues, label_preds, n_class=21):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)   # n_class, n_class
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc