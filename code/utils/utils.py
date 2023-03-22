import random
import torch
import torch.nn.functional as F
import numpy as np
import math


def get_patch(*args, patch_size=17, scale=1):
    """
    Get patch from an image
    """
    if len(args) > 1:
        ih, iw, _ = args[0].shape

        ip = patch_size
        tp = scale * ip

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = scale * ix, scale * iy

        ret = [
            args[0][iy:iy + ip, ix:ix + ip, :],
            *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
        ]
    else:
        th, tw = args[0].shape[:2]  ## HR image

        tp = scale * patch_size

        tx = random.randrange(0, (tw - tp+1))
        ty = random.randrange(0, (th - tp+1))

        return args[0][ty:ty + tp, tx:tx + tp, :]

    return ret


def np2Tensor(*args, rgb_range=255, n_colors=1):
    def _np2Tensor(img):
        img = img.astype('float64')
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # NHWC -> NCHW
        tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
        tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)

        return tensor

    return [_np2Tensor(a) for a in args]

def eventnp2Tensor(*args):
    def _np2Tensor(img):
        img = img.astype('float64')
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # NHWC -> NCHW
        tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
        return tensor

    return [_np2Tensor(a) for a in args]


def data_augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = np.rot90(img)

        return img
    if len(args) == 1:
        return _augment(args[0])

    return [_augment(a) for a in args]


def postprocess(images, rgb_range, ycbcr_flag, device):
    def _postprocess(img, rgb_coefficient, ycbcr_flag, device):
        # if isinstance(img, list):
        #     img = img[0]
        if ycbcr_flag:
            out = img.mul(rgb_coefficient).clamp(16, 235)
        else:
            out = img.mul(rgb_coefficient).clamp(0, 255).round()

        return out

    rgb_coefficient = 255 / rgb_range

    return [_postprocess(img, rgb_coefficient, ycbcr_flag, device) for img in images]


def calc_psnr(img1, img2, rgb_range=1., shave=4):
    if isinstance(img1, torch.Tensor):
        img1 = img1[:, :, shave:-shave, shave:-shave]
        img1 = img1.to('cpu').numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2[:, :, shave:-shave, shave:-shave]
        img2 = img2.to('cpu').numpy()
    mse = np.mean((img1 / rgb_range - img2 / rgb_range) ** 2)
    # print(mse)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calc_grad_sobel(img, device='cuda'):
    if not isinstance(img, torch.Tensor):
        raise Exception("Now just support torch.Tensor. See the Type(img)={}".format(type(img)))
    if not img.ndimension() == 4:
        raise Exception("Tensor ndimension must equal to 4. See the img.ndimension={}".format(img.ndimension()))

    img = torch.mean(img, dim=1, keepdim=True)

    # img = calc_meanFilter(img, device=device)  # meanFilter

    sobel_filter_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).reshape((1, 1, 3, 3))
    sobel_filter_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).reshape((1, 1, 3, 3))
    sobel_filter_X = torch.from_numpy(sobel_filter_X).float().to(device)
    sobel_filter_Y = torch.from_numpy(sobel_filter_Y).float().to(device)
    grad_X = F.conv2d(img, sobel_filter_X, bias=None, stride=1, padding=1)
    grad_Y = F.conv2d(img, sobel_filter_Y, bias=None, stride=1, padding=1)
    grad = torch.sqrt(grad_X.pow(2) + grad_Y.pow(2))

    return grad_X, grad_Y, grad


def calc_meanFilter(img, kernel_size=11, n_channel=1, device='cuda'):
    mean_filter_X = np.ones(shape=(1, 1, kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    mean_filter_X = torch.from_numpy(mean_filter_X).float().to(device)
    new_img = torch.zeros_like(img)
    for i in range(n_channel):
        new_img[:, i:i + 1, :, :] = F.conv2d(img[:, i:i + 1, :, :], mean_filter_X, bias=None,
                                             stride=1, padding=kernel_size // 2)
    return new_img

def label_to_levels(label, num_classes, dtype=torch.float32):
    """Converts integer class label to extended binary label vector

    Parameters
    ----------
    label : int
        Class label to be converted into a extended
        binary vector. Should be smaller than num_classes-1.

    num_classes : int
        The number of class clabels in the dataset. Assumes
        class labels start at 0. Determines the size of the
        output vector.

    dtype : torch data type (default=torch.float32)
        Data type of the torch output vector for the
        extended binary labels.

    Returns
    ----------
    levels : torch.tensor, shape=(num_classes-1,)
        Extended binary label vector. Type is determined
        by the `dtype` parameter.

    Examples
    ----------
    >>> label_to_levels(0, num_classes=5)
    tensor([0., 0., 0., 0.])
    >>> label_to_levels(1, num_classes=5)
    tensor([1., 0., 0., 0.])
    >>> label_to_levels(3, num_classes=5)
    tensor([1., 1., 1., 0.])
    >>> label_to_levels(4, num_classes=5)
    tensor([1., 1., 1., 1.])
    """
    if not label <= num_classes-1:
        raise ValueError('Class label must be smaller or '
                         'equal to %d (num_classes-1). Got %d.'
                         % (num_classes-1, label))
    if isinstance(label, torch.Tensor):
        int_label = int(label.item())
    else:
        int_label = int(label)

    levels = [1]*int_label + [0]*(num_classes - 1 - int_label)
    levels = torch.tensor(levels, dtype=dtype)
    return levels


def levels_from_labelbatch(labels, num_classes, dtype=torch.float32):
    """
    Converts a list of integer class label to extended binary label vectors

    Parameters
    ----------
    labels : list or 1D orch.tensor, shape=(num_labels,)
        A list or 1D torch.tensor with integer class labels
        to be converted into extended binary label vectors.

    num_classes : int
        The number of class clabels in the dataset. Assumes
        class labels start at 0. Determines the size of the
        output vector.

    dtype : torch data type (default=torch.float32)
        Data type of the torch output vector for the
        extended binary labels.

    Returns
    ----------
    levels : torch.tensor, shape=(num_labels, num_classes-1)

    Examples
    ----------
    >>> levels_from_labelbatch(labels=[2, 1, 4], num_classes=5)
    tensor([[1., 1., 0., 0.],
            [1., 0., 0., 0.],
            [1., 1., 1., 1.]])
    """
    levels = []
    for label in labels:
        levels_from_label = label_to_levels(
            label=label, num_classes=num_classes, dtype=dtype)
        levels.append(levels_from_label)

    levels = torch.stack(levels)
    return levels


def proba_to_label(probas):
    """
    Converts predicted probabilities from extended binary format
    to integer class labels

    Parameters
    ----------
    probas : torch.tensor, shape(n_examples, n_labels)
        Torch tensor consisting of probabilities returned by CORAL model.

    Examples
    ----------
    >>> # 3 training examples, 6 classes
    >>> probas = torch.tensor([[0.934, 0.861, 0.323, 0.492, 0.295],
    ...                        [0.496, 0.485, 0.267, 0.124, 0.058],
    ...                        [0.985, 0.967, 0.920, 0.819, 0.506]])
    >>> proba_to_label(probas)
    tensor([2, 0, 5])
    """
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels
