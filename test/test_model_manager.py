import pytest
import numpy as np

from labeled_mask import MaskManager

arr = np.arange(25).reshape([5,5]).astype(np.float)
fill_value = -1
mask1 = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]
mask1_truth = [
    [-1, -1, -1, -1, -1],
    [-1,  6,  7,  8, -1],
    [-1, 11, 12, 13, -1],
    [-1, 16, 17, 18, -1],
    [-1, -1, -1, -1, -1]
]
mask2 = np.ones_like(arr)
mask2[2:, 2:] = 0
mask2_truth = [
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, 12, 13, 14],
    [-1, -1, 17, 18, 19],
    [-1, -1, 22, 23, 24]
]
mask3 = np.ones_like(arr)
mask3 -= np.identity(len(arr))
mask3_truth = [
    [ 0, -1, -1, -1, -1],
    [-1,  6, -1, -1, -1],
    [-1, -1, 12, -1, -1],
    [-1, -1, -1, 18, -1],
    [-1, -1, -1, -1, 24]
]

def test_manager():
    mask_dict = {
        'tag1': mask1,
        'tag2': mask2
    }
    mm = MaskManager(mask_dict=mask_dict, fill_value=fill_value)
    mm.add_mask('tag3', mask3)
    masked1 = mm.mask(target=arr, mask_name='tag1')
    masked2 = mm.mask(target=arr, mask_name='tag2')
    masked3 = mm.mask(target=arr, mask_name='tag3')

    assert np.all(masked1.filled() == mask1_truth)
    assert np.all(masked2.filled() == mask2_truth)
    assert np.all(masked3.filled() == mask3_truth)

def test_get_all():
    mask_dict = {
        'tag1': mask1,
        'tag2': mask2,
        'tag3': mask3
    }
    mm = MaskManager(mask_dict=mask_dict, fill_value=fill_value)
    masked = mm.get_all(arr)
    assert np.all(masked['tag1'] == mask1_truth)
    assert np.all(masked['tag2'] == mask2_truth)
    assert np.all(masked['tag3'] == mask3_truth)