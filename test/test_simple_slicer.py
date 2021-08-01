import pytest
import numpy as np
from collections import OrderedDict

from labeled_mask import SimpleSlicer

def test_get_normal():
    arr = np.arange(10)
    l = OrderedDict(
        tag1=3, 
        tag2=5, 
        tag3=2
    )
    slicer = SimpleSlicer(len_dict=l)
    assert all(slicer.get(arr, 'tag1') == [0,1,2])
    assert all(slicer.get(arr, 'tag2') == [3,4,5,6,7])
    assert all(slicer.get(arr, 'tag3') == [8,9])

def test_get_with_last():
    arr = np.arange(10)
    l = OrderedDict(
        tag1=1, 
        tag2=-1
    )
    slicer = SimpleSlicer(len_dict=l)
    assert all(slicer.get(arr, 'tag1') == [0])
    assert all(slicer.get(arr, 'tag2') == [1,2,3,4,5,6,7,8,9])
