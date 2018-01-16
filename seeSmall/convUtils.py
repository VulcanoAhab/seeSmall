import numpy as np
import collections


class WorkingSet:
    """
    """
    _dataObj=collections.namedtuple("conv", ["label", "_features"])
    _dataset=[]

    @classmethod
    def add_sample(cls, label, features):
        """
        """
        cls._dataset.append(cls._dataObj(label, features))

    @classmethod
    def build_sets(cls, ratio=0.7):
        """
        """
        np.random.shuffle(cls._dataset)
        datasize=len(cls._dataset)
        n=int(datasize*ratio)
        cls._training_set=cls._dataset[:n]
        cls._eval_set=cls._dataset[n:]

    @classmethod
    def features(cls):
        """
        """
        return np.array([d._features for d in cls._dataset])

    @classmethod
    def labels(cls):
        """
        """
        return np.array([d.label for d in cls._dataset])

    @classmethod
    def training_data(cls):
        """
        """
        return np.array([d._features for d in cls._training_set])

     @classmethod
    def training_labels(cls):
        """
        """
        return np.array([d.label for d in cls._training_set])

    @classmethod
    def eval_data(cls):
        """
        """
        return np.array([d._features for d in cls._eval_set])

    @classmethod
    def eval_labels(cls):
        """
        """
        return np.array([d.label for d in cls._eval_set])

class Transforms:
    """
    """
    _height=28
    _width=28

    @classmethod
    def set_size(cls, height, width):
        """
        """
        cls._height=height
        cls._width=width
    
    @classmethod
    def set_color(cls, color=gray):
        """
        not implemented yet
        ---
        """
        cls._color=color

    @classmethod
    def run(cls, image):
        """
        """
        pass