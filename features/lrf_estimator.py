from abc import ABC, abstractmethod

class LRFEstimator(ABC):
    """
    Abstract LRF Estimator class
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, cloud, indices):
        pass

    def __repr__(self):
        return self.__class__.__name__

