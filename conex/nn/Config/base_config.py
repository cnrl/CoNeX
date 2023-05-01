from typing import Tuple, Union, Callable
from pymonntorch import *

import collections.abc


class BaseConfig:
    def make(self):
        return {}

    @staticmethod
    def deep_update(main_dict, update_dict):
        for k, v in update_dict.items():
            if isinstance(v, collections.abc.Mapping):
                main_dict[k] = BaseConfig.deep_update(main_dict.get(k, {}), v)
            else:
                main_dict[k] = v
        return main_dict

    def update(self, update_dict):
        for key, value in update_dict.items():
            if isinstance(getattr(self, key), collections.abc.Mapping):
                setattr(self, key, BaseConfig.deep_update(getattr(self, key), value))
            else:
                setattr(self, key, value)

    def update_make(self, **kwargs):
        self.update(kwargs)
        return self.make()

    def __call__(self, *args, **kwargs):
        self.make(*args, **kwargs)
