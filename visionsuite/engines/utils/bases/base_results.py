from abc import ABCMeta
from visionsuite.engines.utils.helpers import print_class_name_on_instantiation

def update_log(func):
    def wrapper(self, value):
        func(self, value)
        log_entry = {func.__name__: value}
        self._update_log(log_entry)

    return wrapper

@print_class_name_on_instantiation
class BaseResults(metaclass=ABCMeta):
    _log = {}
    _best = {}
    _epoch = 0
    _loss = 999
    _save_loss = False
    _cpu_usage = -1
    _gpu_usage = -1

    @classmethod
    def __update_best(cls, value):
        assert isinstance(value, dict), ValueError(f"Argugments to update best must be dictionary, not{type(value)}")
        cls._best.update(value)

    @classmethod
    def _update_log(cls, value):
        assert isinstance(value, dict), ValueError(f"Argugments to update log must be dictionary, not{type(value)}")
        cls._log.update(value)

    @classmethod
    def get_attribute_names(cls):
        attribute_names = list(cls.__dict__.keys())
        return attribute_names

    def __init__(self):
        pass

    @property
    def log(self):
        return self._log

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    @update_log
    def epoch(self, value):
        assert value >= 0, ValueError(f"Epoch for results cannot be negative, now it is {value}")
        self._epoch = value

    @property
    def loss(self):
        return self._loss

    @loss.setter
    @update_log
    def loss(self, value):
        if value < self._loss:
            self._save_loss = True
        else:
            self._save_loss = False

        self._loss = value

    @property
    def save_loss(self):
        return self._save_loss

    @property
    def cpu_usage(self):
        return self._cpu_usage

    @cpu_usage.setter
    @update_log
    def cpu_usage(self, value):
        self._cpu_usage = value

    @property
    def gpu_usage(self):
        return self._gpu_usage

    @gpu_usage.setter
    @update_log
    def gpu_usage(self, value):
        self._gpu_usage = value

    @property
    def best(self):
        return self._best

    @best.setter
    def best(self, value1, value2=None):
        if value2 is None:
            assert isinstance(value1, dict), ValueError(f"Argument for best must be dictionary, not {type(value1)}")
            value = value1
        else:
            assert isinstance(value1, str) and isinstance(value2, (int, float)), \
                ValueError(f"precision value must be int or float, not {type(value2)}")
            value = {value1: value2}

        self.__update_best(value)
        self._update_log(value)

    def update_best(self, key, val):
        if key not in self.best.keys():
            self.best = {key: val, f'is_{key}_best': True}
        else:
            if getattr(self, key) < val:
                self.best = {key: val, f'is_{key}_best': True}
            else:
                self.best = {f'is_{key}_best': False}