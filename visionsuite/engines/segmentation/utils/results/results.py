from visionsuite.engines.utils.bases.base_results import BaseResults, update_log


class Results(BaseResults):
    __accuracy = -1
    __time_for_a_epoch = -1
    __time_for_epochs = -1
    __classes_accuracy = {}

    @classmethod
    def __update_classes_accuracy(cls, value):
        assert isinstance(value, dict), ValueError(
            f"Argugments to update log must be dictionary, not{type(value)}"
        )
        cls.__classes_accuracy.update(value)

    @classmethod
    def get_attribute_names(cls):
        attribute_names = list(cls.__dict__.keys())
        return attribute_names

    def __init__(self, classes=None, logger=None):
        super().__init__()
        if classes is not None:
            for class_ in classes:
                self.__classes_accuracy[class_] = -1

            if logger is not None:
                logger.info(
                    f"The accuracy for {classes} has been set",
                    self.__ini__.__name__,
                    __class__.__name__,
                )

    @property
    def accuracy(self):
        return self.__accuracy

    @accuracy.setter
    @update_log
    def accuracy(self, value):
        assert value >= 0, ValueError(f"Train accuracy cannot be lower than 0")
        self.__accuracy = value

    @property
    def time_for_a_epoch(self):
        return self.__time_for_a_epoch

    @time_for_a_epoch.setter
    @update_log
    def time_for_a_epoch(self, value):
        self.__time_for_a_epoch = value

    @property
    def time_for_epochs(self):
        return self.__time_for_epochs

    @time_for_epochs.setter
    @update_log
    def time_for_epochs(self, value):
        self.__time_for_epochs = value

    @property
    def classes_accuracy(self):
        return self.__classes_accuracy

    @classes_accuracy.setter
    def classes_accuracy(self, value1, value2=None):
        if value2 is None:
            assert isinstance(value1, dict), ValueError(
                f"Argument for classes_accuracy must be dictionary, not {type(value1)}"
            )
            value = value1
        else:
            assert isinstance(value1, str) and isinstance(
                value2, (int, float)
            ), ValueError(f"Accuracy value must be int or float, not {type(value2)}")
            value = {value1: value2}

        self.__update_classes_accuracy(value)
        self._update_log(value)


