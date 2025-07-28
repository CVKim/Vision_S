from visionsuite.engines.utils.bases.base_results import BaseResults, update_log
from visionsuite.engines.classification.utils.results.results import Results


class TrainResults(Results):
    __instance = None
    __initialized = False

    __learning_rate = -1

    @classmethod
    def get_attribute_names(cls):
        attribute_names = list(cls.__dict__.keys())
        return attribute_names

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls._log = {}

        return cls.__instance

    def __init__(self):
        if self.__initialized:
            return

        super().__init__()
        self.__initialized = True

    def __setattr__(self, name, value):
        allowed_attributes = (
            self.get_attribute_names()
            + Results.get_attribute_names()
            + BaseResults.get_attribute_names()
        )
        if name not in allowed_attributes:
            raise AttributeError(f"Attribute '{name}' is not allowed in this class")
        super().__setattr__(name, value)

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    @update_log
    def learning_rate(self, value):
        self.__learning_rate = value


if __name__ == "__main__":
    train_res1 = TrainResults()
    train_res2 = TrainResults()

    loss = 0.1
    cpu_usage = 8.5
    gpu_usage = 8.1
    iou = 0.1
    time_for_a_epoch = 2
    time_for_epochs = 2
    learning_rate = 0.1
    bg_iou = 0.1
    stabbed_iou = 0.05

    for epoch in range(1, 5):
        loss -= 0.02
        cpu_usage += 0.01
        gpu_usage += 0.02
        iou += 0.05
        time_for_a_epoch += 0.01
        time_for_epochs += 0.05
        learning_rate += 0.1
        bg_iou += 0.1
        stabbed_iou += 0.05

        train_res1.epoch = epoch
        train_res1.loss = loss
        train_res1.cpu_usage = cpu_usage
        train_res1.gpu_usage = gpu_usage
        train_res1.iou = iou
        train_res1.time_for_a_epoch = time_for_a_epoch
        train_res1.time_for_epochs = time_for_epochs
        train_res1.learning_rate = learning_rate
        train_res1.classes_iou = {"bg_iou": bg_iou, "stabbed_iou": stabbed_iou}

        print("1. ", train_res1.log)
        print("2. ", train_res2.log)