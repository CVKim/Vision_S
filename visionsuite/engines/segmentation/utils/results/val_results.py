from visionsuite.engines.utils.bases.base_results import BaseResults, update_log
from visionsuite.engines.segmentation.utils.results.results import Results


class ValResults(Results):
    __instance = None
    __initialized = False

    __best_accuracy = -1

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
    def best_loss(self):
        return self.__best_loss

    @best_loss.setter
    @update_log
    def best_loss(self, value):
        self.__best_loss = value

    @property
    def best_accuracy(self):
        return self.__best_accuracy

    @best_accuracy.setter
    @update_log
    def best_accuracy(self, value):
        self.__best_accuracy = value


if __name__ == "__main__":
    val_res1 = ValResults()
    val_res2 = ValResults()

    loss = 0.1
    cpu_usage = 8.5
    gpu_usage = 8.1
    iou = 0.1
    time_for_a_epoch = 2
    time_for_epochs = 2
    bg_iou = 0.1
    stabbed_iou = 0.05

    for epoch in range(1, 5):
        loss -= 0.02
        cpu_usage += 0.01
        gpu_usage += 0.02
        iou += 0.05
        time_for_a_epoch += 0.01
        time_for_epochs += 0.05
        bg_iou += 0.1
        stabbed_iou += 0.05

        val_res1.epoch = epoch
        val_res1.loss = loss
        val_res1.cpu_usage = cpu_usage
        val_res1.gpu_usage = gpu_usage
        val_res1.iou = iou
        val_res1.time_for_a_epoch = time_for_a_epoch
        val_res1.time_for_epochs = time_for_epochs
        val_res1.classes_iou = {"bg_iou": bg_iou, "stabbed_iou": stabbed_iou}

        print("1. ", val_res1.log)
        print("2. ", val_res2.log)