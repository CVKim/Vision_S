import torch

from visionsuite.engines.utils.torch_utils.dist import reduce_across_processes

class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None
        self.values = {}

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.inference_mode():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        
        self.update_values(acc_global, acc, iu)
        
        return acc_global, acc, iu

    def update_values(self, acc_global, acc, iu):
        self.values.update({"acc_global": acc_global.item()*100})
        self.values.update({ f"{idx}_acc": _acc for idx, _acc in enumerate((acc * 100).tolist())})
        self.values.update({ f"{idx}_iou": _iou for idx, _iou in enumerate((iu * 100).tolist())})
        self.values.update({"mean IoU": iu.mean().item() * 100})


    def reduce_from_all_processes(self):
        self.mat = reduce_across_processes(self.mat).to(torch.int64)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return ("global correct: {:.1f}\naverage row correct: {}\nIoU: {}\nmean IoU: {:.1f}").format(
            acc_global.item() * 100,
            [f"{i:.1f}" for i in (acc * 100).tolist()],
            [f"{i:.1f}" for i in (iu * 100).tolist()],
            iu.mean().item() * 100,
        )