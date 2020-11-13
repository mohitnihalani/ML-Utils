import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
class AccuracyMeter(object):
    """Computes and stores the average and current topk accuracy"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.acc = 0
        self.avg = 0

    def update(self, acc, n=1):
        self.count += n
        self.acc += acc
        self.avg = self.acc/self.count