import torch
import numpy as np
import torch.nn.init as init


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def init_logfile(filename, text):
    f = open(filename, 'w')
    f.write(text + "\n")
    f.close()


def log(filename, text):
    f = open(filename, 'a')
    f.write(text + "\n")
    f.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)
