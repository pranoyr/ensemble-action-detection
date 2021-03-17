import torch
import numpy as np

# bboxes = torch.Tensor([[1,2,3,4],
#                 [5,6,7,8],
#                 [9,8,7,6]])


# outputs = torch.Tensor([[0.1,0.1,0.1,0.06,0.7],
#                         [0.6,0.1,0.1,0.8,0.1],
#                         [0.1,0.5,0.5,0.1,0.1]])

# scores, indices = torch.topk(outputs, dim=1, k=2)
# print(indices)

# for i, preds in enumerate(indices):
#     mask = scores[i] > 0.5
#     results = preds[mask]
#     print(results)
#     print(bboxes[i])



#print(bboxes[mask, :])



# from torchvision.models import resnet

# model = resnet.resnet18(pretrained=True)
# print(model)
  

# model = torch.nn.Sequential(*(list(model.children())[:-2]))
# print(model)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

a =  AverageMeter('Acc@1', ':6.2f')
a.update(0.3, 30)
a.update(0.2, 30)
print(a.avg)

print(5 % 10)