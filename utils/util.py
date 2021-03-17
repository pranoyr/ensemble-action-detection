import csv
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


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


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))
    return value


class Metric:
    """ Computer precision/recall for multilabel classifcation
    """

    def __init__(self, num_classes):
        # For each class
        self.precision = dict()
        self.recall = dict()
        self.average_precision = dict()
        self.gt = []
        self.y = []
        self.num_classes = num_classes
    
    def update(self, outputs, targets):
        self.y.append(outputs.detach().cpu())
        self.gt.append(targets.detach().cpu())

    def compute_metrics(self):
        preds = torch.cat(self.y)
        targets = torch.cat(self.gt)
        preds = preds.numpy()
        targets = targets.numpy()

        # for i in range(self.num_classes):
        #     self.precision[i], self.recall[i], _ = precision_recall_curve(
        #         targets[:, i], preds[:, i])
        #     self.average_precision[i] = average_precision_score(
        #         targets[:, i], preds[:, i])

        # A "micro-average": quantifying score on all classes jointly
        # self.precision["micro"], self.recall["micro"], _ = precision_recall_curve(
        #     targets.ravel(), preds.ravel())
        self.average_precision["micro"] = average_precision_score(targets, preds,
                                                                  average="micro")
        return self.average_precision["micro"]


# def calculate_accuracy(outputs, targets):
#     batch_size = targets.size(0)

#     # _, pred = outputs.topk(1, 1, True)
#     _ , preds = torch.topk(outputs, dim=1 ,k=3)
#     preds = preds.t()


#     return n_correct_elems / batch_size

# if (__name__ == '__main__'):
#     metric = Metric()
#     outputs = torch.Tensor([[1, 1], [1, 0], [1, 1], [1, 0], [1, 1]])
#     targets = torch.Tensor([[1, 1], [1, 1], [1, 1], [1, 0], [1, 0]])

#     metric.update(outputs, targets)
#     metric.update(outputs, targets)
#     acc = metric.compute_metrics()
#     print(acc)


# output = F.sigmoid(output)
# scores , preds = torch.topk(output, dim=1 ,k=2)
# preds = preds.type(torch.FloatTensor)

# total = preds.size(0)*preds.size(1)
# correct = torch.eq(preds, targets_int_encoded).sum().item()
# acc = (correct/total) * 100
