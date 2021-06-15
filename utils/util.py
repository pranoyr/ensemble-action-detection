import csv
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


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

		self.precision["micro"], self.recall["micro"], _ = precision_recall_curve(
			targets.ravel(), preds.ravel())

		self.average_precision["micro"] = average_precision_score(targets, preds,
																  average="micro")
		precision = self.precision["micro"]
		recall = self.recall["micro"]
		avg = self.average_precision["micro"]
		print(precision)
		print(recall)
		print(avg)

		return avg


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
			correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


class ProgressMeter(object):
	def __init__(self, num_batches, meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def display(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		print('\t'.join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'