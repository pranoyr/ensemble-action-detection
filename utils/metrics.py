from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import numpy as np


# For each class
precision = dict()
recall = dict()
average_precision = dict()

labels = [0]

# [[0,2,4],[2,4]]
y_score = np.array([[1,1],[1,0],[1,1],[1,0],[1,1]])

Y_test = np.array([[1,1],[1,1],[1,1],[1,0],[1,0]])


for i in range(2):
    
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

print(average_precision)

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))


# import numpy as np
# from sklearn.metrics import precision_recall_fscore_support
# y_pred = np.array([[1,1],[1,0],[1,1],[1,0],[1,1]])
# y_true = np.array([[1,0],[1,1],[1,1],[1,0],[1,0]])

# a = precision_recall_fscore_support(y_true, y_pred, average='micro')
# print(a)