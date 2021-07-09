import numpy as np
from tsne_torch import TorchTSNE as TSNE
import matplotlib.pyplot as plt



X = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
# X_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(X)
X_emb = TSNE(n_components=2).fit_transform(X)
print(X_emb)



tx = X_emb[:, 0]
ty = X_emb[:, 1]


# initialize a matplotlib plot
fig = plt.figure()
ax = fig.add_subplot(111)

# for every class, we'll add a scatter plot separately
colors_per_class = {'Mask':(255,0,0), 'UnMask':(0,255,0)}
labels = ["Mask"] * 4

for label in colors_per_class:
    # find the samples of the current class in the data
    indices = [i for i, l in enumerate(labels) if l == label]

    # extract the coordinates of the points of this class only
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)

    # convert the class color to matplotlib format
    color = np.array(colors_per_class[label], dtype=np.float) / 255

    # add a scatter plot with the corresponding color and label
    ax.scatter(current_tx, current_ty, c=color, label=label)

# build a legend using the labels we set previously
ax.legend(loc='best')

# finally, show the plot
plt.show()
# plt.savefig("output.jpg")
       
