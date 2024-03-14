import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
T_ce = np.load('T/popane_ce.npy')
Y_ce = np.load('Y/popane_ce.npy')

T_ib = np.load('T/popane_IB.npy')
Y_ib = np.load('Y/popane_IB.npy')

T_hhoib = np.load('T/popane_HHOIB.npy')
Y_hhoib = np.load('Y/popane_HHOIB.npy')

T_ce = tsne.fit_transform(T_ce)
T_ib = tsne.fit_transform(T_ib)
T_hhoib = tsne.fit_transform(T_hhoib)
fig, axs = plt.subplots(1, 3, figsize=(24, 6), constrained_layout=True)
axs[0].scatter(T_ce[:, 0], T_ce[:, 1], c=Y_ce[:], cmap='tab10', marker='o', s=10)
axs[1].scatter(T_ib[:, 0], T_ib[:, 1], c=Y_ib[:], cmap='tab10', marker='o', s=10)
axs[2].scatter(T_hhoib[:, 0], T_hhoib[:, 1], c=Y_hhoib[:], cmap='tab10', marker='o', s=10)
axs[0].xaxis.set_visible(False)
axs[0].yaxis.set_visible(False)
axs[1].xaxis.set_visible(False)
axs[1].yaxis.set_visible(False)
axs[2].xaxis.set_visible(False)
axs[2].yaxis.set_visible(False)
axs[0].set_title('Cross Entropy', fontsize=30)
axs[1].set_title(r'IB,$\beta$=0.01', fontsize=30)
axs[2].set_title(r'HHO-IB,$\beta$=0.01', fontsize=30)
plt.savefig('cluster.pdf')
plt.show()
