import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
x = np.arange(0, 201, 1)
#################################################################################
IXT_train_0001=np.load('0.001/IXT_train/popane_HHOIB_0.001_200.npy')
IYT_train_0001=np.load('0.001/IYT_train/popane_HHOIB_0.001_200.npy')

IXT_test_0001=np.load('0.001/IXT_test/popane_HHOIB_0.001_200.npy')
smoothed_IXT_test_0001 = savgol_filter(IXT_test_0001, window_length=15, polyorder=3)
IYT_test_0001=np.load('0.001/IYT_test/popane_HHOIB_0.001_200.npy')
smoothed_IYT_test_0001 = savgol_filter(IYT_test_0001, window_length=15, polyorder=3)
#################################################################################
IXT_train_001=np.load('0.01/IXT_train/popane_HHOIB_0.01_200.npy')
IYT_train_001=np.load('0.01/IYT_train/popane_HHOIB_0.01_200.npy')

IXT_test_001=np.load('0.01/IXT_test/popane_HHOIB_0.01_200.npy')
smoothed_IXT_test_001 = savgol_filter(IXT_test_001, window_length=15, polyorder=3)
IYT_test_001=np.load('0.01/IYT_test/popane_HHOIB_0.01_200.npy')
smoothed_IYT_test_001 = savgol_filter(IYT_test_001, window_length=15, polyorder=3)
#################################################################################
IXT_train_0015=np.load('0.015/IXT_train/popane_HHOIB_0.015_200.npy')
IYT_train_0015=np.load('0.015/IYT_train/popane_HHOIB_0.015_200.npy')

IXT_test_0015=np.load('0.015/IXT_test/popane_HHOIB_0.015_200.npy')
smoothed_IXT_test_0015 = savgol_filter(IXT_test_0015, window_length=15, polyorder=3)
IYT_test_0015=np.load('0.015/IYT_test/popane_HHOIB_0.015_200.npy')
smoothed_IYT_test_0015 = savgol_filter(IYT_test_0015, window_length=15, polyorder=3)
#################################################################################
IXT_train_01=np.load('0.1/IXT_train/popane_HHOIB_0.1_200.npy')
IYT_train_01=np.load('0.1/IYT_train/popane_HHOIB_0.1_200.npy')

IXT_test_01=np.load('0.1/IXT_test/popane_HHOIB_0.1_200.npy')
smoothed_IXT_test_01 = savgol_filter(IXT_test_01, window_length=15, polyorder=3)
IYT_test_01=np.load('0.1/IYT_test/popane_HHOIB_0.1_200.npy')
smoothed_IYT_test_01 = savgol_filter(IYT_test_01, window_length=15, polyorder=3)

fig, axs = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)

# 绘制上面4张子图
axs[0, 0].plot(x, IXT_train_0001,color='#1E90FF',label='I(X;T)')
axs[0, 0].plot(x, IYT_train_0001,color='#FFA500',label='I(Y;T)')
axs[0, 0].set_title(r'$\beta = 0.001$')
axs[0, 0].set_ylabel('Train', fontsize=30)
axs[0, 0].legend()

axs[0, 1].plot(x, IXT_train_001,color='#1E90FF',label='I(X;T)')
axs[0, 1].plot(x, IYT_train_001,color='#FFA500',label='I(Y;T)')
axs[0, 1].set_title(r'$\beta = 0.01$')
axs[0, 1].legend()

axs[0, 2].plot(x, IXT_train_0015,color='#1E90FF',label='I(X;T)')
axs[0, 2].plot(x, IYT_train_0015,color='#FFA500',label='I(Y;T)')
axs[0, 2].set_title(r'$\beta = 0.015$')
axs[0, 2].legend()

axs[0, 3].plot(x, IXT_train_01,color='#1E90FF',label='I(X;T)')
axs[0, 3].plot(x, IYT_train_01,color='#FFA500',label='I(Y;T)')
axs[0, 3].set_title(r'$\beta = 0.1$')
axs[0, 3].legend()
# 绘制下面4张子图
axs[1, 0].plot(x, IXT_test_0001,color='#87CEFA')
axs[1, 0].plot(x, smoothed_IXT_test_0001,color='#1E90FF',label='I(X;T)')
axs[1, 0].plot(x, IYT_test_0001,color='#FFE4B5')
axs[1, 0].plot(x, smoothed_IYT_test_0001,color='#FFA500',label='I(Y;T)')
axs[1, 0].set_ylabel('Validation', fontsize=30)
axs[1, 0].legend()

axs[1, 1].plot(x, IXT_test_001,color='#87CEFA')
axs[1, 1].plot(x, smoothed_IXT_test_001,color='#1E90FF',label='I(X;T)')
axs[1, 1].plot(x, IYT_test_001,color='#FFE4B5')
axs[1, 1].plot(x, smoothed_IYT_test_001,color='#FFA500',label='I(Y;T)')
axs[1, 1].legend()

axs[1, 2].plot(x, IXT_test_0015,color='#87CEFA')
axs[1, 2].plot(x, smoothed_IXT_test_0015,color='#1E90FF',label='I(X;T)')
axs[1, 2].plot(x, IYT_test_0015,color='#FFE4B5')
axs[1, 2].plot(x, smoothed_IYT_test_0015,color='#FFA500',label='I(Y;T)')
axs[1, 2].legend()

axs[1, 3].plot(x, IXT_test_01,color='#87CEFA')
axs[1, 3].plot(x, smoothed_IXT_test_01,color='#1E90FF',label='I(X;T)')
axs[1, 3].plot(x, IYT_test_01,color='#FFE4B5')
axs[1, 3].plot(x, smoothed_IYT_test_01,color='#FFA500',label='I(Y;T)')
axs[1, 3].legend()
# 调整子图之间的间距

plt.tight_layout()
plt.savefig('MI.pdf')
plt.show()
