import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
from Optimizer import Optimizer
from ReservoirComputer import ReservoirComputer

# Read generated data corresponding to Lorenz Attractor
x_true = np.loadtxt('.\Datasets\Lorenz_Dataset25p\X_true.txt')
X_train = np.loadtxt('.\Datasets\Lorenz_Dataset25p\X_train.txt')
y_train = np.loadtxt('.\Datasets\Lorenz_Dataset25p\y_train.txt')
Time_train = np.loadtxt('.\Datasets\Lorenz_Dataset25p\Time_train.txt')
X_test = np.loadtxt('.\Datasets\Lorenz_Dataset25p\X_test.txt')
y_test = np.loadtxt('.\Datasets\Lorenz_Dataset25p\y_test.txt')
Time_test = np.loadtxt('.\Datasets\Lorenz_Dataset25p\Time_test.txt')
Time_Total = np.concatenate((Time_train, Time_test), axis=0)
X_Total = np.concatenate((X_train[:, 0:2], X_test[:, 0:2]))

model = torch.load('./Models/Model_Pruned_25p1.pth', weights_only=False)
y_pred_test = model.predict(X_Total)
mse_total = mean_squared_error(x_true[0:9999,:], y_pred_test[0:9999,:])

X_Total = np.concatenate((X_train, X_test))
x_Total_plot = np.concatenate((X_Total[:, 0:2], np.zeros((9999,1))),axis=1)

## Noisy data figure
plt.figure(figsize=(12, 4))
plt.style.use('.\Styles\AIPStyles.mplstyle')
plt.subplot(2,1,1)
plt.plot(Time_Total[:], X_Total[:, 0], color='orange', markersize=1, label='True State')
plt.xlim([0,50])# Full line for true state
plt.ylabel(r"${x}(t)$", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.subplot(2,1,2)
plt.plot(Time_Total[:], X_Total[:, 1], color='orange', markersize=1, label='True State')  # Full line for true state
plt.xlim([0,50])# Full line for true state
plt.xlabel('Time (s)')
plt.ylabel(r"${y}(t)$", fontsize=14)
plt.xlabel(r"$\it{t} (s)$", fontsize=14, labelpad=-5)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()

## Clean reconstructed data figure
plt.figure(figsize=(12, 6))
plt.style.use('.\Styles\AIPStyles.mplstyle')
plt.subplot(3,1,1)
plt.plot(Time_Total[:], y_pred_test[:, 0], color='blue', markersize=1, label='True State')
plt.ylim([-20,20])
plt.xlim([0,50])# Full line for true state
plt.ylabel(r"$\it{x}(t)$", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.subplot(3,1,2)
plt.plot(Time_Total[:], y_pred_test[:, 1], color='blue', markersize=1, label='True State')  # Full line for true state
plt.xlim([0,50])# Full line for true state
plt.ylabel(r"$\it{y}(t)$", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.subplot(3,1,3)
plt.plot(Time_Total[:], y_pred_test[:, 2], color='blue', markersize=1, label='True State')  # Full line for true state
plt.xlim([0,50])# Full line for true state
plt.ylim([0,50])
plt.ylabel(r"$\it{z}(t)$", fontsize=14)
plt.xlabel(r"$\it{t} (s)$", fontsize=14, labelpad=-5)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()

# Plot the results
fig = plt.figure(figsize=(6, 6))
plt.style.use('./Styles/AIPStyles.mplstyle')
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
ax2 = fig.add_subplot(111, projection='3d')
ax2.plot(*x_Total_plot[100::2].T, color='orange', lw=1.0, label='Incomplete & Noisy')
ax2.plot(*y_pred_test[100:,:].T, color='blue', lw=1.0, label='Complete & Denoised')
ax2.set_xlabel(r"$x(t)$", labelpad=0, fontsize=12)
ax2.set_ylabel(r"$y(t)$", labelpad=0, fontsize=12)
ax2.set_zlabel(r"$z(t)$", labelpad=-2, fontsize=12)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.legend()
plt.show()
