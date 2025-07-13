# This Code is used to generate figure 2
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
from Optimizer import Optimizer
from ReservoirComputer import ReservoirComputer

# Lorenz attractor parameters
rho = 28.0
beta = 8.0 / 3.0
dt = 0.005
n = 10000

def lorenz_system(state, dt, sigma):
    x, y, z = state
    dx = sigma * (y - x) * dt
    dy = (x * (rho - z) - y) * dt
    dz = (x * y - beta * z) * dt
    return np.array([x + dx, y + dy, z + dz])

row_size = 10
col_size0 = 6
col_size = 3
Sigma_save = np.zeros((col_size,col_size))
SNR_Test_save = np.zeros((col_size0,col_size,row_size))
Gain_save = np.zeros((col_size0,col_size,row_size))
Nrmlz_mse = np.zeros((col_size0,col_size,row_size))

Error = np.array(['1_4p','1p','5p','10p','25p','100p'])
Error_char = np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
Error_rms = np.array(['Initial_','Tuned_','Pruned_'])
Error_rms1 = np.array([0.25, 1, 5, 10, 25, 100])
# np.random.seed(42)

for ii in range(0,col_size0):
    for pp in range(0,col_size):
        x_true = np.zeros((n, 3))
        x_true[0] = [1.0, 1.0, 1.0]
        for i in range(1, n - 1):
            x_true[i] = lorenz_system(x_true[i - 1], dt, sigma=10)
        # Add noise to the observations
        x_true_rms1 = ((x_true[:, 0] ** 2).mean()) ** 0.5
        x_true_rms2 = ((x_true[:, 1] ** 2).mean()) ** 0.5
        x_true_rms3 = ((x_true[:, 2] ** 2).mean()) ** 0.5
        SNR = 100 / Error_rms1[ii]
        noise_level = (1 / SNR) * np.array([x_true_rms1, x_true_rms2, x_true_rms3])
        Time_train = np.arange(0, int(n / 2), dtype=int) * dt
        Noise = np.array([np.random.normal(0, noise_level[0], int(n / 2)),
                          np.random.normal(0, noise_level[1], int(n / 2)),
                          np.random.normal(0, noise_level[2], int(n / 2))]).T
        Time_test = np.arange(int(n / 2) + 1, n, dtype=int) * dt
        X_test = x_true + np.random.normal(0, noise_level, (n, 3))
        for qq in range(0,row_size):
            model = torch.load('./Models/Model_' + Error_rms[pp] + Error[ii] + Error_char[qq] + '.pth', weights_only=False)
            # Load the entire model
            y_pred_test = model.predict(X_test[:,0:2])
            mse_total = mean_squared_error(x_true, y_pred_test) / mean_squared_error(x_true, np.zeros_like(y_pred_test))
            Nrmlz_mse[ii, pp, qq] = mse_total

matrix2 = np.average(Nrmlz_mse, axis=-1).T
Nrmlzd_mse = matrix2[:, ::-1]
group_1 = Nrmlzd_mse[0,:].tolist()
group_2 = Nrmlzd_mse[1,:].tolist()
group_3 = Nrmlzd_mse[2,:].tolist()
group_4 = [0.248953, 0.007365, 0.000217, 0.000178, 0.000172, 0.000169]# EKF Imported

# Sample data
categories = ['1','4','10','20','100','400']
num_categories = len(categories)
indices = np.arange(num_categories)
bar_width = 0.1
print(indices)
print(group_1)

# Plot the matrix with the custom colormap
plt.figure(figsize=(5,4))
plt.style.use('./Styles/AIPStyles.mplstyle')
plt.bar(indices, group_1, bar_width, label='Trained RC', color='b')
plt.bar(indices + bar_width, group_2, bar_width, label='Tuned RC', color='g')
plt.bar(indices + 2 * bar_width, group_3, bar_width, label='Truncated RC', color='r')
plt.bar(indices + 3 * bar_width, group_4, bar_width, label='EKF', color='c')
plt.ylabel(r"$\mathrm{ln(NMSE)}$", labelpad=-2)
plt.xlabel(r"$\mathrm{SNR_{Test}}$", labelpad=-6)
plt.yscale('log')
plt.ylim([0.0001,10])
plt.xticks(indices + bar_width, categories, rotation=90)
plt.legend(loc='upper left', ncol=2)
plt.show()
