# This Code is used to generate figure 4
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
col_size = 39
Sigma_save = np.zeros((col_size0,col_size))
SNR_Test_save = np.zeros((col_size0,col_size,row_size))
Gain_save = np.zeros((col_size0,col_size,row_size))
Nrmlz_mse = np.zeros((col_size0,col_size,row_size))

Error_char = np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
Error_rms = np.array(['Initial_robust','Tuned_robust','Pruned_robust'])
Error_rms1 = np.array([0.25, 1, 5, 10, 25, 100])

for ii in range(0,col_size0):
    for pp in range(0,col_size):
        sigma = 0.5+pp*0.5
        Sigma_save[ii,pp] = sigma
        x_true = np.zeros((n, 3))
        x_true[0] = [1.0, 1.0, 1.0]
        for i in range(1, n - 1):
            x_true[i] = lorenz_system(x_true[i - 1], dt, sigma=sigma)
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
            model = torch.load('./Models/Model_Tuned_25p_robust' + Error_char[qq] + '.pth', weights_only=False)
            # Load the entire model
            y_pred_test = model.predict(X_test[:,0:2])
            mse_total = mean_squared_error(x_true, y_pred_test) / mean_squared_error(x_true, np.zeros_like(y_pred_test))
            Nrmlz_mse[ii, pp, qq] = mse_total
            rmse_indiv = np.array([mean_squared_error(x_true[:,0], y_pred_test[:,0])**0.5/x_true_rms1,
                                 mean_squared_error(x_true[:,1], y_pred_test[:,1])**0.5/x_true_rms2,
                                 mean_squared_error(x_true[:,2], y_pred_test[:,2])**0.5/x_true_rms3])
            SNR_Test = 1/rmse_indiv
            Gain = SNR_Test/SNR
            Gain_save[ii, pp, qq] = Gain.mean()
            print(f"Tuned MSE: {mse_total:.6f}")
            print(f"SNR1: {SNR_Test[0]:.6f}, SNR2: {SNR_Test[1]:.6f}, SNR3: {SNR_Test[2]:.6f}")
            print(f"Gain1: {Gain[0]:.6f}, Gain2: {Gain[1]:.6f}, Gain3: {Gain[2]:.6f}")

matrix2 = np.average(Gain_save, axis=-1).T
plt.figure(figsize=(5,4))
plt.style.use('./Styles/AIPStyles.mplstyle')
plt.plot(Sigma_save[0,:], matrix2[:,5],'--g',label='SNR = 1')
plt.plot(Sigma_save[0,:], matrix2[:,4],'--b',label='SNR = 4')
plt.plot(Sigma_save[0,:], matrix2[:,3],'--r',label='SNR = 10')
plt.plot(Sigma_save[0,:], matrix2[:,2],'g',label='SNR = 20')
plt.plot(Sigma_save[0,:], matrix2[:,1],'b',label='SNR = 100')
plt.plot(Sigma_save[0,:], matrix2[:,0],'r',label='SNR = 400')
plt.xlabel(r"$\sigma$")
plt.ylabel(r"Denoising Gain")
plt.legend()
# plt.title(r"RC Trained on $\sigma = 8 & 10 $ and SNR = 4")
plt.legend(loc='upper left', ncol=3)
plt.xlim([0,20])
plt.ylim([0,6])
plt.show()
