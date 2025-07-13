# This Code is used to generate figure 4
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
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
col_size = 6
Sigma_save = np.zeros((col_size,col_size))
SNR_Test_save = np.zeros((col_size,col_size,row_size))
Gain_save = np.zeros((col_size,col_size,row_size))

Error = np.array(['1_4p', '1p', '5p', '10p', '25p', '100p'])
Error_char = np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
Error_rms = np.array([1/4, 1, 5, 10, 25, 100])
# np.random.seed(42)

for ii in range(0,col_size):
    for pp in range(0,col_size):
        x_true = np.zeros((n, 3))
        x_true[0] = [1.0, 1.0, 1.0]
        for i in range(1, n - 1):
            x_true[i] = lorenz_system(x_true[i - 1], dt, sigma=10)
        # Add noise to the observations
        x_true_rms1 = ((x_true[:, 0] ** 2).mean()) ** 0.5
        x_true_rms2 = ((x_true[:, 1] ** 2).mean()) ** 0.5
        x_true_rms3 = ((x_true[:, 2] ** 2).mean()) ** 0.5
        SNR = 100 / Error_rms[ii]
        noise_level = (1 / SNR) * np.array([x_true_rms1, x_true_rms2, x_true_rms3])
        Time_train = np.arange(0, int(n / 2), dtype=int) * dt
        Noise = np.array([np.random.normal(0, noise_level[0], int(n / 2)),
                          np.random.normal(0, noise_level[1], int(n / 2)),
                          np.random.normal(0, noise_level[2], int(n / 2))]).T
        Time_test = np.arange(int(n / 2) + 1, n, dtype=int) * dt
        X_test = x_true + np.random.normal(0, noise_level, (n, 3))
        for qq in range(0,row_size):
            model = torch.load('./Models/Model_Pruned_' + Error[pp] + Error_char[qq] + '.pth', weights_only=False)
            # Load the entire model
            y_pred_test = model.predict(X_test[:,0:2])
            mse_total = mean_squared_error(x_true, y_pred_test)
            rmse_indiv = np.array([mean_squared_error(x_true[:,0], y_pred_test[:,0])**0.5/x_true_rms1,
                                 mean_squared_error(x_true[:,1], y_pred_test[:,1])**0.5/x_true_rms2,
                                 mean_squared_error(x_true[:,2], y_pred_test[:,2])**0.5/x_true_rms3])
            SNR_Test = 1/rmse_indiv
            Gain = SNR_Test/SNR
            print(f"Tuned MSE: {mse_total:.6f}")
            print(f"SNR1: {SNR_Test[0]:.6f}, SNR2: {SNR_Test[1]:.6f}, SNR3: {SNR_Test[2]:.6f}")
            print(f"Gain1: {Gain[0]:.6f}, Gain2: {Gain[1]:.6f}, Gain3: {Gain[2]:.6f}")
            Gain_save[ii, pp, qq] = Gain.mean()
            SNR_Test_save[ii, pp, qq] = SNR_Test.mean()

matrix = np.average(Gain_save, axis=-1)
matrix = matrix[:, ::-1]
# Plot the matrix
# Create a custom colormap
reds = plt.cm.Reds_r(np.linspace(0.2, 0.99, 22))
greens = plt.cm.Greens(np.linspace(0, 1, 106))
newcolors = np.vstack((reds, greens))
newcmp = ListedColormap(newcolors)

# Plot the matrix with the custom colormap
plt.style.use('./Styles/AIPStyles.mplstyle')
plt.imshow(matrix, cmap=newcmp, interpolation='nearest', vmin=0, vmax=6)
plt.colorbar()
# plt.title('Denoising Gain Matrix')

# Add numbers to each cell
for (i, j), val in np.ndenumerate(matrix):
    plt.text(j, i, f'{val:.3f}', ha='center', va='center', color='black')

# Set custom ticks and labels for both axes
plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=[1, 4, 10, 20, 100, 400])
plt.yticks(ticks=[0, 1, 2, 3, 4, 5], labels=[400, 100, 20, 10, 4, 1])
plt.xlabel(r'$\mathrm{SNR_{Train}}$')
plt.ylabel(r'$\mathrm{SNR_{Test}}$')
plt.show()

