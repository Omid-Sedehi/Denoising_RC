import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Lorenz attractor parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
dt = 0.005

# Lorenz attractor dynamics
def lorenz_system(state, dt):
    x, y, z = state
    dx = sigma * (y - x) * dt
    dy = (x * (rho - z) - y) * dt
    dz = (x * y - beta * z) * dt
    return np.array([x + dx, y + dy, z + dz])

# Extended Kalman Filter implementation
def lorenz_predict(state):
    return lorenz_system(state, dt)

# Read generated data corresponding to Lorenz Attractor
x_true = np.loadtxt('.\Datasets\Lorenz_Dataset5p\X_true.txt')
X_train = np.loadtxt('.\Datasets\Lorenz_Dataset5p\X_train.txt')
y_train = np.loadtxt('.\Datasets\Lorenz_Dataset5p\y_train.txt')
Time_train = np.loadtxt('.\Datasets\Lorenz_Dataset5p\Time_train.txt')
X_test = np.loadtxt('.\Datasets\Lorenz_Dataset5p\X_test.txt')
y_test = np.loadtxt('.\Datasets\Lorenz_Dataset5p\y_test.txt')
Time_test = np.loadtxt('.\Datasets\Lorenz_Dataset5p\Time_test.txt')
Time_Total = np.concatenate((Time_train, Time_test), axis=0)

SNR = 20
MS1 = np.square(x_true[:,0]).mean()
MS2 = np.square(x_true[:,1]).mean()
noise_level = SNR*np.array([[MS1,0],[0,MS2]])

n = 9999
# Initialize state and covariance matrices
x_est = np.zeros((n, 3))
P = np.eye(3)
Q = (10**-14) * np.eye(3)
R = noise_level * np.eye(2)
x_noisy = np.concatenate((X_train[:, 0:2], X_test[:, 0:2]))
x_est[0,:] = [1,1,1]

for i in range(1, n-1):
    # Prediction step
    x_pred = lorenz_predict(x_est[i - 1])
    F = np.array([[-sigma, sigma, 0],
                  [rho - x_est[i - 1, 2], -1, -x_est[i - 1, 0]],
                  [x_est[i - 1, 1], x_est[i - 1, 0], -beta]]) * dt + np.eye(3)
    P = F @ P @ F.T + Q
    # Update step
    y_tilde = x_noisy[i,:] - x_pred[0:2,]
    H = np.eye(2,3)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    x_est[i] = x_pred + K @ y_tilde
    P = (np.eye(3) - K @ H) @ P



X_Total = x_noisy
y_pred_Total = x_est
mse_EKF = mean_squared_error(x_true[0:9999,], y_pred_Total[0:9999,]) / mean_squared_error(y_pred_Total, np.zeros_like(y_pred_Total))
print(f"EKF MSE: {mse_EKF:.6f}")

# Plot the results
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(Time_Total, X_Total[:, 0], '-r', label='Noisy')
plt.plot(Time_Total, y_pred_Total[:, 0], '-b', label='Estimated')
plt.plot(Time_Total, x_true[0:9999, 0], '--k', label='Ground Truth')
plt.ylabel('x(t)')
plt.xlim([0,50])
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(Time_Total, X_Total[:, 1], '-r', label='Noisy')
plt.plot(Time_Total, y_pred_Total[:, 1], '-b', label='Estimated')
plt.plot(Time_Total, x_true[0:9999, 1], '--k', label='Ground Truth')
plt.ylabel('y(t)')
plt.xlim([0,50])
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(Time_Total, y_pred_Total[:, 2], '-b', label='Estimated')
plt.plot(Time_Total, x_true[0:9999, 2], '--k', label='Ground Truth')
plt.ylabel('z(t)')
plt.xlim([0,50])
plt.xlabel('Time (s)')
plt.legend()
plt.show()


# Plot the results
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_true[:, 0], x_true[:, 1], x_true[:, 2], '-b', markersize=1, label='True State')  # Full line for true state
# ax.plot(X_Total[:, 0], X_Total[:, 1], X_Total[:, 2], '--k', markersize=1, label='Noisy Observations')
ax.plot(y_pred_Total[:, 0], y_pred_Total[:, 1], y_pred_Total[:, 2], '--r', markersize=1, label='Estimated State')  # Full line for estimated state
ax.set_title('Lorenz Attractor with Noisy Data and Reservoir Computing')
ax.legend()
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)')
plt.show()
