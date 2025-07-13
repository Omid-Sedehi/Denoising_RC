import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ReservoirComputer import ReservoirComputer

# Example usage
if __name__ == "__main__":
    # Read generated data corresponding to Lorenz Attractor
    x_true = np.loadtxt('.\Dataset\AINF_Dataset10p_GWN\X_true.txt')
    X_train = np.loadtxt('.\Dataset\AINF_Dataset10p_GWN\X_train.txt')
    y_train = np.loadtxt('.\Dataset\AINF_Dataset10p_GWN\y_train.txt')
    Time_train = np.loadtxt('.\Dataset\AINF_Dataset10p_GWN\Time_train.txt')
    X_test = np.loadtxt('.\Dataset\AINF_Dataset10p_GWN\X_test.txt')
    y_test = np.loadtxt('.\Dataset\AINF_Dataset10p_GWN\y_test.txt')
    Time_test = np.loadtxt('.\Dataset\AINF_Dataset10p_GWN\Time_test.txt')
    Time = np.loadtxt('.\Dataset\AINF_Dataset10p_GWN\Time.txt')
    Time_Total = np.concatenate((Time_train, Time_test), axis=0)

    mu1_x_true = np.average(x_true[:,0])
    mu2_x_true = np.average(x_true[:,1])
    sd1_x_true = np.std(x_true[:,0])
    sd2_x_true = np.std(x_true[:,1])
    mu1_X_train = np.average(X_train[:,0])
    mu2_X_train = np.average(X_train[:,1])
    sd1_X_train = np.std(X_train[:,0])
    sd2_X_train = np.std(X_train[:,1])
    mu1_y_train = np.average(y_train[:,0])
    mu2_y_train = np.average(y_train[:,1])
    sd1_y_train = np.std(y_train[:,0])
    sd2_y_train = np.std(y_train[:,1])

    X_train[:, 0] = (X_train[:, 0] - mu1_X_train) / sd1_X_train
    X_train[:, 1] = (X_train[:, 1] - mu2_X_train) / sd2_X_train
    X_test[:, 0] = (X_test[:, 0] - mu1_X_train) / sd1_X_train
    X_test[:, 1] = (X_test[:, 1] - mu2_X_train) / sd2_X_train
    y_train[:, 0] = (y_train[:, 0] - mu1_y_train) / sd1_y_train
    y_train[:, 1] = (y_train[:, 1] - mu2_y_train) / sd2_y_train
    y_test[:, 0] = (y_test[:, 0] - mu1_y_train) / sd1_y_train
    y_test[:, 1] = (y_test[:, 1] - mu2_y_train) / sd2_y_train

    Nrmse = np.zeros((20,1))
    Gain = np.zeros((20,1))
    # Create and train the initial reservoir
    for qq in range(20):
        x_true[:, 0] = (x_true[:, 0] - mu1_x_true) / sd1_x_true
        x_true[:, 1] = (x_true[:, 1] - mu2_x_true) / sd2_x_true

        rc = ReservoirComputer(n_inputs=2, n_outputs=2, n_nodes=50)
        initial_nodes = rc.get_nodes()
        rc.fit(X_train[:,0:2], y_train)

        # Evaluate initial performance
        y_pred_initial = rc.predict(X_test[:,0:2])
        mse_initial = mean_squared_error(y_test, y_pred_initial)
        print(f"Initial MSE: {mse_initial:.6f}")

        # Perform hyperparameter tuning
        print("\nPerforming hyperparameter tuning...")
        best_hyper_params, best_hyper_score = rc.tune_hyperparameters(X_train[:,0:2], y_train, X_test[:,0:2], y_test)
        X_Total = np.concatenate((X_train[:,0:2], X_test[:,0:2]))

        # Evaluate after tuning
        y_pred_tuned = rc.predict(X_Total)
        y_pred_test_tuned = rc.predict(X_test[:,0:2])
        mse_tuned = mean_squared_error(y_test, y_pred_test_tuned)
        tuned_nodes = rc.get_nodes()

        print("\nHyperparameter Tuning Results:")
        print(f"Tuned MSE: {mse_tuned:.6f}")
        print(f"Best hyperparameters:")
        print(f"  n_nodes: {rc.n_nodes}")
        print(f"  spectral_radius: {rc.spectral_radius:.4f}")
        print(f"  leaking_rate: {rc.leaking_rate:.4f}")
        print(f"  input_scaling: {rc.input_scaling:.4f}")
        print(f"  connectivity: {rc.connectivity:.4f}")

        # Save the entire model
        # torch.save(rc, './Models_GWN_Error/Model_Tuned_10p_GWN'+qq+'.pth')

    ####### Perform Pruning ###########################################################################################
        # Perform pruning
        print("\nPerforming one-shot pruning...")
        best_pruning_params, best_pruning_score = rc.one_shot_pruning(X_train, y_train, X_test, y_test)

        # Evaluate after pruning
        y_pred_pruned = rc.predict(X_Total)
        y_pred_test_pruned = rc.predict(X_test)
        mse_pruned = mean_squared_error(y_test, y_pred_test_pruned)
        pruned_nodes = rc.get_nodes()

        print("\nPruning Results:")
        print(f"Pruned MSE: {mse_pruned:.6f}")
        print(f"Best pruning parameters: {best_pruning_params}")
        print(f"Best pruning score: {best_pruning_score}")

        print("\nPlot saved as 'reservoir_comparison.png'")
        print(f"Initial nodes: {initial_nodes}")
        print(f"Tuned nodes: {tuned_nodes}")
        print(f"Pruned nodes: {pruned_nodes}")

        # Save the entire model
        torch.save(rc, './Models_Pink_Error/Model_Pruned_10p_GWN'+ str(qq) +'.pth')

        # Post-processing step
        x_true[:,0] = sd1_x_true*x_true[:, 0] + mu1_x_true
        x_true[:,1] = sd2_x_true*x_true[:, 1] + mu2_x_true
        # X_Total[:, 0] = sd1_X_train*X_Total[:, 0] + mu1_X_train
        # X_Total[:, 1] = sd2_X_train*X_Total[:, 1] + mu2_X_train
        y_pred_tuned[:, 0] = sd1_y_train*y_pred_pruned[:, 0] + mu1_y_train
        y_pred_tuned[:, 1] = sd2_y_train*y_pred_pruned[:, 1] + mu2_y_train
        x_true_mse1 = ((x_true[:, 0] ** 2).mean())**0.5
        x_true_mse2 = ((x_true[:, 1] ** 2).mean())**0.5
        Nmse_indiv = np.array([mean_squared_error(x_true[:, 0], y_pred_tuned[:, 0])**0.5 / x_true_mse1,
                               mean_squared_error(x_true[:, 1], y_pred_tuned[:, 1])**0.5 / x_true_mse2])
        Nrmse[qq,0] = Nmse_indiv.mean()
        SNR2 = 10
        SNR_Test = 1 / Nrmse[qq,0]
        Gain[qq,0] = SNR_Test / SNR2

    np.savez('Error_Gain_GWN.npz', Nmse=Nrmse, Gain = Gain)
