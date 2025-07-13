# This code is used for creating models used for the paper.
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
from ReservoirComputer import ReservoirComputer


# Example usage
if __name__ == "__main__":
    # Read generated data corresponding to Lorenz Attractor
    x_true = np.loadtxt('.\Datasets\Lorenz_Dataset25p\X_true.txt')
    X_train = np.loadtxt('.\Datasets\Lorenz_Dataset25p\X_train.txt')
    y_train = np.loadtxt('.\Datasets\Lorenz_Dataset25p\y_train.txt')
    Time_train = np.loadtxt('.\Datasets\Lorenz_Dataset25p\Time_train.txt')
    X_test = np.loadtxt('.\Datasets\Lorenz_Dataset25p\X_test.txt')
    y_test = np.loadtxt('.\Datasets\Lorenz_Dataset25p\y_test.txt')
    Time_test = np.loadtxt('.\Datasets\Lorenz_Dataset25p\Time_test.txt')
    Time_Total = np.concatenate((Time_train, Time_test), axis=0)

    row_size = 10
    Error_char = np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

    for pp in range(0,row_size):
        # Create and train the initial reservoir
        rc = ReservoirComputer(n_inputs=2, n_outputs=3, n_nodes=500)
        initial_nodes = rc.get_nodes()
        rc.fit(X_train[:, 0:2], y_train)

        # Evaluate initial performance
        y_pred_initial = rc.predict(X_test[:, 0:2])
        mse_initial = mean_squared_error(y_test, y_pred_initial)
        print(f"Initial MSE: {mse_initial:.6f}")
        torch.save(rc, './Models/Model_Initial_25p' + Error_char[pp] + '.pth')

        # Perform hyperparameter tuning
        print("\nPerforming hyperparameter tuning...")
        best_hyper_params, best_hyper_score = rc.tune_hyperparameters(X_train[:, 0:2], y_train, X_test[:, 0:2], y_test)
        X_Total = np.concatenate((X_train[:, 0:2], X_test[:, 0:2]))

        # Evaluate after tuning
        y_pred_tuned = rc.predict(X_Total[:, 0:2])
        y_pred_test_tuned = rc.predict(X_test[:, 0:2])
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
        torch.save(rc, './Models/Model_Tuned_25p' + Error_char[pp] + '.pth')
        # Perform pruning
        print("\nPerforming one-shot pruning...")
        best_pruning_params, best_pruning_score = rc.one_shot_pruning(X_train[:,0:2], y_train, X_test[:,0:2], y_test)

        # Evaluate after pruning
        y_pred_pruned = rc.predict(X_Total[:,0:2])
        y_pred_test_pruned = rc.predict(X_test[:,0:2])
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
        torch.save(rc, './Models/Model_Pruned_25p' + Error_char[pp] + '.pth')

