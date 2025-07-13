# This code is used for creating models used for the paper.
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
from Optimizer import Optimizer
from ReservoirComputer import ReservoirComputer

# Example usage
if __name__ == "__main__":
    # Read generated data corresponding to Lorenz Attractor
    x_true2 = np.loadtxt('.\Datasets\Lorenz_Dataset100p\X_true.txt')
    X_train2 = np.loadtxt('.\Datasets\Lorenz_Dataset100p\X_train.txt')
    y_train2 = np.loadtxt('.\Datasets\Lorenz_Dataset100p\y_train.txt')
    Time_train2 = np.loadtxt('.\Datasets\Lorenz_Dataset100p\Time_train.txt')
    X_test2 = np.loadtxt('.\Datasets\Lorenz_Dataset100p\X_test.txt')
    y_test2 = np.loadtxt('.\Datasets\Lorenz_Dataset100p\y_test.txt')
    Time_test2 = np.loadtxt('.\Datasets\Lorenz_Dataset100p\Time_test.txt')
    Time_Total2 = np.concatenate((Time_train2, Time_test2), axis=0)

    # Read generated data corresponding to Lorenz Attractor
    x_true3 = np.loadtxt('.\Datasets\Lorenz_Dataset25p\X_true.txt')
    X_train3 = np.loadtxt('.\Datasets\Lorenz_Dataset25p\X_train.txt')
    y_train3 = np.loadtxt('.\Datasets\Lorenz_Dataset25p\y_train.txt')
    Time_train3 = np.loadtxt('.\Datasets\Lorenz_Dataset25p\Time_train.txt')
    X_test3 = np.loadtxt('.\Datasets\Lorenz_Dataset25p\X_test.txt')
    y_test3 = np.loadtxt('.\Datasets\Lorenz_Dataset25p\y_test.txt')
    Time_test3 = np.loadtxt('.\Datasets\Lorenz_Dataset25p\Time_test.txt')
    Time_Total3 = np.concatenate((Time_train3, Time_test3), axis=0)

    # Read generated data corresponding to Lorenz Attractor
    x_true4 = np.loadtxt('.\Datasets\Lorenz_Dataset25p_sigma8\X_true.txt')
    X_train4 = np.loadtxt('.\Datasets\Lorenz_Dataset25p_sigma8\X_train.txt')
    y_train4 = np.loadtxt('.\Datasets\Lorenz_Dataset25p_sigma8\y_train.txt')
    Time_train4 = np.loadtxt('.\Datasets\Lorenz_Dataset25p_sigma8\Time_train.txt')
    X_test4 = np.loadtxt('.\Datasets\Lorenz_Dataset25p_sigma8\X_test.txt')
    y_test4 = np.loadtxt('.\Datasets\Lorenz_Dataset25p_sigma8\y_test.txt')
    Time_test4 = np.loadtxt('.\Datasets\Lorenz_Dataset25p_sigma8\Time_test.txt')
    Time_Total4 = np.concatenate((Time_train4, Time_test4), axis=0)

    # Contacenation
    X_train = np.concatenate((X_train2, X_train3, X_train4), axis=0)
    y_train = np.concatenate((y_train2, y_train3, y_train4), axis=0)
    X_test = np.concatenate((X_test2, X_test3, X_test4), axis=0)
    y_test = np.concatenate((y_test2, y_test3, y_test4), axis=0)

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
        torch.save(rc, './Models/Model_Initial_25p_robust' + Error_char[pp] + '.pth')

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
        torch.save(rc, './Models/Model_Tuned_25p_robust' + Error_char[pp] + '.pth')
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
        torch.save(rc, './Models/Model_Pruned_25p_robust' + Error_char[pp] + '.pth')

