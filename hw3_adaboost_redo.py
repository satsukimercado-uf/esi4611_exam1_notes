import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import os
import sys
import warnings
from typing import List, Tuple, Optional, Union
import pandas as pd
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import tree
from sklearn.datasets import make_classification, make_circles

def plot_stump_decision(x: np.ndarray, 
                       y: np.ndarray, 
                       model: DecisionTreeClassifier,
                       dataset_weights: Optional[np.ndarray] = None) -> Tuple[plt.Figure, plt.Axes]:
    
    # Create a matplotlib figure and axes
    plt.clf()
    fig, ax = plt.subplots(figssize=(8,6))

    # Plot positive and negative samples with different markers based on dataset_weights
    n = y.shape[0]
    size_scale = 100*n
    pos_size = size_scale*dataset_weights[data[:,-1] == 1]
    neg_size = size_scale*dataset_weights[data[:,-1] == -1]

    y = y.reshape(-1,1) # turn into (n,1) array
    data = np.hstack(x,y) # combine into one (n,3) array with 2 features 1 label
    pos = data[data[:,-1] == 1]
    neg = data[data[:,-1] == -1]

    ax.scatter(pos[:,0], pos[:,1], c="blue", marker="+", s=pos_size)
    ax.scatter(neg[:,0], neg[:,1], c="red", marker="_", s=neg_size) 

    # Create a mesh to visualize the decision boundary, code given in adaboost_practice in hw3
    DecisionBoundaryDisplay.from_estimator(
    model,
    x,
    response_method='predict',
    xlabel='feature1', ylabel='feature2',
    alpha=0.1, colors=["orange", "black", "blue", "black"],
    ax= ax
)

    # Use model.predict() to get predictions and identify misclassified points
    y_pred = model.predict(x)
    y_true = y
    misclassified = data[data[:,-1] != y_pred]  # return entire entries, features & labels 
    print(misclassified)

    # Circle misclassified points, code given in adaboost_practice in hw3
    ax.scatter(misclassified[:,0], misclassified[:,1], facecolors='none', edgecolors='red', s=200, linewidths=2)

    # Return the figure and axes 
    plt.title("Adaboost")
    plt.xlabel('feature1')
    plt.ylabel('feature2')
    plt.show()

    return fig, ax

def adaboost_round(x: np.ndarray, 
                  y: np.ndarray, 
                  dataset_weights: np.ndarray) -> Tuple[DecisionTreeClassifier, np.ndarray, float]:
    
    # Create and train a DecisionTreeClassifier with max_depth = 1 and sample_weight = dataset_weights
    stump = DecisionTreeClassifier(max_depth = 1)
    dataset_weights = dataset_weights.reshape(-1,1) # turns into (n,1) array

    stump.fit(x,y,sample_weight=dataset_weights)

    # Get predictions and calculate weighted error rate
    y_pred = stump.predict(x).reshape(-1,1) # turns into (n,1) array

    errors = y_pred != y # returns (n,1) array of False/ True that is treated as 0/1
    weighted_error = np.dot(dataset_weights, errors)

    # Calculate model weight (w) = 0.5 * ln((1 - weighted_error) / weighted_error)
    zero_epsilon = 1e-1 # used to correct potentially dividing by zero
    model_weight = 0.5*np.log((1-weighted_error)/(weighted_error+zero_epsilon))

    # Update dataset weights alpha = alpha*e^(+- weightederror) 
    updated_alphas = []
    for alpha, y_pred, y in zip(dataset_weights, y_pred, y):
        if y_pred == y: # correctly predicted, weight should go down
            new_alpha = alpha *np.exp(-model_weight)
        else: # misclassified, weight should go up
            new_alpha = alpha *np.exp(-model_weight)
        updated_alphas.append(new_alpha)

    # Normalize the weights so that they add up to 1
    z = sum(updated_alphas)
    updated_alphas = updated_alphas/z

    # Return values model, updated dataset weights, model weight
    return stump, updated_alphas, model_weight

def run_adaboost(x: np.ndarray, 
                y: np.ndarray, 
                r: int) -> Tuple[List[DecisionTreeClassifier], List[float]]:

    # Initialize uniform weights: dataset_weights = np.ones(len(y))/ len(y)
    dataset_weights = np.ones(len(y))/ len(y)

    # Initialize lists to store results of stumps and model_weights for each iteration of adaboost
    stumps = []
    model_weights = []

    # Create 'figs' directory to store matplotlib figs, code given in hw3
    os.makedirs('figs', exist_ok=True)

    # For each round, r, of adaboost
    for i in range(r):
        # Call adaboost_round() to get current stump, updated_weights, and model_weight 
        stump, dataset_weights, model_weight = adaboost_round(x,y,dataset_weights)

        # Store the stump by calling tree.plot_tree to plot a stump
        stumps.append(stump)
        model_weights.append(model_weight)

        # Call plot_decision_stump, create and save a plot
        fig, ax = plot_stump_decision(x, y,stump, dataset_weights)
        plt.savefig(f'figs/round{i+1}.png')

        # Return stumps and model_weights list
        return stumps, model_weights
    

    # def plot_ensemble(x: np.ndarray, 
    #              y: np.ndarray, 
    #              stumps: List[DecisionTreeClassifier], 
    #              model_weights: List[float]) -> Tuple[plt.Figure, plt.Axes]:
    
    #     # Create array of (n,1) for stump predictions
    #     stumps_pred = []
    #     for st in stumps:
    #         pred = st.predict(x)
    #         stumps_pred.append(pred)

    #     # Compute weighted sum 
    #     weighted_sum = np.dot(stumps_pred, model_weights)

    #     # Find final ensemble prediction, which is the sign of the weighted sum 
    #     final_y_pred = np.sign(weighted_sum)

    #     # Reuse code in plot_stump_decision to draw decision boundary and circle misclassified points 
    #     data = np.hstack(x,y)
    #     misclassified = data[data[:,-1] != final_y_pred]  # return entire entries, features & labels 
    #     print(misclassified)


# class AdaBoostEnsemble:
#     """
#     Optional class-based implementation of AdaBoost ensemble.
#     Students can use this approach for more advanced coding practice.
#     """

#     def __init__(self):
#         self.stumps: List[DecisionTreeClassifier] = []
#         self.model_weights: List[float] = []

#     def fit(self, X: np.ndarray, y: np.ndarray, r: int):
#         """Fit the AdaBoost ensemble."""
#         # TODO: Students can implement this for practice
#         pass

#     def predict(self, X: np.ndarray) -> np.ndarray:
#         """Make predictions using the ensemble."""
#         # TODO: Students can implement this for practice
        pass

    