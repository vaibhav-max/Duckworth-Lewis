import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union
import scipy.optimize as opt
from tqdm import tqdm

if not os.path.exists('/data/home/vvaibhav/AI/DA/models'):
    os.makedirs('/data/home/vvaibhav/AI/DA/models')
if not os.path.exists('/data/home/vvaibhav/AI/DA/plots'):
    os.makedirs('/data/home/vvaibhav/AI/DA/plots')

class DLModel:
    """
        Model Class to approximate the Z function as defined in the assignment.
    """

    def __init__(self):
        """Initialize the model."""
        self.Z0 = [None] * 10
        self.L = None

    def get_predictions(self, X, Z_0=None, w=10, L=None) -> np.ndarray:

        """Get the predictions for the given X values."""
        L = self.L
        Z_0 = self.Z0[w-1]
        return Z_0 * (1 - np.exp(-L * X / Z_0))
    

    def calculate_loss(self, Params, X, Y, w=10) -> float:
        """Calculate the loss for the given parameters and datapoints."""
        Z_0, L = Params[w-1], Params[-1]
        predictions = self.get_predictions(X, Z_0, w, L)
        return np.mean((predictions - Y) ** 2)

    def save(self, path):
        """Save the model to the given path.

        Args:
            path (str): Location to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump((self.L, self.Z0), f)
    
    def load(self, path):
        """Load the model from the given path.

        Args:
            path (str): Location to load the model.
        """
        with open(path, 'rb') as f:
            (self.L, self.Z0) = pickle.load(f)

def get_data(data_path) -> Union[pd.DataFrame, np.ndarray]:
    """Loads the data from the given path and returns a pandas dataframe."""
    data = pd.read_csv(data_path)
    return data

def preprocess_data(data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """Preprocesses the dataframe by cleaning and formatting."""
    data = data.dropna()  # Remove rows with missing values
    data = data[data['Error.In.Data'] != 1]
    data = data[data['Innings'] == 1]
    data = data.drop(['Date'], axis=1)  # Remove unnecessary columns
    return data

def fit_function(data, params):
    L_value = params[-1]
    Z0_values = params[:-1]
    predictions = []

    #print("fit_function")
    for w in range(1, 11):
        filtered_data = data[data['Wickets.in.Hand'] == w]
        #filtered_data = filtered_data.loc[filtered_data.groupby('Match')['Runs.Remaining'].idxmax()]
        overs = filtered_data['Over'].values
        wickets = filtered_data['Wickets.in.Hand'].values

        #print(overs.shape, wickets.shape)
        for u, w in zip(overs, wickets):
            Z0_w = Z0_values[w-1]
            predictions.append(Z0_w * (1 - np.exp(-L_value * u / Z0_w)))

    return np.array(predictions)

def train_model(data: pd.DataFrame, model: DLModel) -> DLModel:
    
    dfs_list = []

    print("train_model")
    for w in tqdm(range(1, 11)):
        filtered_data = data[data['Wickets.in.Hand'] == w]
        #filtered_data = filtered_data.loc[filtered_data.groupby('Match')['Runs.Remaining'].idxmax()]
        dfs_list.append(filtered_data)

    # Concatenate all DataFrames in the list
    result = pd.concat(dfs_list, ignore_index=True)

    Z_data = result['Runs.Remaining'].values  #By using .values, you can work with the data as a NumPy array
 
    w_10_data = result[result['Wickets.in.Hand'] == 10]
    Z_data_w_10 = w_10_data['Runs.Remaining'].values

    # Initial parameter values: Z0 values and L
    initial_params = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 0.005])

   # Define bounds for Z0 values and L
    Z0_bounds = [(0, None)] * 10  # Z0 values should be between 0 and 330
    L_bounds = (0, 10)  # L should be less than 10

    bounds = Z0_bounds + [L_bounds]

    # Define constraints for Z0 order
    def z0_order_constraint(params):
    # Check that the differences between consecutive Z0 values are non-negative 
        z0_differences = np.diff(params)
        min_difference = 35
        return min_difference - z0_differences
    
    z0_constraints = {'type': 'ineq', 'fun': z0_order_constraint}

    # Perform L-BFGS-B optimization with bounds and constraints
    result = opt.minimize(
        fun=lambda params: np.mean((fit_function(data, params) - Z_data) ** 2),
        x0=initial_params,
        method='SLSQP', #bfgs
        bounds=bounds,
        constraints=z0_constraints,
        options={'maxiter': 100}  # Adjust the maximum number of iterations as needed
    )

    # Extract the optimized parameters
    optimized_params = result.x

    L_opt = optimized_params[-1]
    Z0_opt = optimized_params[:-1]

    # Update the model with optimized parameters
    model.L = L_opt
    model.Z0 = Z0_opt.tolist()

    return model

def plot(data, model: DLModel, plot_path: str) -> None:

    """Plots the model predictions against the number of overs."""
    
    Z0_values = model.Z0
    L_value = model.L
    # Z0_values = [7.845940149290002, 16.8899268186723, 29.933382439128135, 45.62858992368021, 67.8621951011102, 96.42544605794794, 130.78650496367416, 166.9818333220018, 203.08489046123785, 239.0145634344657]
    # L_value = 11
    # Generate a range of overs from 0 to 50
    overs = np.linspace(0, 50, 100)

    # Initialize an array to store predicted scores
    predicted_scores = []

    # Calculate predicted scores using the formula for each set of Z0 values and overs
    for Z0 in Z0_values:
        predicted_score = Z0 * (1 - np.exp(-L_value * overs / Z0))
        predicted_scores.append(predicted_score)
        #print(Z0, predicted_score)

    # Plot the predicted scores against the number of overs
    plt.figure(figsize=(10, 6))
    for i, predicted_score in enumerate(predicted_scores):
        plt.plot(overs, predicted_score, label=f'w={i+1}')

    plt.xlabel('Number of Overs Remaining')
    plt.ylabel('Predicted Score')
    plt.title('Predicted Scores vs Number of Overs')
    plt.legend()
    plt.savefig(plot_path)
    plt.grid(True)
    plt.show()


def print_model_params(model: DLModel) -> List[float]:
    """Prints the 11 model parameters."""
    print("Z0 values:", model.Z0)
    print("L value:", model.L)
    return model.Z0 + [model.L]

def calculate_loss(model: DLModel, data: Union[pd.DataFrame, np.ndarray]) -> float:
    """Calculates the normalized squared error loss for the model and data."""
    X = data['Over']
    Y = data['Runs']
    losses = []
    print("calculate_loss")
    for w in tqdm(range(1, 11)):
        predictions = model.get_predictions(X, w=w)
        loss = np.mean((predictions - Y) ** 2)
        losses.append(loss)
    normalized_loss = np.sum(losses) / len(data)
    print("Normalized Squared Error Loss:", normalized_loss)
    return normalized_loss

def main(args):
    """Main Function"""

    data = get_data(args['data_path'])  # Loading the data
    print("Data loaded.")
    
    data = preprocess_data(data)  # Preprocess the data
    print("Data preprocessed.")
    
    model = DLModel()  # Initializing the model
    #train_model(data, model)

    model = train_model(data, model)  # Training the model
    model.save(args['model_path'])  # Saving the model
    
    plot(data, model, args['plot_path'])  # Plotting the model
    
    print_model_params(model)  # Printing the model parameters

    calculate_loss(model, data)  # Calculate the normalized squared error

    

# ... (remaining code)
if __name__ == '__main__':
    args = {
        "data_path": "/data/home/vvaibhav/AI/DA/04_cricket_1999to2011.csv",
        "model_path": "/data/home/vvaibhav/AI/DA/models/model.pkl",  # ensure that the path exists
        "plot_path": "/data/home/vvaibhav/AI/DA/plots/plot.png",  # ensure that the path exists
    }
    main(args)