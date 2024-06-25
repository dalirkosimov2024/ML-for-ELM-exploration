import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
import matplotlib.lines as mlines

basepath = "/home/lcv10/Documents/ML_for_ELMs/Logbook/Pictures/2D_Bayesian_optimisation/"
# Define the parameters
a = 2.21
h = 1
k = -10

# Define the 2D function
def target_function(X):
    x, y = X
    return (x * np.cos(a) + y * np.sin(a) - h)**2 + k + x * np.sin(a) - y * np.cos(a)

# Generate a grid of points
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
Z = target_function([X, Y])

# Flatten the grid for training data
xy = np.vstack([X.ravel(), Y.ravel()]).T
z = Z.ravel()

# Select the kernel and fit the Gaussian Process model
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Initial training points
np.random.seed(1)
initial_indices = np.random.choice(len(xy), 10, replace=False)
x_train = xy[initial_indices]
y_train = z[initial_indices]

gpr.fit(x_train, y_train)

# Predict on the entire grid
mean_prediction, std_prediction = gpr.predict(xy, return_std=True)
mean_prediction = mean_prediction.reshape(X.shape)
std_prediction = std_prediction.reshape(X.shape)

# Plotting function
def plot_gp(gpr, x_train, y_train, X, Y, Z, next_point, label, iteration, label_for_saving):
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    mean_prediction, std_prediction = gpr.predict(xy, return_std=True)
    mean_prediction = mean_prediction.reshape(X.shape)
    std_prediction = std_prediction.reshape(X.shape)

    fig, ax = plt.subplots(figsize=(6.,4.))

    # Plot the true function
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$j_{\parallel}$", rotation=0)
    ax.contour(X, Y, Z, [0], colors="grey")
    plt.suptitle(f"2D fit with {label} acquisition function, iteration {iteration}")
    ax.scatter(x_train[:, 0], x_train[:, 1], color="blue", marker='x', label='Training Points')

    # Plot the GP mean prediction
    ax.contour(X, Y, mean_prediction, [0], colors="red", linestyles="--", label="Posterior mean")
    ax.scatter(next_point[0],next_point[1], color="green", marker="o")
    truth_line = mlines.Line2D([], [], color="grey", label="Stability boundary")
    sample_line = mlines.Line2D(
        [], [], color="blue", marker="x", linestyle="none", label=f"Training points"
    )
    mean_line = mlines.Line2D([], [], color="red", linestyle="--", label="Posterior mean")
    next_point = mlines.Line2D([], [], color="green", marker="o",linestyle="none", label="Next point")

    ax.legend(
        handles=[truth_line, sample_line, mean_line, next_point],
        bbox_to_anchor=(1, 1),
        loc="upper left",
    )
    ax.set_xlim(0,10)
    ax.set_ylim(0,10)
    plt.tight_layout()
    

    




# Expected improvement function
def expected_improvement(x, gpr, best_y):
    mean_prediction, std_prediction = gpr.predict(x.reshape(-1, 2), return_std=True)
    z = (mean_prediction - best_y) / std_prediction
    ei = (mean_prediction - best_y) * norm.cdf(z) + std_prediction * norm.pdf(z)
    return ei

# Upper confidence bound function
def upper_confidence_bound(x, gpr, beta):
    mean_prediction, std_prediction = gpr.predict(x.reshape(-1, 2), return_std=True)
    ucb = mean_prediction + beta * std_prediction
    return ucb

# Perform Bayesian optimization
def bayesian_optimization(gpr, x_train, y_train, num_iter=10, acquisition='ei'):
    for i in range(num_iter):
        if acquisition == 'ei':
            best_y = max(y_train)
            ei = expected_improvement(xy, gpr, best_y)
            next_point = xy[np.argmax(ei)]
            label = str("expected improvement")
            label_for_saving = str("expected_improvement")
        elif acquisition == 'ucb':
            ucb = upper_confidence_bound(xy, gpr, beta=2.0)
            next_point = xy[np.argmax(ucb)]
            label = str("upper confidence bound")
            label_for_saving = str("upper_confidence_bound")
        
        # Add new point to training data
        new_y = target_function(next_point)
        x_train = np.vstack([x_train, next_point])
        y_train = np.append(y_train, new_y)
        
        # Retrain the model
        gpr.fit(x_train, y_train)
        
        # Plot the results
        plot_gp(gpr, x_train, y_train, X, Y, Z, next_point=next_point, label = label, iteration = i, label_for_saving=label_for_saving)
        print(next_point)
        
        
        
# Run Bayesian optimization with expected improvement
bayesian_optimization(gpr, x_train, y_train, num_iter=10, acquisition='ei')

# Run Bayesian optimization with upper confidence bound
bayesian_optimization(gpr, x_train, y_train, num_iter=10, acquisition='ucb')
