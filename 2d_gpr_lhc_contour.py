import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
import matplotlib.lines as mlines
from sklearn.metrics import mean_squared_error 
from contourpy import contour_generator
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable 



# Global variables 
n_samples = 6
test_size = 0.1
basepath = "/home/lcv10/Documents/ml_for_elms/logbook/pictures/9_7_24"

# Stability function definition
def custom_function(x, y, a=2.21, h=1, k=-10):
    return (x * np.cos(a) + y * np.sin(a) - h)**2 + k + x * np.sin(a) - y * np.cos(a)

# Generate sample points using Latin Hypercube Sampling
def generate_latin_hypercube_samples(n_samples, dimensions):
    sampler = qmc.LatinHypercube(d=dimensions)
    sample = sampler.random(n=n_samples)
    # Transform the sample to the desired range, for example, [0, 10] for both x and y
    sample = qmc.scale(sample, l_bounds=[0, 0], u_bounds=[10, 10])
    return sample

# Function that iterates a GPR process over a LHC sampling distribution
def lhc_gpr():
    rmse_list  = np.array([])

    # Call LHC function
    dimensions = 2
    samples = generate_latin_hypercube_samples(n_samples, dimensions)

    # Evaluate the function at these sample points
    z_values = np.array([custom_function(x, y) for x, y in samples])

    # Split the data into a training and testing set 
    x_train, x_test, y_train, y_test = train_test_split(samples, z_values, test_size=test_size)

    # Select a constant and RBF kernel
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    # Train the model
    gp.fit(x_train, y_train)

    # Generate a grid of points over the region of interest
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    x_grid, y_grid = np.meshgrid(x, y)
    grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    # Define the 
    z = np.array([custom_function(xi, yi) for xi, yi in grid_points]).reshape(x_grid.shape)
    print(f"z: {len(z)} \n")

    # Predict the function values at the grid points
    zfit, z_pred_std = gp.predict(grid_points, return_std=True)
    zfit = zfit.reshape(x_grid.shape)
    z_pred_std = z_pred_std.reshape(x_grid.shape)
    line_x, line_y, next_point, zfit, z_pred_std = acquisition_function(gp, grid_points, x_grid)





    # Calculating RMSE
    mse = mean_squared_error(z, zfit)
    rmse = np.sqrt(mse)
    rmse_list = np.append(rmse_list, rmse)

    # Plotting the results
    
    fig, ax = plt.subplots(figsize=(6.,4.))
    
    ax.contour(x_grid, y_grid, z, [0], color="grey", alpha=0.4)
    ax.contour(x_grid, y_grid, zfit,[0], colors='red', linestyles='--')
    ax.scatter(x_train[:, 0], x_train[:, 1], marker="x", color="blue")
    ax.scatter(x_test[:, 0], x_test[:, 1], color="orange")
    truth_line = mlines.Line2D([], [], color="grey", label="Stability boundary")
    mean_line = mlines.Line2D([], [], color="red", linestyle="--", label="Posterior mean")
    sample_line = mlines.Line2D([], [], color="blue", marker="x", linestyle="none", label="Training points")
    testing_points = mlines.Line2D([], [], color="orange", marker="o",linestyle="none", label="Testing Points")
    ax.legend(
        handles=[truth_line, sample_line, mean_line, testing_points],
        bbox_to_anchor=(1, 1),
        loc="upper left",)
    plt.suptitle(f"Toy stability space with {n_samples} initial points")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$j_{\parallel}$", rotation=0)
    ax.set_xticks((0, 0, 10))
    ax.set_yticks((0, 0, 10))
    ax.set_xlim(0,10)
    ax.set_ylim(0,10)
    plt.tight_layout()


    num_iter = 25

    for i in range(num_iter):
            # add new x and y points to training set
            new_y = custom_function(next_point[0], next_point[1])
            x_train = np.vstack([x_train, next_point])
            y_train = np.append(y_train, new_y)
            # train model
            gp.fit(x_train, y_train)

            # get posterior mean
            line_x, line_y, next_point, zfit, z_pred_std= acquisition_function(gp, grid_points, x_grid)
            mse = mean_squared_error(z, zfit)
            rmse = np.sqrt(mse)
            rmse_list = np.append(rmse_list, rmse)
            # plot
            fig, ax = plt.subplots(figsize=(6.,4.))
            ax.scatter(next_point[0], next_point[ 1], color="green", marker="o")
            ax.contour(x_grid, y_grid, z, [0], color="grey", alpha=0.4)
            ax.contour(x_grid, y_grid, zfit,[0], colors='red', linestyles='--')
            ax.scatter(x_train[:, 0], x_train[:, 1], marker="x", color="blue")
            ax.scatter(x_test[:, 0], x_test[:, 1], color="orange")
            truth_line = mlines.Line2D([], [], color="grey", label="Stability boundary")
            mean_line = mlines.Line2D([], [], color="red", linestyle="--", label="Posterior mean")
            sample_line = mlines.Line2D([], [], color="blue", marker="x", linestyle="none", label="Training points")
            testing_points = mlines.Line2D([], [], color="orange", marker="o",linestyle="none", label="Testing Points")
            next_points = mlines.Line2D([], [], color="green", marker="o",linestyle="none", label="Next Point")
            ax.legend(
                handles=[truth_line, mean_line, sample_line,testing_points, next_points],
                bbox_to_anchor=(1, 1),
                loc="upper left",)
            plt.suptitle(f"Toy stability space with {n_samples+i+1} points with a custom acquisition function")
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$j_{\parallel}$", rotation=0)
            ax.set_xticks((0, 0, 10))
            ax.set_yticks((0, 0, 10))
            ax.set_xlim(0,10)
            ax.set_ylim(0,10)            
            contour_plot = ax.contourf(x_grid, y_grid, z_pred_std, cmap="plasma", alpha=0.25)
            cbar = plt.colorbar(contour_plot, shrink=0.55,anchor=(0, 0))
            cbar.set_label("Standard deviation")
            plt.tight_layout()
            plt.show()
            
    with open("rmse.csv", "ab") as f:
        np.savetxt(f, rmse_list)

        
        
def acquisition_function(gp, grid_points, x_grid):
        zfit, z_pred_std = gp.predict(grid_points, return_std=True)
        zfit = zfit.reshape(x_grid.shape)
        z_pred_std = z_pred_std.reshape(x_grid.shape)

        # attain array of posterior mean contour
        cont_gen = contour_generator(z=zfit)
        lines = cont_gen.lines(0)
        line_x = np.array([])
        line_y = np.array([])
        for x in lines:
            for i in x:
                line_x = np.append(line_x, i[0]/10)
                line_y = np.append(line_y, i[1]/10)

        # array of posterior mean
        arr = np.stack((line_x, line_y),axis=1)

        # select random index from array as the next point
        next_point = random.choice(arr)
        return line_x, line_y, next_point, zfit, z_pred_std

   
def rmse_reader():
     rmse_custom_acq_func = np.loadtxt("rmse.csv")
     rmse_norm = np.loadtxt("rmse_norm.csv")
     rmse_norm = np.array_split(rmse_norm,11)
     rmse_custom_acq_func = np.array_split(rmse_custom_acq_func,11)

     for i in range(len(rmse_custom_acq_func)): 
        plt.plot(range(len(rmse_custom_acq_func[0])), rmse_custom_acq_func[i], linestyle="--", label=f"iter {i+1}")
     plt.legend()
     plt.title("RMSE against number of observations with the custom acquisition function")
     plt.xlabel("Number of observations")
     plt.ylabel("RMSE")
     plt.show()

     rmse_custom_acq_func = averager(rmse_custom_acq_func)
     rmse_norm = averager(rmse_norm)
     
     plt.plot(range(len(rmse_custom_acq_func)), rmse_custom_acq_func, color="blue",label = "custom acquisition function (random selection)")
     plt.plot(range(len(rmse_custom_acq_func)), rmse_custom_acq_func, color="blue", marker="o")
     plt.plot(range(len(rmse_norm)), rmse_norm, color="red", label = "latin hypercube sampling")
     plt.plot(range(len(rmse_norm)), rmse_norm, color="red", marker="o")
     plt.title("Average RMSE against number of observations")
     plt.xlabel("Number of observations")
     plt.ylabel("RMSE")
     plt.legend()

     plt.show()

     
def averager(arrays):
     num_arrays = len(arrays)
     array_len = len(arrays[0])
     sum_array = [0]*array_len

     for array in arrays:
          for i in range(array_len):
               sum_array[i] += array[i]
     avg_array = [total / num_arrays for total in sum_array]

     return avg_array

lhc_gpr()