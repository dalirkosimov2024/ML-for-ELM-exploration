import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
import matplotlib.lines as mlines
from sklearn.metrics import mean_squared_error 

# Global variables 
n_sample_list = np.arange(4,17)
n_sample_list_squared = n_sample_list**2
test_size = 0.2

# Function definition
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
    for n_samples in n_sample_list_squared:

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

        # Predict the function values at the grid points
        zfit, z_pred_std = gp.predict(grid_points, return_std=True)
        zfit = zfit.reshape(x_grid.shape)

        # Calculating RMSE
        mse = mean_squared_error(z, zfit)
        rmse = np.sqrt(mse)
        rmse_list = np.append(rmse_list, rmse)

        # Plotting the results
        """
        fig, ax = plt.subplots(figsize=(6.,4.))
        ax.contour(x_grid, y_grid, z,  [0], color="grey", alpha=0.4)
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
        plt.suptitle(f"Toy stability space with {n_samples} points with latin-hypercube sampling")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$j_{\parallel}$", rotation=0)
        ax.set_xticks((0, 0, 10))
        ax.set_yticks((0, 0, 10))
        ax.set_xlim(0,10)
        ax.set_ylim(0,10)
        plt.tight_layout()
        plt.show()
        #plt.savefig(f"/home/lcv10/Documents/ML_for_ELMs/Logbook/Pictures/lhc_{n_samples}_samples.png")
        """
    np.savetxt("lhc.csv", rmse_list)
        

# Function that iterates a GPR process over a uniform sampling distribution
def uniform_gpr():
    rmse_list  = np.array([])
    for nv in n_sample_list:

        # Define the space - also defines the uniform distribution plots
        xv = np.linspace(0, 10,nv )  
        yv = np.linspace(0, 10, nv)  
        x, y = np.meshgrid(xv, yv)  

        # Equation for the toy curve
        z = custom_function(x,y,a=2.21, h=1, k=-10)

        # Another set of parameters for plotting
        xv1 = np.linspace(0, 10,10 )  
        yv1 = np.linspace(0, 10, 10)  
        x1, y1 = np.meshgrid(xv1, yv1)
        z1 = custom_function(x1, y1, a=2.21, h=1, k=-10)

        # Convert to column vectors for scikit-learn
        X = np.column_stack((x.reshape(-1), y.reshape(-1)))
        Z = z.reshape(-1, 1)

        # Select a constant and RBF kernel
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))

        # Split the data into a training and testing set 
        X_train, X_test, Y_train, Y_test = train_test_split(X, Z, test_size=test_size)
        gpr = GaussianProcessRegressor(kernel, normalize_y=True)

        # Train the model
        gpr.fit(X_train, Y_train)

        # Predict and reshape back to grid for plotting
        Zfit, Zstd = gpr.predict(X, return_std=True)
        zstd = Zstd.reshape(x.shape)
        zfit = Zfit.reshape(x.shape)
       

        mean_prediction, std_prediction = gpr.predict(X, return_std=True)

        # Calculating the RMSE
        mse = mean_squared_error(z, zfit)
        rmse = np.sqrt(mse)
        rmse_list = np.append(rmse_list, rmse)

        # Plotting the results 
        """
        fig, ax = plt.subplots(figsize=(6.,4.))
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$j_{\parallel}$", rotation=0)
        ax.set_xlim((0, 10))
        ax.set_ylim((0, 10))
        ax.set_aspect("equal")
        ax.set_xticks((0, 0, 10))
        ax.set_yticks((0, 0, 10))
        ax.grid(False)
        lev = np.linspace(0.0, 250.0, 6)
        ax.contour(x1, y1, z1, lev, colors="grey")  # True function
        ax.contour(x, y, z, lev, colors="none")  
        ax.plot(*X_train.T, "x", color="blue")  # Trained
        ax.plot(*X_test.T, "o", color="orange") # Tested
        ax.contour(x, y, zfit, lev, colors="red", linestyles="dashed")  # Posterior mean
        truth_line = mlines.Line2D([], [], color="grey", label="Stability boundary")
        sample_line = mlines.Line2D(
            [], [], color="blue", marker="x", linestyle="none", label=f"Training points"
        )
        mean_line = mlines.Line2D([], [], color="red", linestyle="--", label="Posterior mean")
        testing_points = mlines.Line2D([], [], color="orange", marker="o",linestyle="none", label="Testing Points")
        ax.legend(
            handles=[truth_line, sample_line, mean_line, testing_points],
            bbox_to_anchor=(1, 1),
            loc="upper left",
        )
        plt.suptitle(f"Toy stability space with {nv**2} uniform points")
        plt.tight_layout()
        plt.show()
        """
    np.savetxt("uniform.csv",rmse_list)

def plotter():
    uniform = np.loadtxt("uniform.csv")
    lhc= np.loadtxt("lhc.csv")

    plt.plot(n_sample_list_squared, uniform, color="green", marker=".", linestyle = "none",label = "Uniform distribution")
    plt.plot(n_sample_list_squared, uniform, "--", alpha=0.4,color="green")
    plt.plot(n_sample_list_squared, lhc, color="red", marker=".", linestyle = "none",label = "LHC distribution")
    plt.plot(n_sample_list_squared, lhc ,"--", alpha=0.4, color="red")
    #plt.plot(nv_list, c, color="blue",marker="o",linestyle = "none", label = "Train size = 70%")
    #plt.plot(nv_list, c, "--", alpha=0.4, color="blue")
    plt.legend()
    plt.xlabel("Number of uniform points plotted")
    plt.ylabel("Root mean squared error")
    plt.suptitle(f"RMSE comparing LHC to uniform distribution, test size = {test_size}")
    plt.show()

