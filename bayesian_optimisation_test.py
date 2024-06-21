#imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm

# define functions
x = np.linspace(-3,3,1000)
m = 0.995

def gaussian_function(x):
    return np.exp(-x**2/0.2)

y = gaussian_function(x)

# split the points to testing and training
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=m)

# select RBF kernel
kernel = RBF(length_scale=1,length_scale_bounds=(1e-2,1e2))

# apply kernel and fit GPR over the trained samples
gpr = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=9)
gpr.fit(x_train.reshape(-1,1), y_train)
gpr.kernel_

# retreive mu (mean, where the function should be) and sigma (standard deviation, how much the function deviates)
mean_prediction, std_prediction = gpr.predict(x.reshape(-1,1), return_std=True)

# plotting
def plotting():
    plt.plot(x,y, color="grey", alpha=0.5, label="Gaussian function")
    plt.ylim(0,1.1)
    plt.scatter(x_train,y_train, marker = "x", color = "blue", label="Trained points")
    plt.plot(x, mean_prediction, label="Posterior mean", linestyle="--", color="red")
    plt.fill_between(
        x.ravel(),
        mean_prediction - 1.95 * std_prediction,
        mean_prediction + 1.95 * std_prediction,
        alpha=0.3,
        label=r"95% confidence interval",
        color="orange"
    )
    plt.legend()
    plt.xlabel(r"Triangularity, $\delta$")
    plt.ylabel(r"Growth rates, $\gamma$")
    plt.suptitle("GPR fit over a Gaussian")
    plt.show()

# expected improvement function
def expected_improvement_func():
    def expected_improvement(x, gpr, best_y):
        mean_prediction, std_prediction = gpr.predict(x.reshape(-1,1), return_std=True)
        z = ( mean_prediction - best_y ) / std_prediction
        ei  = ( mean_prediction - best_y ) * norm.cdf(z) + mean_prediction*norm.pdf(z)
        return ei
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=m)
    best_idx = np.argmax(y_train) # find the maximum value in this array, which is used as the next point
    best_x = x_train[best_idx]
    best_y = y_train[best_idx]
    ei = expected_improvement(x, gpr, best_y)

    # plotting
    plt.plot(x, ei) 
    plt.axvline(x[np.argmax(ei)], linestyle="--", color="red", label="Next point location")
    plt.xlabel(r"Triangularity, $\delta$")
    plt.ylabel("Expected improvement")
    plt.suptitle("Expected improvement function across triangularity")
    plt.show()

    # iterate
    num_iter = 10

    for i in range(num_iter):

        # for each new point, retrain the model and get new posterior mean and standard deviations
        gpr.fit(x_train.reshape(-1,1), y_train)
        mean_prediction, std_prediction = gpr.predict(x.reshape(-1,1), return_std=True)

        # again attain the highest element in the array
        best_idx = np.argmax(y_train)
        best_x = x_train[best_idx]
        best_y = y_train[best_idx]
        ei = expected_improvement(x, gpr, best_y)

        # plot
        plt.plot(x,y, color="grey", alpha=0.5, label="Gaussian function")
        plt.plot(x, ei, linestyle="--", color="green", alpha=0.4,label="Surrogate function")
        plt.scatter(x_train, y_train, color="blue", marker="x", label="Previous Points")
        plt.ylim(-0.2,1.1)

        # plotting, to make it clear what is happening with each iteration
        if i < num_iter - 1:
            new_x = x[np.argmax(ei)]
            new_y = gaussian_function(new_x)
            x_train = np.append(x_train, new_x)
            y_train = np.append(y_train, new_y)
            plt.plot(x, mean_prediction, label="Posterior mean", linestyle="--", color="red")
            plt.scatter(new_x, new_y, color = "green", marker = "o", label ="New Points")
            plt.fill_between(
            x.ravel(),
            mean_prediction - 1.95 * std_prediction,
            mean_prediction + 1.95 * std_prediction,
            alpha=0.2,
            label=r"95% confidence interval",
            color="orange")
        plt.suptitle("Bayesian optimisation with 'expected improvement' aquisition function")
        plt.xlabel(r"Triangularity, $\delta$")
        plt.ylabel(r"Growth rates, $\gamma$")
        plt.legend()
        plt.show()


# upper confidence bound function
def upper_confidence_bound_func():
    def upper_confidence_bound(x, gpr, beta):
        mean_prediction, std_prediction = gpr.predict(x.reshape(-1,1), return_std=True)
        ucb = mean_prediction + beta * std_prediction
        return ucb
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=m)
    beta = 2
    ucb = upper_confidence_bound(x, gpr, beta)

    best_idx = np.argmax(y_train) # find the maximum value in this array, which is used as the next point
    best_x = x_train[best_idx]
    best_y = y_train[best_idx]

    ucb = upper_confidence_bound(x, gpr, best_y)
    plt.plot(x, ucb) 
    plt.axvline(x[np.argmax(ucb)], linestyle="--", color="red", label="Next point location")
    plt.xlabel(r"Triangularity, $\delta$")
    plt.ylabel("Upper confidence bound")
    plt.suptitle("Upper confidence bound function across triangularity")
    plt.show()
   

    # iterate
    num_iter = 10

    for i in range(num_iter):

        # for each new point, retrain the model and get new posterior mean and standard deviations
        gpr.fit(x_train.reshape(-1,1), y_train)
        mean_prediction, std_prediction = gpr.predict(x.reshape(-1,1), return_std=True)

        # again attain the highest element in the array
        best_idx = np.argmax(y_train)
        best_x = x_train[best_idx]
        best_y = y_train[best_idx]
        ucb = upper_confidence_bound(x, gpr, beta)
        
        # plotting
        plt.plot(x,y, color="grey", alpha=0.5, label="Gaussian function")
        plt.plot(x, ucb, linestyle="--", color="green", alpha=0.4,label="Surrogate function")
        plt.scatter(x_train, y_train, color="blue", marker="x", label="Previous Points")
        plt.ylim(-0.2,1.1)

        # plotting, to make it clear what is happening with each iteration
        if i < num_iter - 1:
            new_x = x[np.argmax(ucb)]
            new_y = gaussian_function(new_x)
            x_train = np.append(x_train, new_x)
            y_train = np.append(y_train, new_y)
            plt.plot(x, mean_prediction, label="Posterior mean", linestyle="--", color="red")
            plt.scatter(new_x, new_y, color = "green", marker = "o", label ="New Points")
            plt.fill_between(
            x.ravel(),
            mean_prediction - 1.95 * std_prediction,
            mean_prediction + 1.95 * std_prediction,
            alpha=0.2,
            label=r"95% confidence interval",
            color="orange")
        plt.suptitle("Bayesian optimisation with 'upper confidence bound' aquisition function")
        plt.xlabel(r"Triangularity, $\delta$")
        plt.ylabel(r"Growth rates, $\gamma$")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    while True:

        action = input("Plot initial? (y/n): ")
        if action =="y":
             plotting()
             pass
        else:
             pass

        action = input("Which function? (ei/ucb): ")
        if action == "ei":
                expected_improvement_func()
                action2 = input("End? (y/n): ")
                if action2 == "n":
                    pass
                else:
                     break
        elif action == "ucb": 
                upper_confidence_bound_func()
                action2 = input("End? (y/n): ")
                if action2 == "n":
                    pass
                else:
                     break
        else:
                print("Unkown input")
    

