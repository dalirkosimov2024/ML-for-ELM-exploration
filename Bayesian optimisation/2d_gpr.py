# Imports
import numpy as np
import os
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split

basepath ="/home/lcv10/Documents/ML_for_ELMs/Logbook/Pictures"

# Values for the contour plot
a = 2.21; h = 1; k = -10
test_size = 0.5
rmse_list  = np.array([])
# number of points
nv = 10
nv_squared = nv**2

xv = np.linspace(0, 10,nv )  # x-vector
yv = np.linspace(0, 10, nv)  # y-vector
x, y = np.meshgrid(xv, yv)  # x and y are (nv, nv) matrices

# equation for the toy curve
z = (x*np.cos(a)+y*np.sin(a)-h)**2 + k + x*np.sin(a) - y*np.cos(a)

# another set of parameters for plotting
xv1 = np.linspace(0, 10,10 )  
yv1 = np.linspace(0, 10, 10)  
x1, y1 = np.meshgrid(xv1, yv1)
z1 = (x1*np.cos(a)+y1*np.sin(a)-h)**2 + k + x1*np.sin(a) - y1*np.cos(a)

# Add noise, here we choose 0
noise_level = 0
z_noisy = z + np.random.normal(size=x.shape) * noise_level

# Convert to column vectors for scikit-learn
X = np.column_stack((x.reshape(-1), y.reshape(-1)))
Z = z_noisy.reshape(-1, 1)


kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))

# Fit to 10% of the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Z, test_size=test_size)
gpr = GaussianProcessRegressor(kernel, normalize_y=True)
gpr.fit(X_train, Y_train)

print(gpr.kernel_)
# RBF(length_scale=[7.86, 3.71]) + WhiteKernel(noise_level=0.00678)

# Predict and reshape back to grid for plotting
Zfit, Zstd = gpr.predict(X, return_std=True)
zstd = Zstd.reshape(x.shape)
zfit = Zfit.reshape(x.shape)
print(zstd)

mean_prediction, std_prediction = gpr.predict(X, return_std=True)

mse = mean_squared_error(z, zfit)
rmse = np.sqrt(mse)
print(rmse)
rmse_list = np.append(rmse_list, rmse)

if len(rmse_list) == 19:
    np.savetxt(f"{test_size}.csv", rmse_list)

# Plot posterior mean

# Set up figure
fig, ax = plt.subplots(figsize=(6.,4.))
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$j_{\parallel}$", rotation=0)
ax.set_xlim((0, 10))
ax.set_ylim((0, 10))
ax.set_aspect("equal")
ax.set_xticks((0, 0, 10))
ax.set_yticks((0, 0, 10))
ax.grid(False)



# Do the plotting
lev = np.linspace(0.0, 250.0, 6)
ax.contour(x1, y1, z1, lev, colors="grey")  # Truth
ax.contour(x, y, z, lev, colors="none")  # Truth
ax.plot(*X_train.T, "x", color="blue")  # Training samples
ax.plot(*X_test.T, "o", color="orange")
ax.contour(x, y, zfit, lev, colors="red", linestyles="dashed")  # Posterior mean

# Legend
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
# Write out
plt.tight_layout()

#plt.savefig(f"toy_gpr_{nv**2}_points.png")
plt.show()

# Plot posterior standard deviation, not sure what this means though

# Set up figure
fig, ax = plt.subplots(figsize=(6.,4.))
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_xlim((0, 10))
ax.set_ylim((0, 10))
ax.set_aspect("equal")
ax.set_xticks((0, 0, 10))
ax.set_yticks((0, 0, 10))
ax.grid(False)

# Do the plotting
ax.plot(*X_train.T, "o", color="C1")  # Training samples

lev = np.linspace(0, 5,20)
hc = ax.contourf(x, y, zstd, lev)  # Posterior std
for hci in hc.collections:
    hci.set_edgecolor("face")

# Colorbar
hcb = plt.colorbar(hc)
hcb.ax.grid(False)
hcb.set_label("Posterior standard deviation")

# Write out
plt.tight_layout()
plt.show()

"""
plt.plot(nv_list_squared, rmse_list, "--", alpha=0.4, color="blue")
plt.plot(nv_list_squared, rmse_list, "o", color="orange")
plt.xlabel("Number of uniform points plotted")
plt.ylabel("Root mean squared error")
plt.suptitle("RMSE against increasing spatial points for a 10% train-data case")
plt.show()
"""
