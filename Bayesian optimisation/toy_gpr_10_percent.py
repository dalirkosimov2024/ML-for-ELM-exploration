# Imports
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split

# Values for the contour plot
a = 2.21
h = 1
k = -10

# number of points
nv = 5

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

# Set up the kernel: 
# Radial basis function with two length scales for smooth variations
# and white kernel to account for noise
guess_l = (2.0, 1.0)  # In general, x and y have different scales
bounds_l = ((1e-1, 100.0),) * 2  # Same bounds for x and y
guess_n = 6.0  # Amount of noise
bounds_n = (1e-20, 10.0)  # Bounds for noise
kernel = RBF(  # Kernel objects can simply be summed using +
    length_scale=guess_l, length_scale_bounds=bounds_l
) + WhiteKernel(noise_level=guess_n, noise_level_bounds=bounds_n)

# Fit to 10% of the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Z, test_size=0.9)
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
ax.plot(*X_train.T, "x", color="blue")  # Training samples
ax.plot(*X_test.T, "o", color="orange")
ax.contour(x, y, zfit, lev, colors="red", linestyles="dashed")  # Posterior mean

# Legend
truth_line = mlines.Line2D([], [], color="black", label="Stability boundary")
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
plt.show()
plt.savefig(f"toy_gpr_{nv**2}_points.png")

# Plot posterior standard deviation, not sure what this means though

# Set up figure
"""fig, ax = plt.subplots(figsize=(6.,4.))
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
lev = np.linspace(6.0, 16.0, 11)
hc = ax.contourf(x, y, zstd, lev)  # Posterior std
for hci in hc.collections:
    hci.set_edgecolor("face")

# Colorbar
hcb = plt.colorbar(hc)
hcb.ax.grid(False)
hcb.set_label("Posterior standard deviation")

# Write out
plt.tight_layout()
plt.show()"""
