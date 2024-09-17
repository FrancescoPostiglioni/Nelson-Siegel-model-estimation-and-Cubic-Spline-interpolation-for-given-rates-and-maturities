
# Nelson-Siegel class (1987) model fitting given seven couples of rates and maturities
# Francesco Postiglioni
# 16/09/2024
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

# Dataset
maturities = np.array([.5, 1, 2, 3, 5, 7, 10]) # in years
rates = np.array([.5, .75, 1, 1.25, 1.5, 1.75, 2]) # 100s of basis points

# Nelson-Siegel class like function
def nelson_siegel(t, beta0, beta1, beta2, tau):
    return beta0 + (np.exp(-(t)/tau)) * (beta1 + beta2 * (t/tau))

# Model fitting
popt, pcov = curve_fit(nelson_siegel, maturities, rates, p0=[.03, -.02, .02, 1])

# Parameters
beta0, beta1, beta2, tau = popt
print(f"Estimated parameters: beta0={beta0}, beta1={beta1}, beta2={beta2}, tau={tau}")

# Plot of fitted curve
plt.figure(figsize=(10, 6))
plt.scatter(maturities, rates, color = 'red', label = 'Reference rate data')
plt.plot(maturities, nelson_siegel(maturities, *popt), label ='Nelson-Siegel curve')
plt.xlabel("Maturity (years)")
plt.ylabel('Rate (%)')
plt.title('Nelson-Siegel fit given by a reference rate in different maturities')
plt.legend()
plt.grid(True)
plt.show()

# Cubic spline interpolation for 5 datapoints (3x2 joining conditions)
x = np.array([1, 3, 5, 7, 10])
y = np.array([.5, 1, 1.5, 1.75, 2])

# Spline
cs = CubicSpline(x, y)

# Plot
x_new = np.linspace(1, 10, 100)
y_new =cs(x_new)

plt.figure(figsize=(10,6))
plt.plot(x, y, 'o', label='Reference rate data')
plt.plot(x_new, y_new, '-', label='Cubic Spline')
plt.legend()
plt.xlabel('maturity (years)')
plt.ylabel('Rate (%)')
plt.title('Cubic interpolant spline (done by a bank)')
plt.show()

