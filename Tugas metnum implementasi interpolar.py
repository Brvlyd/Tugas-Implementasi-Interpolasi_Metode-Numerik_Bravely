import numpy as np
import matplotlib.pyplot as plt

# Data tegangan (x) dan waktu patah (y)
x_data = np.array([5, 10, 15, 20, 25, 30, 35, 40])
y_data = np.array([40, 30, 25, 40, 18, 20, 22, 15])

# Fungsi untuk interpolasi Polinom Lagrange
def lagrange_interpolation(x_data, y_data, x):
    def L(k, x):
        term = 1
        for i in range(len(x_data)):
            if i != k:
                term *= (x - x_data[i]) / (x_data[k] - x_data[i])
        return term
    
    P = 0
    for k in range(len(x_data)):
        P += y_data[k] * L(k, x)
    return P

# Fungsi untuk interpolasi Polinom Newton
def newton_interpolation(x_data, y_data, x):
    n = len(x_data)
    divided_diff = np.zeros((n, n))
    divided_diff[:,0] = y_data
    
    for j in range(1, n):
        for i in range(n-j):
            divided_diff[i][j] = (divided_diff[i+1][j-1] - divided_diff[i][j-1]) / (x_data[i+j] - x_data[i])
    
    def N(x):
        result = divided_diff[0,0]
        product_term = 1
        for i in range(1, n):
            product_term *= (x - x_data[i-1])
            result += divided_diff[0, i] * product_term
        return result
    
    return N(x)

# Plotting hasil interpolasi
x_range = np.linspace(5, 40, 400)
y_lagrange = [lagrange_interpolation(x_data, y_data, x) for x in x_range]
y_newton = [newton_interpolation(x_data, y_data, x) for x in x_range]

plt.plot(x_data, y_data, 'o', label='Data asli', color='orange')
plt.plot(x_range, y_lagrange, label='Interpolasi Lagrange', color='red')
plt.plot(x_range, y_newton, '--', label='Interpolasi Newton', color='blue')
plt.xlabel('Tegangan, x (kg/mm^2)')
plt.ylabel('Waktu patah, y (jam)')
plt.legend()
plt.title('Interpolasi dengan Polinom Lagrange dan Newton')
plt.grid(True)
plt.show()
