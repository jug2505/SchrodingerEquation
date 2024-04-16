import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('cmake-build-debug/solution_cuda.txt', usecols=(0, 1, 2)) 
X = data[:, 0]
Y = data[:, 1]
Z = data[:, 2]

X = np.unique(X)
Y = np.unique(Y)
X, Y = np.meshgrid(X, Y)
Z = Z.reshape(X.shape)

plt.figure()
plt.contourf(X, Y, Z, cmap='viridis')  
plt.colorbar()  
plt.xlabel('$\\zeta$')
plt.ylabel('$\\tau$')
plt.title('$\\rho/|a_0|^2$')

plt.savefig('beam.png')
