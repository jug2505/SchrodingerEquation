import numpy as np
import matplotlib.pyplot as plt

# Load the data from the text file
data = np.loadtxt('cmake-build-debug/solution_cuda.txt', usecols=(0, 1, 2))  # assuming your data is stored in 'your_file.txt'
X = data[:, 0]
Y = data[:, 1]
Z = data[:, 2]

# Reshape the data (assuming X and Y are regularly spaced)
X = np.unique(X)
Y = np.unique(Y)
X, Y = np.meshgrid(X, Y)
Z = Z.reshape(X.shape)

# Plot the contour map
plt.figure()
plt.contourf(X, Y, Z, cmap='viridis')  # Using 'viridis' colormap; you can choose any other colormap
plt.colorbar()  # Add a colorbar for reference
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Map of Z')

plt.savefig('foo.png')
