import numpy
import matplotlib.pyplot as plt


a = numpy.loadtxt("covarianceMatrix.txt")
plt.imshow(a, interpolation="nearest")
plt.colorbar()
plt.show()

#eigenvalues = numpy.loadtxt("eigenvalues.txt")
#plt.plot(eigenvalues, marker=".")
#plt.show()