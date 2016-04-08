import numpy
import matplotlib.pyplot as plt

signal = numpy.loadtxt("RoundTripNoReductionCovarianceMatrix1.txt")
covarianceMatrix1 = numpy.cov(signal, bias=0, rowvar=1)
numpy.savetxt("RoundTripNoReductionCovarianceMatrix1Python.txt", covarianceMatrix1, fmt='%10.5f')

a = numpy.loadtxt("RoundTripNoReductionCovarianceMatrix1Python.txt")
plt.imshow(a, interpolation="nearest")
plt.colorbar()
plt.show()