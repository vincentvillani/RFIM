import numpy
import matplotlib.pyplot as plt

a = numpy.loadtxt("signal.txt")
plt.imshow(a, interpolation="nearest")
plt.colorbar()
plt.show()