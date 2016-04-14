import pylab

data = pylab.loadtxt("RFIMBefore.txt")


pylab.plot( data[:,0])
pylab.plot(data[:,1])

pylab.legend()
pylab.title("Before RFIM")
pylab.xlabel("Sample Number")
pylab.ylabel("Value")

pylab.show()


data = pylab.loadtxt("RFIMAfter.txt")


pylab.plot(data[:,0])
pylab.plot(data[:,1])

pylab.legend()
pylab.title("After RFIM")
pylab.xlabel("Sample Number")
pylab.ylabel("Value")

pylab.show()