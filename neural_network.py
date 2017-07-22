import numpy
import scipy.special

class neural_network:

	# initialize the neural network
	def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
		self.innodes = inputNodes
		self.hnodes = hiddenNodes
		self.outnodes = outputNodes

		self.lr = learningRate

		#self.wih = (numpy.random.rand(self.hnodes, self.innodes) - 0.5)
		#self.who = (numpy.random.rand(self.outnodes, self.hnodes) - 0.5)

		self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.innodes)))
		self.who = (numpy.random.normal(0.0, pow(self.outnodes, -0.5), (self.outnodes, self.hnodes)))

		# define the activation function
		self.activationFunction = lambda x: scipy.special.expit(x)
		pass
	
	# train the neural network
	def train(self, inputList, targetList):
		# convert the lists into 2D array
		inputs = numpy.array(inputList, ndmin=2).T
		targets = numpy.array(targetList, ndmin=2).T

		hiddenInputs = numpy.dot(self.wih, inputs)
		hiddenOutputs = self.activationFunction(hiddenInputs)

		finalInputs = numpy.dot(self.who, hiddenOutputs)
		finalOutputs = self.activationFunction(finalInputs)

		#error
		outputErrors = targets - finalOutputs
		hiddenErrors = numpy.dot(self.who.T, outputErrors)

		#update the weights for the links between the hidden and output layers
		self.who += self.lr * numpy.dot((outputErrors * finalOutputs * (1 - finalOutputs)), numpy.transpose(hiddenOutputs))

		#update the weights for the links between the input and hidden layers
		self.wih += self.lr * numpy.dot((hiddenErrors * hiddenOutputs * (1 - hiddenOutputs)), numpy.transpose(inputs))
		pass

	# query the neural network
	def query(self, inputList):
		# convert the list into 2D array
		inputs = numpy.array(inputList, ndmin=2).T
		
		hiddenInputs = numpy.dot(self.wih, inputs)
		hiddenOutputs = self.activationFunction(hiddenInputs)

		finalInputs = numpy.dot(self.who, hiddenOutputs)
		finalOutputs = self.activationFunction(finalInputs)
		
		return finalOutputs
