import neural_network as nn
import numpy
import scipy.special
import matplotlib.pyplot

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.3

n = nn.neural_network(input_nodes, hidden_nodes, output_nodes, learning_rate)

trainingDataFile = open("mnist_train.csv", 'r')
trainingDataList = trainingDataFile.readlines()
trainingDataFile.close()

# train the neural network
# go through the records in training data set
for record in trainingDataList:
	allValues = record.split(',')
	#scale and shift input values
	inputs = (numpy.asfarray(allValues[1:])/ 255.0 * 0.99) + 0.01
	targets = numpy.zeros(output_nodes) + 0.01
	targets[int(allValues[0])] = 0.99
	n.train(inputs, targets)
	pass

imageArray = numpy.asfarray(allValues[1:]).reshape((28,28)) 
matplotlib.pyplot.imshow(imageArray, cmap="Greys", interpolation="None")



##################################################################################################
testDataFile = open("mnist_test.csv", 'r')
testDataList = testDataFile.readlines()
testDataFile.close()

testValues = testDataList[0].split(',')
print(testValues[0])

imageArray = numpy.asfarray(testValues[1:]).reshape((28,28)) 
matplotlib.pyplot.imshow(imageArray, cmap="Greys", interpolation="None")
matplotlib.pyplot.show()

n.query((numpy.asfarray(testValues[1:]) / 255.0 * 0.99) + 0.01)

##################################################################################################

# test the neural network
scorecard = [] #scorecard of neural network

for record in testDataList:
	allValues = record.split(',')
	correctLabel = int(allValues[0])
	# scale the inputs
	inputs = (numpy.asfarray(allValues[1:])/ 255.0 * 0.99) + 0.01
	# query the network
	outputs = n.query(inputs)
	# index of highest value corresponds to the label
	label = numpy.argmax(outputs)
	print("Correct Label : ", correctLabel, " network's answer : ", label)
	if(label == correctLabel):
		scorecard.append(1)
	else:
		scorecard.append(0)
		pass
