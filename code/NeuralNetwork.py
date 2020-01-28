import numpy as np
import configure

#sigmoid function calculator and derivative of sigmoid function calculator
class SigmoidFunction:
    @staticmethod
    def calcSig(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def calcSigDer(x):
        return SigmoidFunction.calcSig(x)*(1-SigmoidFunction.calcSig(x))

#output of layer calculator
class LayerOutputCalculator:
    def __init__(self, weights, input):
        self.outputNonActivated = weights*np.vstack((input, 1))
        self.outputActivated = SigmoidFunction.calcSig(self.outputNonActivated)

class NeuralNetwork:
    def __init__(self, inputLayer, hiddenLayer):
        #Xavier initialization
        self.inputLayer = inputLayer
        self.hiddenLayer = np.random.randn(hiddenLayer,inputLayer+1)*np.sqrt(1/inputLayer)
        self.outputLayer = np.random.randn(configure.config_global.noOfNNOutputs,hiddenLayer+1)*np.sqrt(1/hiddenLayer)

    #propagates input trought network and retuns class
    def classify(self, input):
        hiddenLayerOutput = LayerOutputCalculator(self.hiddenLayer, input)
        outputLayerOutput = LayerOutputCalculator(self.outputLayer, hiddenLayerOutput.outputActivated)
        return np.where(outputLayerOutput.outputActivated == max(outputLayerOutput.outputActivated))[0][0]

    #training the neural network with backpropagation
    def trainNetwork(self, trainingDataset):
        trainingDataset = np.asmatrix(trainingDataset)
        for i in range(configure.config_global.noOfEpochs):
            for j in trainingDataset:
                desiredClass = int(j[0,-1])
                nnInput = np.transpose(j[0,:-1])
                desiredOutput = np.matrix('0;'*(configure.config_global.noOfNNOutputs-1)+'0')
                desiredOutput[desiredClass] = 1

                hiddenLayerOutput = LayerOutputCalculator(self.hiddenLayer, nnInput)
                outputLayerOutput = LayerOutputCalculator(self.outputLayer, hiddenLayerOutput.outputActivated)

                d2 = np.multiply(np.multiply((desiredOutput-outputLayerOutput.outputActivated),(1-outputLayerOutput.outputActivated)),outputLayerOutput.outputActivated)
                dw2 = configure.config_global.learningRate * (d2 * np.transpose(np.vstack((hiddenLayerOutput.outputActivated, 1))))
                self.outputLayer += dw2

                d1 = np.multiply(np.multiply((np.transpose(self.outputLayer[:,:-1])* d2),(1-hiddenLayerOutput.outputActivated)),hiddenLayerOutput.outputActivated)
                dw1 = configure.config_global.learningRate * d1 * np.transpose(np.vstack((nnInput, 1)))
                self.hiddenLayer += dw1







if __name__ == "__main__":
    pass
