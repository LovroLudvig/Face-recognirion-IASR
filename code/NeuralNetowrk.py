import numpy as np

class SigmoidFunction:
    @staticmethod
    def calcSig(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def calcSigDer(x):
        return SigmoidFunction.calcSig(x)*(1-SigmoidFunction.calcSig(x))

class PerceptronOutputCalculator:
    def __init__(self, weights, input):
        self.outputNonActivated = weights*input
        self.outputActivated = SigmoidFunction.calcSig(self.outputNonActivated)

class NeuralNetwork:
    def __init__(self, inputLayer, hiddenLayer):
        #Xavier initialization
        noOfOutputs = 7
        self.hiddenLayer = np.random.randn(hiddenLayer,inputLayer)*np.sqrt(1/inputLayer)
        self.outputLayer = np.random.randn(7,hiddenLayer)*np.sqrt(1/hiddenLayer)
