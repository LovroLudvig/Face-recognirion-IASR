import numpy as np
from configure import Config

class SigmoidFunction:
    @staticmethod
    def calcSig(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def calcSigDer(x):
        return SigmoidFunction.calcSig(x)*(1-SigmoidFunction.calcSig(x))

class LayerOutputCalculator:
    def __init__(self, weights, input):
        self.outputNonActivated = weights*input
        self.outputActivated = SigmoidFunction.calcSig(self.outputNonActivated)

class NeuralNetwork:
    def __init__(self, inputLayer, hiddenLayer):
        #Xavier initialization
        self.config = Config()
        self.inputLayer = inputLayer
        self.hiddenLayer = np.random.randn(hiddenLayer,inputLayer)*np.sqrt(1/inputLayer)
        self.outputLayer = np.random.randn(self.config.noOfNNOutputs,hiddenLayer)*np.sqrt(1/hiddenLayer)

    def classify(self, input):
        hiddenLayerOutput = LayerOutputCalculator(self.hiddenLayer, input)
        outputLayerOutput = LayerOutputCalculator(self.outputLayer, hiddenLayerOutput.outputActivated)
        return np.where(outputLayerOutput.outputActivated == max(outputLayerOutput.outputActivated))[0][0]

if __name__ == "__main__":
    print("Nothing to test")
