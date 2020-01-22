class Config:
    def __init__(self, learningRate, noOfEpochs, noOfHidNeur, noOfEigenValues):
        self.modeTrain = "training"
        self.modeTest = "testing"
        self.noOfNNOutputs = 7
        self.learningRate = learningRate
        self.noOfEpochs = noOfEpochs
        self.noOfHidNeur = noOfHidNeur
        self.noOfEigenValues = noOfEigenValues

def setUpConfig(learningRate, noOfEpochs, noOfHidNeur, noOfEigenValues):
    global config_global
    config_global = Config(learningRate, noOfEpochs, noOfHidNeur, noOfEigenValues)
