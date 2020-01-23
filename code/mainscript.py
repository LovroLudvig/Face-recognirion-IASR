from NeuralNetwork import NeuralNetwork
from featureExtractor import FeatureExtractor
import numpy as np
from DataLoader import DataLoader
import configure

#best results:
#(0.05, 60, 30, 30)
#0.5428571428571428
#(0.1, 50, 60, 30)
#0.5714285714285714
#(0.1, 300, 100, 30)
#0.6

configure.setUpConfig(0.1, 50, 60, 30)
fe = FeatureExtractor("eigenfaces.csv", "average_face.csv")

#prepare data for training:
dl = DataLoader(configure.config_global.modeTrain)
dl.load_all_images()
datasetTrain = fe.generate_dataset(dl.images)

#train NN:
nn = NeuralNetwork(configure.config_global.noOfEigenValues, configure.config_global.noOfHidNeur)
nn.trainNetwork(datasetTrain)

#prepare data for testing:
dl = DataLoader(configure.config_global.modeTest)
dl.load_all_images()
dataset = fe.generate_dataset(dl.images)

output_guess = []
for sample in dataset:
    desired = sample[-1]
    calculated = nn.classify(np.transpose(np.asmatrix(sample[:-1])))
    output_guess += [desired == calculated]


print(sum(output_guess)/np.shape(dataset)[0])

#prepare data for testing:
dl = DataLoader(configure.config_global.modeTrain)
dl.load_all_images()
dataset = fe.generate_dataset(dl.images)

output_guess = []
for sample in dataset:
    desired = sample[-1]
    calculated = nn.classify(np.transpose(np.asmatrix(sample[:-1])))
    output_guess += [desired == calculated]


print(sum(output_guess)/np.shape(dataset)[0])
