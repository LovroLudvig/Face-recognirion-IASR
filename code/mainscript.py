from NeuralNetwork import NeuralNetwork
from featureExtractor import FeatureExtractor
from configure import Config
import numpy as np
from DataLoader import DataLoader


if __name__ == "__main__":
    config = Config()
    dl = DataLoader(config.modeTrain)
    dl.load_all_images()
    fe = FeatureExtractor("eigenfaces.csv", "average_face.csv")
    dataset = fe.generate_dataset(dl.images)
    nn = NeuralNetwork(config.noOfEigenValues, config.noOfHidNeur)
    nn.trainNetwork(dataset)

    output_guess = []
    for sample in dataset:
        desired = sample[-1]
        calculated = nn.classify(np.transpose(np.asmatrix(sample[:-1])))
        output_guess += [desired == calculated]
    print(sum(output_guess)/np.shape(dataset)[0])


    dl = DataLoader(config.modeTest)
    dl.load_all_images()
    dataset = fe.generate_dataset(dl.images)
    
    output_guess = []
    for sample in dataset:
        desired = sample[-1]
        calculated = nn.classify(np.transpose(np.asmatrix(sample[:-1])))
        output_guess += [desired == calculated]

    print(sum(output_guess)/np.shape(dataset)[0])
        


