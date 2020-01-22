from NeuralNetwork import NeuralNetwork
from PCA import FeatureExtractor
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

    dl = DataLoader(config.modeTest)
    dl.load_all_images()
    dataset = fe.generate_dataset(dl.images)
    
    print(nn.classify(np.transpose(np.asmatrix(dataset[0][:-1]))))
    print(nn.classify(np.transpose(np.asmatrix(dataset[1][:-1]))))
    print(nn.classify(np.transpose(np.asmatrix(dataset[2][:-1]))))
    print(nn.classify(np.transpose(np.asmatrix(dataset[3][:-1]))))
    print(nn.classify(np.transpose(np.asmatrix(dataset[4][:-1]))))
    print(nn.classify(np.transpose(np.asmatrix(dataset[5][:-1]))))
