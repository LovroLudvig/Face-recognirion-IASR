from NeuralNetwork import NeuralNetwork
from PCA import PCA
from configure import Config
import numpy as np
from DataLoader import DataLoader


if __name__ == "__main__":
    config = Config()
    dl = DataLoader(config.modeTrain)
    dl.load_all_images()
    pca = PCA(config.noOfEigenValues, dl.all_faces)
    dataset = pca.generate_dataset(dl.images)
    nn = NeuralNetwork(config.noOfEigenValues, config.noOfHidNeur)
    nn.trainNetwork(dataset)

    dl = DataLoader(config.modeTest)
    dl.load_all_images()
    dataset = pca.generate_dataset(dl.images)
    
    print(nn.classify(np.transpose(np.asmatrix(dataset[0][:-1]))))
    print(nn.classify(np.transpose(np.asmatrix(dataset[1][:-1]))))
    print(nn.classify(np.transpose(np.asmatrix(dataset[2][:-1]))))
    print(nn.classify(np.transpose(np.asmatrix(dataset[3][:-1]))))
    print(nn.classify(np.transpose(np.asmatrix(dataset[4][:-1]))))
    print(nn.classify(np.transpose(np.asmatrix(dataset[5][:-1]))))
