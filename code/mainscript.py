from NeuralNetwork import NeuralNetwork
from DataLoader import DataLoader
from PCA import PCA
from configure import Config
import numpy as np


if __name__ == "__main__":
    dl = DataLoader(Config().modeTrain)
    dl.load_all_images()
    pca = PCA(20, dl.all_faces)
    print(pca.extract_features(dl.all_faces[0], transposed=False))

    dataset = pca.generate_dataset(dl.images)[:10, :]
    print(dataset)
    nn = NeuralNetwork(20, 50)
    print(nn.classify(np.transpose(dataset[0][:-1])))
