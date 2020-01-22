from NeuralNetwork import NeuralNetwork
from DataLoader import DataLoader
from PCA import PCA
from configure import Config


if __name__ == "__main__":
    dl = DataLoader(Config().mode)
    dl.load_all_images()