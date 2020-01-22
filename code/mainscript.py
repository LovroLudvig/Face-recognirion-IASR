from NeuralNetwork import NeuralNetwork
from PCA import PCA
from configure import Config


if __name__ == "__main__":
    print("here")
    dl = DataLoader(Config().mode)
    print("here")
    dl.load_all_images()
    print("here")
    pca = PCA(20, dl.all_faces)
    print(pca.extract_features())
