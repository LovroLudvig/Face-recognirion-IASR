from NeuralNetwork import NeuralNetwork
from DataLoader import DataLoader
from PCA import PCA
from configure import Config


if __name__ == "__main__":
    dl = DataLoader(Config().mode)
    dl.load_all_images()
    pca = PCA(20, dl.all_faces)
    print(pca.extract_features(dl.all_faces[0], transposed=False))
    print(pca.generate_dataset(dl.images)[0:10, :])
