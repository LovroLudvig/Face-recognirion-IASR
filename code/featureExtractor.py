import numpy as np
import configure
from PIL import Image
from matplotlib import pyplot as plt
from numpy import linalg as LA
from DataLoader import DataLoaderHelper

class FeatureExtractor():

    def __init__(self, eigenfaces_url, average_face_url):
        self.eigenfaces = np.loadtxt(eigenfaces_url, delimiter=',')[:, -(configure.config_global.noOfEigenValues+1): -1]
        self.average_face = np.loadtxt(average_face_url, delimiter=',')

    def generate_dataset(self, images):
        dataset = []
        for data_class in range(len(images)):
            class_images = images[data_class]
            for image in class_images:
                features = self.extract_features(image, transposed = False)
                dataset += [np.append(features, data_class)]
        return np.vstack(dataset)

    def extract_features(self, image_matrix, **kwargs):
        features = (image_matrix.flatten() - self.average_face) @ self.eigenfaces
        if kwargs['transposed'] == True:
            return np.transpose(features)
        return features

def extract_eigenfaces(number_of_eigenfaces, list_of_face_matrices):
    # flatten 2d image and stack into matrix
    faces_matrix = np.vstack([face_matrix.flatten() for face_matrix in list_of_face_matrices])
    # sum all images vertically and divide by num_of_images
    average_face = np.sum(faces_matrix, axis = 0)/np.size(faces_matrix, axis = 0)
    # subtract average_face from all faces in faces_matrix
    diffrences_matrix = np.vstack([face - average_face for face in faces_matrix])
    DataLoaderHelper.save_image(average_face, "avg_face")
    covarience_matrix = np.cov(diffrences_matrix.T)
    eigen_val, eigen_vecs = LA.eigh(covarience_matrix)
    for i in range (1,20):
        DataLoaderHelper.save_image(eigen_vecs[:,-i] + np.amin(eigen_vecs), "eigenface"+str(i))
    return eigen_vecs, average_face

if __name__ == "__main__":
    from DataLoader import DataLoader
    configure.setUpConfig(0.05,1000,50,50)
    dl = DataLoader(configure.config_global.modeTrain)
    dl.load_all_images()
    print("PCA init start")
    eigenfaces, average_face = extract_eigenfaces(configure.config_global.noOfEigenValues, dl.all_faces)
    print("PCA end")
    np.savetxt('generatedData/eigenfaces.csv', eigenfaces, delimiter=',')
    np.savetxt('generatedData/average_face.csv', average_face, delimiter=',')
