import numpy as np
import configure
from PIL import Image
from matplotlib import pyplot as plt
from numpy import linalg as LA

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
    faces_matrix = np.vstack([face_matrix.flatten() for face_matrix in list_of_face_matrices]) # flatten 2d image and stack into matrix
    average_face = np.sum(faces_matrix, axis = 0)/np.size(faces_matrix, axis = 0)
    diffrences_matrix = np.vstack([face - average_face for face in faces_matrix])

    average_face2 = np.expand_dims(average_face, axis=0)
    newpic = average_face2*255
    newpic = newpic.reshape(64,64).astype(np.uint8)
    Image.fromarray(newpic).save("pictures/avg.png")
    print("average face saved.")

    covarience_matrix = np.cov(diffrences_matrix.T)
    eigen_val, eigen_vecs = LA.eigh(covarience_matrix)

    for i in range (1,20):
        img = np.expand_dims(eigen_vecs[:,-i], axis=0)
        img = (img+np.amin(eigen_vecs))*255
        Image.fromarray(img.reshape(64,64).astype(np.uint8)).save("pictures/eigen"+str(i)+".png")
        print("eigenpic saved.")

    eigentrain = eigen_vecs
    return eigentrain, average_face

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
