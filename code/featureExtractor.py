import numpy as np
import configure
from PIL import Image
from matplotlib import pyplot as plt
from numpy import linalg as LA

class FeatureExtractor():

    def __init__(self, eigenfaces_url, average_face_url):
        self.eigenfaces = np.loadtxt(eigenfaces_url, delimiter=',')[:,:configure.config_global.noOfEigenValues]
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
    print("shape of diffmat:", diffrences_matrix.shape)

    print("pic")
    print("shape of avg:", average_face.shape)
    average_face2 = np.expand_dims(average_face, axis=0)
    print("shape of avg after:", average_face2.shape)

    newpic = average_face2*255
    print("avgface2 min, ", np.amin(newpic))
    newpic = newpic.reshape(64,64).astype(np.uint8)
    Image.fromarray(newpic).save("./avg.png")
    print("pic saved.")

    #covarience_matrix = diffrences_matrix.transpose() @ diffrences_matrix
    covarience_matrix = np.cov(diffrences_matrix.T)
    print("cov shape:", covarience_matrix.shape)
    eigen_val, eigen_vecs = LA.eigh(covarience_matrix)
    #eigen_vecs, eigen_vals, _ = np.linalg.svd(covarience_matrix)
    print("eigen shep:", eigen_vecs.shape)

    for i in range (1,20):
        #eigen_vecs[:, -i] = np.expand_dims(eigen_vecs[:,-i], axis=0)
        img = np.expand_dims(eigen_vecs[:,-i], axis=0)
        img = (img+np.amin(eigen_vecs))*255
        #img = eigen_vecs[:,-i]*255
        Image.fromarray(img.reshape(64,64).astype(np.uint8)).save("./eigen"+str(i)+".png")
        print("eigenpic saved.")


    #transform_matrix = generate_transform_matrix(number_of_eigenfaces, covarience_matrix)
    #return np.transpose(transform_matrix), average_face
    #return (eigen_vecs[:,0:number_of_eigenfaces]), average_face
    eigentrain = eigen_vecs[:, -(configure.config_global.noOfEigenValues+1): -1]
    print(eigentrain)
    print("eigentrain shae:", eigentrain.shape)
    return eigentrain, average_face


def generate_transform_matrix(number_of_components, covarience_matrix):
    e_vecs, e_vals = np.linalg.eig(covarience_matrix)
    return np.real(e_vecs[0:number_of_components, :])

if __name__ == "__main__":
    from DataLoader import DataLoader
    configure.setUpConfig(0.05,1000,50,50)
    dl = DataLoader(configure.config_global.modeTrain)
    dl.load_all_images()
    print("PCA init start")
    eigenfaces, average_face = extract_eigenfaces(configure.config_global.noOfEigenValues, dl.all_faces)
    print("PCA end")
    np.savetxt('eigenfaces.csv', eigenfaces, delimiter=',')
    np.savetxt('average_face.csv', average_face, delimiter=',')
