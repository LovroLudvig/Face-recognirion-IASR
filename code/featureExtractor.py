import numpy as np
import configure

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
    faces_matrix = np.vstack([face_matrix.flatten() for face_matrix in list_of_face_matrices])
    average_face = np.sum(faces_matrix, axis = 0)/np.size(faces_matrix, axis = 0)
    diffrences_matrix = np.vstack([face - average_face for face in faces_matrix])
    covarience_matrix = diffrences_matrix.transpose() @ diffrences_matrix
    transform_matrix = generate_transform_matrix(number_of_eigenfaces, covarience_matrix)
    return np.transpose(transform_matrix), average_face

def generate_transform_matrix(number_of_components, covarience_matrix):
    e_vals, e_vecs = np.linalg.eig(covarience_matrix)
    return np.real(e_vecs[0:number_of_components, :])

if __name__ == "__main__":
    from DataLoader import DataLoader
    configure.setUpConfig(0.05,20,10,500)
    dl = DataLoader(configure.config_global.modeTrain)
    dl.load_all_images()
    print("PCA init start")
    eigenfaces, average_face = extract_eigenfaces(configure.config_global.noOfEigenValues, dl.all_faces)
    print("PCA end")
    np.savetxt('eigenfaces.csv', eigenfaces, delimiter=',')
    np.savetxt('average_face.csv', average_face, delimiter=',')
