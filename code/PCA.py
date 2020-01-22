import numpy as np

class PCA():

    def __init__(self, number_of_components, list_of_face_matrices):
        print("PCA initialization started")
        faces_matrix = np.vstack([face_matrix.flatten() for face_matrix in list_of_face_matrices])
        self.average_face = np.sum(faces_matrix, axis = 0)/np.size(faces_matrix, axis = 0)
        diffrences_matrix = np.vstack([face - self.average_face for face in faces_matrix])
        covarience_matrix = diffrences_matrix.transpose() @ diffrences_matrix
        transform_matrix = PCA._generate_transform_matrix(number_of_components, covarience_matrix)
        self._transform_matrix = np.transpose(transform_matrix)
        print("PCA initialization done")

    @staticmethod
    def _generate_transform_matrix(number_of_components, covarience_matrix):
        #TODO: Remove complex values from eigenvalues and eigenvectors
        e_vals, e_vecs = np.linalg.eig(covarience_matrix)
        return np.real(e_vecs[0:number_of_components, :])

    # generates dataset with number_of_component features and class of image
    def generate_dataset(self, images):
        dataset = []
        for data_class in range(len(images)):
            class_images = images[data_class]
            for image in class_images:
                features = self.extract_features(image, transposed = False)
                dataset += [np.append(features, data_class)]
        return np.vstack(dataset)

    def extract_features(self, image_matrix, **kwargs):
        features = (image_matrix.flatten() - self.average_face) @ self._transform_matrix
        if kwargs['transposed'] == True:
            return np.transpose(features)
        return features

if __name__ == "__main__":
    print("No functionality yet")
    pass
