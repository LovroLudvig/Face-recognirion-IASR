import numpy as np

class PCA():

    def __init__(self, number_of_components, list_of_face_matrices):
        diffrences_matrix = _create_differences_matrix(list_of_face_matrices)
        covarience_matrix = diffrences_matrix.transpose() @ diffrences_matrix
        transform_matrix = _generate_transform_matrix(number_of_components, covarience_matrix)
        self._transform_matrix = np.transpose(transform_matrix)

    @staticmethod
    def _create_differences_matrix(list_of_face_matrices):
        faces_matrix = np.vstack([face_matrix.flatten() for face_matrix in list_of_face_matrices])
        average_face = np.sum(faces_matrix, axis = 0)/np.size(faces_matrix, axis = 0)
        return [face - average_face for face in faces_matrix]

    @staticmethod
    def _generate_transform_matrix(number_of_components, covarience_matrix):
        #TODO: Remove complex values from eigenvalues and eigenvectors
        e_vals, e_vecs = np.linalg.eig(covarience_matrix)
        return e_vecs[0:number_of_components, :]

    def generate_dataset(self, images):
        dataset = []
        for data_class in range(len(images)):
            class_images = images[data_class]
            for image in class_images:
                features = self.extract_features(image, transposed = False)
                dataset += [np.append(features, data_class)]
        return np.vstack(dataset)

    def extract_features(self, image_matrix, **kwargs):
        features = image_matrix.flatten() @ self._transform_matrix
        if kwargs['transposed'] == True:
            return np.transpose(features)
        return features

if __name__ == "__main__":
    print("No functionality yet")
