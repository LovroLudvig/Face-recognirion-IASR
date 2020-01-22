import numpy as np

class PCA():

    def __init__(self, number_of_components, list_of_face_matrices):
        diffrences_matrix = self._create_differences_matrix(list_of_face_matrices)
        covarience_matrix = self._create_covarience_matrix(diffrences_matrix)
        self._transform_matrix = np.transpose(self._generate_transform_matrix(number_of_components, covarience_matrix))

    def _create_differences_matrix(self, list_of_face_matrices):
        faces_matrix = self._combine_faces(list_of_face_matrices)
        average_face = self._create_average_face(faces_matrix)
        diffrences_matrix = [face - average_face for face in data_matrix]
        return diffrences_matrix

    def _create_covarience_matrix(self, diffrences_matrix):
        return diffrences_matrix.transpose() @ diffrences_matrix

    def _combine_faces(self, list_of_face_matrices):
        data_matrix = [face_matrix.flatten() for face_matrix in list_of_face_matrices]
        data_matrix = np.vstack(tuple(M))
        return data_matrix

    def _create_average_face(self, data_matrix):
        return np.sum(data_matrix, axis = 0)/np.size(C, axis = 0)

    def _generate_transform_matrix(self, number_of_components, covarience_matrix):
        e_vals, e_vecs = np.linalg.eig(covarience_matrix)
        return e_vecs[0:number_of_components, :] 

    def generate_dataset(images):
        dataset = []
        for data_class in range(len(images)):
            class_images = images[data_class]
            for image in class_images:
                features = self.extract_features(image, transposed = False)
                dataset += [np.append(features, data_class)]
        return np.vstack(tuple(dataset))

    def extract_features(self, image_matrix, **kwargs):
        features = image_matrix.flatten() @ self._transform_matrix
        if kwargs['transposed'] == True:
            return np.transpose(features)
        return features

if __name__ == "__main__":
    print("No functionality yet")