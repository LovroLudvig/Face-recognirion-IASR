import numpy as np

""" dont send background here """
def combine_faces(*list_of_face_matrices):
    data_matrix = [face_matrix.flatten() for face_matrix in list_of_face_matrices]
    data_matrix = np.vstack(tuple(M))
    return data_matrix

def create_average_face(data_matrix):
    return np.sum(data_matrix, axis = 0)/np.size(C, axis = 0)

def create_differences_matrix(*list_of_face_matrices):
    faces_matrix = combine_faces(list_of_face_matrices)
    average_face = create_average_face(faces_matrix)
    diffrences_matrix = [face-average_face for face in data_matrix]
    return diffrences_matrix

def create_covarience_matrix(diffrences_matrix):
    return diffrences_matrix.transpose() @ diffrences_matrix



if __name__ == "__main__":
    print("No functionality yet")