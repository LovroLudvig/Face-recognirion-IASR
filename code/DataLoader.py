from PIL import Image
import numpy as np
import os
from configure import Config

config = Config()

def main():

        face1 = []
        face2 = []
        face3 = []
        face4 = []
        face5 = []
        face = [face1, face2, face3, face4, face5]
        unknown_faces = []
        all_faces = []
        background = []
        paths = []

        if config.mode == "training":
            mode = "training"
        else:
            mode = "testing"

        print("Starting to work...")

        # defining the path and directory with images
        # Face 1 - 5
        paths.append("../lfwcrop_grey/izabrana_lica/" + mode + "/lice1/")
        paths.append("../lfwcrop_grey/izabrana_lica/" + mode + "/lice2/")
        paths.append("../lfwcrop_grey/izabrana_lica/" + mode + "/lice3/")
        paths.append("../lfwcrop_grey/izabrana_lica/" + mode + "/lice4/")
        paths.append("../lfwcrop_grey/izabrana_lica/" + mode + "/lice5/")

        for i in paths:
            print(i)
            file_list = path_conj(i)
            load_images(file_list, face[paths.index(i)])

        # Unknown faces
        pathtrain = "../lfwcrop_grey/izabrana_lica/" + mode + "/unknown_lica/"
        file_list = path_conj(pathtrain)
        load_images(file_list, unknown_faces)

        # Background
        pathtrain = "../lfwcrop_grey/izabrana_lica/" + mode + "/resized background 64x64/"
        file_list = path_conj(pathtrain)
        load_images(file_list, background)

        # All faces
        all_faces = np.concatenate((face1, face2, face3, face4, face5, unknown_faces))
        print(len(face1), len(unknown_faces), len(all_faces), len(background))

def path_conj(pathname):
    pathtrain = pathname
    filestrain = os.listdir(pathtrain)
    train_list_file = [os.path.join(pathtrain, f) for f in filestrain]
    file_list = train_list_file
    return file_list

def load_images(file_list, face_list):
    for file_name in file_list:
        with Image.open(file_name) as image:
            face_list.append(np.array(image) / 255)
    return face_list

main()