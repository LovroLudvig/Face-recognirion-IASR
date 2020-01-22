from PIL import Image
import numpy as np
import os

def main():

        face1 = []
        face2 = []
        face3 = []
        face4 = []
        face5= []
        unknown_faces = []
        all_faces = []
        background = []

        print("Starting to work...")

        # defining the path and directory with images
        # Face 1
        pathtrain = "../lfwcrop_grey/izabrana_lica/lice1/"
        filestrain = os.listdir(pathtrain)
        train_list_file = [os.path.join(pathtrain, f) for f in filestrain]
        file_list = train_list_file

        # loading the images into a list of matrices
        for file_name in file_list:
            with Image.open(file_name) as image:
                face1.append(np.array(image)/255)

        # Face 2
        pathtrain = "../lfwcrop_grey/izabrana_lica/lice2/"
        filestrain = os.listdir(pathtrain)
        train_list_file = [os.path.join(pathtrain, f) for f in filestrain]
        file_list = train_list_file

        # loading the images into a list of matrices
        for file_name in file_list:
            with Image.open(file_name) as image:
                face2.append(np.array(image) / 255)

        #Face 3
        pathtrain = "../lfwcrop_grey/izabrana_lica/lice3/"
        filestrain = os.listdir(pathtrain)
        train_list_file = [os.path.join(pathtrain, f) for f in filestrain]
        file_list = train_list_file

        # loading the images into a list of matrices
        for file_name in file_list:
            with Image.open(file_name) as image:
                face3.append(np.array(image) / 255)
        #Face 4
        pathtrain = "../lfwcrop_grey/izabrana_lica/lice4/"
        filestrain = os.listdir(pathtrain)
        train_list_file = [os.path.join(pathtrain, f) for f in filestrain]
        file_list = train_list_file

        # loading the images into a list of matrices
        for file_name in file_list:
            with Image.open(file_name) as image:
                face4.append(np.array(image) / 255)

        # Face 5
        pathtrain = "../lfwcrop_grey/izabrana_lica/lice5/"
        filestrain = os.listdir(pathtrain)
        train_list_file = [os.path.join(pathtrain, f) for f in filestrain]
        file_list = train_list_file

        # loading the images into a list of matrices
        for file_name in file_list:
            with Image.open(file_name) as image:
                face5.append(np.array(image) / 255)

        # Unknown faces
        pathtrain = "../lfwcrop_grey/izabrana_lica/unknown_lica/"
        filestrain = os.listdir(pathtrain)
        train_list_file = [os.path.join(pathtrain, f) for f in filestrain]
        file_list = train_list_file

        # loading the images into a list of matrices
        for file_name in file_list:
            with Image.open(file_name) as image:
                unknown_faces.append(np.array(image) / 255)

        # Background
        pathtrain = "../lfwcrop_grey/izabrana_lica/resized background 64x64/"
        filestrain = os.listdir(pathtrain)
        train_list_file = [os.path.join(pathtrain, f) for f in filestrain]
        file_list = train_list_file

        # loading the images into a list of matrices
        for file_name in file_list:
            with Image.open(file_name) as image:
                background.append(np.array(image) / 255)

        # All faces
        all_faces = np.concatenate((face1, face2, face3, face4, face5))
        print(len(all_faces))
        print(len(background))
        #max_el = np.amax (raw_data)
        #min_el = np.amin(raw_data)
        #print("(max el, min el): ", max_el, min_el)
main()