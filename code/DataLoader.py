from PIL import Image
import numpy as np
import os
from configure import Config

class DataLoaderHelper:
    @staticmethod
    def path_conj(pathname):
        pathtrain = pathname
        filestrain = os.listdir(pathtrain)
        train_list_file = [os.path.join(pathtrain, f) for f in filestrain]
        file_list = train_list_file
        return file_list

    @staticmethod
    def load_images(file_list):
        return_list = []
        for file_name in file_list:
            with Image.open(file_name) as image:
                return_list.append(np.array(image) / 255)
        return return_list

    @staticmethod
    def getImagePaths(mode):
        return ["../lfwcrop_grey/izabrana_lica/" + mode + "/lice1/",
            "../lfwcrop_grey/izabrana_lica/" + mode + "/lice2/",
            "../lfwcrop_grey/izabrana_lica/" + mode + "/lice3/",
            "../lfwcrop_grey/izabrana_lica/" + mode + "/lice4/",
            "../lfwcrop_grey/izabrana_lica/" + mode + "/lice5/",
            "../lfwcrop_grey/izabrana_lica/" + mode + "/unknown_lica/",
            "../lfwcrop_grey/izabrana_lica/" + mode + "/resized background 64x64/"]

class DataLoader:
    def __init__(self, mode):
        self.images = [[], [], [], [], [], [], []]
        self.all_faces = []
        self.mode = mode

    def load_all_images(self):
        print("started loading...")
        paths = DataLoaderHelper.getImagePaths(self.mode)
        for i in range(len(paths)):
            file_list = DataLoaderHelper.path_conj(paths[i])
            self.images[i] = DataLoaderHelper.load_images(file_list)

        self.all_faces = np.concatenate((self.images[0], self.images[1], self.images[2], self.images[3], self.images[4], self.images[5]))
        print(len(self.images[0]), len(self.all_faces), len(self.images[6]))
        print("finished loading.")

#USE THIS TO TEST:
#dl = DataLoader(Config().mode)
#dl.load_all_images()
