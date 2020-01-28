from PIL import Image, ImageEnhance
import numpy as np
import os
import configure

class DataLoaderHelper:
    #saves an image
    @staticmethod
    def save_image(image, name):
        img = image * 255
        #img = np.expand_dims(image, axis=0) * 255
        Image.fromarray(img.reshape(64, 64).astype(np.uint8)).save("pictures/" + name + ".png")
        print("image " + name + ".png saved.")

    # function path_conj(pathname) takes pathname and takes all image names from pathname
    # directory, and joins them into one path
    # return list of all paths in pathname directory
    @staticmethod
    def path_conj(pathname):
        pathtrain = pathname
        filestrain = os.listdir(pathtrain)
        train_list_file = [os.path.join(pathtrain, f) for f in filestrain]
        file_list = train_list_file
        return file_list

    # function load_images(file_list) opens all images from file_list and
    # preprocesses them with contrast manipulation and normalization
    # from [0, 255] -> [0, 1]
    @staticmethod
    def load_images(file_list):
        return_list = []
        for file_name in file_list:
            with Image.open(file_name) as image:
                # image contrast enhancer
                im_enh = ImageEnhance.Contrast(image)
                image = im_enh.enhance(5.0)
                #im_sharp = ImageEnhance.Sharpness(image)
                #image = im_sharp.enhance(5.0)
                return_list.append(np.array(image)/255)
        return return_list

    # function getImagePaths(mode) takes argument mode = {training, testing}
    # returns list of paths to faces directories
    @staticmethod
    def getImagePaths(mode):
        return ["../lfwcrop_grey/dataset/" + mode + "/face1/",
            "../lfwcrop_grey/dataset/" + mode + "/face2/",
            "../lfwcrop_grey/dataset/" + mode + "/face3/",
            "../lfwcrop_grey/dataset/" + mode + "/face4/",
            "../lfwcrop_grey/dataset/" + mode + "/face5/"]
           # "../lfwcrop_grey/dataset/" + mode + "/unknown_faces/",
            #"../lfwcrop_grey/dataset/" + mode + "/resized background 64x64/"]

class DataLoader:
    def __init__(self, mode):
                     # 0   1   2   3   4  u    b
        self.images = [[], [], [], [], []]# [], []]
        self.all_faces = [] #0 1 2 3 4 lica
        self.mode = mode

    def load_all_images(self):
        paths = DataLoaderHelper.getImagePaths(self.mode)
        for i in range(len(paths)):
            file_list = DataLoaderHelper.path_conj(paths[i])
            self.images[i] = DataLoaderHelper.load_images(file_list)

        self.all_faces = np.concatenate((self.images[0], self.images[1], self.images[2], self.images[3], self.images[4]))# self.images[5]))


if __name__ == "__main__":
    pass
