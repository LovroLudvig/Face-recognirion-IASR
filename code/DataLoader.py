from PIL import Image
import numpy as np
import os

def main():

        raw_data = []
        print("Starting to work...")
        # defining the path and directory with images
        pathtrain = "../lfwcrop_grey/izabrana_lica/lice1/"
        filestrain = os.listdir(pathtrain)
        train_list_file = [os.path.join(pathtrain, f) for f in filestrain]
        file_list = train_list_file

        # loading the images into a list of matrices
        for file_name in file_list:
            with Image.open(file_name) as image:
                raw_data.append(np.array(image))

        print(raw_data[1])
        #return self.raw_data
main()