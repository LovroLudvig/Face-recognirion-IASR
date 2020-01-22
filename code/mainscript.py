from NeuralNetwork import NeuralNetwork
from featureExtractor import FeatureExtractor
import numpy as np
from DataLoader import DataLoader
import configure

best = 0.0
for lr in range(5, 21, 5):
    for ne in range(20, 301, 10):
        for nhn in range(10, 101, 10):
            for nev in range(10, 71, 10):
                configure.setUpConfig(lr/100, ne, nhn, nev)
                fe = FeatureExtractor("eigenfaces.csv", "average_face.csv")

                #prepare data for training:
                dl = DataLoader(configure.config_global.modeTrain)
                dl.load_all_images()
                datasetTrain = fe.generate_dataset(dl.images)

                #train NN:
                nn = NeuralNetwork(configure.config_global.noOfEigenValues, configure.config_global.noOfHidNeur)
                nn.trainNetwork(datasetTrain)

                #prepare data for testing:
                dl = DataLoader(configure.config_global.modeTest)
                dl.load_all_images()
                dataset = fe.generate_dataset(dl.images)

                output_guess = []
                for sample in dataset:
                    desired = sample[-1]
                    calculated = nn.classify(np.transpose(np.asmatrix(sample[:-1])))
                    output_guess += [desired == calculated]


                correct_classifications = sum(output_guess)/np.shape(dataset)[0]
                if correct_classifications > best:
                    best = correct_classifications
                    best_tuple = (lr/100, ne, nhn, nev)
                    print(best_tuple)
                    print(best)
