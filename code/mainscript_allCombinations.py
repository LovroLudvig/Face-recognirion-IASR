from NeuralNetwork import NeuralNetwork
from featureExtractor import FeatureExtractor
import numpy as np
from DataLoader import DataLoader
import configure

#best results:
#(0.05, 60, 30, 30)
#0.5428571428571428
#(0.1, 50, 60, 30)
#0.5714285714285714
#(0.1, 300, 100, 30)
#0.6
fe = FeatureExtractor("generatedData/eigenfaces.csv", "generatedData/average_face.csv")
best = 0.0
for lr in range(4, 7, 1):
    for ne in range(380, 421, 5):
        for nhn in range(40, 91, 10):
            for nev in range(50, 91, 10):
                configure.setUpConfig(lr/100, ne, nhn, nev)

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

                #classification:
                output_guess = []
                for sample in dataset:
                    desired = sample[-1]
                    calculated = nn.classify(np.transpose(np.asmatrix(sample[:-1])))
                    output_guess += [desired == calculated]

                #print configure and success rate if better
                correct_classifications = sum(output_guess)/np.shape(dataset)[0]
                if correct_classifications > best:
                    best = correct_classifications
                    best_tuple = (lr/100, ne, nhn, nev)
                    print(best_tuple)
                    print(best)
