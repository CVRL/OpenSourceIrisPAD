"""
This file implements the manager class for extracting features, training, and testing iris PAD models.
"""
import filter
import numpy as np
import random
import h5py
import cv2 as cv
import os


class manager(object):

    def __init__(self):
        self.extractFeatures = False
        self.trainModel = False
        self.testImages = False
        self.majorityVoting = False
        self.segmentation = "wi"
        self.modelTypes = []
        self.imageDir = ""
        self.splitDir = ""
        self.trainFilename = ""
        self.testFilename = ""
        self.modelSizes = []
        self.bitSizes = []
        self.extractionFilename = ""
        self.extractionDir = ""
        self.modelDir = ""

    def load_config(self, filename):
        # open the file
        configFile = open(filename, 'r')

        # define dictionary to store and initialize values
        config = {
            # commands
            "Extract features": "no",
            "Train model": "no",
            "Test images": "no",
            "Majority voting": "no",
            "Segmentation": "wi",
            "Model type": "svm",
            # inputs
            "Image directory": "",
            "Split directory": "",
            "Training set filename": "",
            "Testing set filename": "",
            "Training sizes": "",
            "Bitsizes": "",
            # outputs
            "Extraction destination filename": "",
            "Extraction destination directory": "",
            "Model directory": ""
        }


        # Loop on lines
        for line in configFile:

            # ignore if empty
            if line.strip():
                # filter out comments
                pos = line.find('#')
                if not (pos == -1):
                    line = line[:pos]

            # ignore if empty now (if it was a comment)
            if line.strip():
                # find equal sign
                pos = line.find('=')
                if not (pos == -1):
                    # trim key and value
                    key = line[:pos].strip()
                    value = line[(pos+1):].strip()
                    # look in dictionary
                    if key in config:
                        config[key] = value
                    else:
                        print(key)
                        print("Incorrect option")
        # close the file
        configFile.close()

        # create variables to store information
        self.extractFeatures = (config["Extract features"].lower() == "yes")
        self.trainModel = (config["Train model"].lower() == "yes")
        self.testImages = (config["Test images"].lower() == "yes")
        self.majorityVoting = (config["Majority voting"].lower() == "yes")
        self.segmentation = config["Segmentation"]
        self.modelTypes = config["Model type"].split(',')
        self.imageDir = config["Image directory"]
        self.splitDir = config["Split directory"]
        self.trainFilename = config["Training set filename"]
        self.testFilename = config["Testing set filename"]
        self.modelSizes = [int(i) for i in config["Training sizes"].split(',')]
        self.bitSizes = [int(i) for i in config["Bitsizes"].split(',')]
        self.extractionFilename = config["Extraction destination filename"]
        self.extractionDir = config["Extraction destination directory"]
        self.modelDir = config["Model directory"]

    def show_config(self):
        print("====================")
        print("Configuration")
        print("====================")

        # process
        print("Process: ")
        if self.extractFeatures:
            print("- Extract features")
        if self.trainModel:
            print("-Train model")
        if self.testImages:
            print("-Test images")

        # Extraction parameters
        print("====================")
        if self.extractFeatures:
            print("- Features will be stored in directory: " + self.extractionDir)
            print("- Feature file names will be in format: " + self.extractionFilename + "_filtertype_size_size_bits.hdf5")
            print("- Feature sets: (bits, size)")
            for i in range(len(self.modelSizes)):
                print("     " + str(self.bitSizes[i]) + "," + str(self.modelSizes[i]))
            print("====================")
        if self.trainModel:
            print("- The training set will be: " + self.splitDir + self.trainFilename)
            print("- Models to be trained: ")
            for i in range(len(self.modelSizes)):
                print("     " + self.get_model_filename(i))
            print("====================")
        if self.testImages:
            if self.majorityVoting:
                print("- Majority voting will be used to determine result from multiple models")
                print("- In the case of a tie, a random decision will be made")
            else:
                print("- Models will be tested separately")
            print("- The testing set will be: " + self.splitDir + self.testFilename)
            print("- Models to be tested: ")
            for i in range(len(self.modelSizes)):
                print("     " + self.get_model_filename(i))
            print("====================")

    def run(self):
        print("====================")
        print("Start processing...")
        print("====================")

        # load the sets for testing and training
        self.load_sets()

        # extract features
        if (self.extractFeatures):
            # concatenate lists of files
            allImages = self.trainingSet + self.testingSet

            # extract features
            for i in range(len(self.bitSizes)):
                print("Extracting set " + str(i + 1) + " of " + str(len(self.bitSizes)) + "...")
                filter.extract(self.imageDir, allImages, self.extractionDir, self.extractionFilename, self.modelSizes[i], "bg", self.bitSizes[i])

        # train models

        if self.trainModel:
            # loop through all model sizes requested
            for i in range(len(self.modelSizes)):
                print("Training model " + str(i + 1) + " out of " + str(len(self.modelSizes)))
                print("     BSIF (bits,size) (" + str(self.bitSizes[i]) + "," + str(self.modelSizes[i]) + ") | " + self.modelTypes[i])

                # load data for current size
                features = self.load_features(self.bitSizes[i], self.modelSizes[i], self.trainingSet)

                # train model
                if self.modelTypes[i] == "svm":
                    # train a new model
                    model = cv.ml.SVM_create()
                    model.setType(cv.ml.SVM_C_SVC)
                    model.setKernel(cv.ml.SVM_RBF)
                    model.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

                    model.trainAuto(features, cv.ml.ROW_SAMPLE, self.trainingClass)
                    # save the model
                    model.save(self.modelDir + self.get_model_filename(i))

                elif self.modelTypes[i] == "rf":
                    model = cv.ml.RTrees_create()
                    model.train(features, cv.ml.ROW_SAMPLE, self.trainingClass)
                    model.save(self.modelDir + self.get_model_filename(i))

                elif self.modelTypes[i] == "mlp":
                    # need to change responses to (.8,-.8) and (-.8,.8)
                    nnResponses = np.zeros(shape=(len(self.trainingSet), 2), dtype=np.float32)
                    for j in range(len(self.trainingSet)):
                        if self.trainingClass[j] == 1:
                            #textured
                            nnResponses[i,0] = 0.8
                            nnResponses[i,1] = -0.8
                        elif self.trainingClass[j] == 0:
                            # clear or None
                            nnResponses[i,0] = -0.8
                            nnResponses[i,1] = 0.8

                    # define mlp parameters
                    layerSize = np.array([[features.shape[1]],[features.shape[1] * 2],[2]])
                    model = cv.ml.ANN_MLP_create()
                    model.setLayerSizes(layerSize)
                    model.setTrainMethod(cv.ml.ANN_MLP_BACKPROP)
                    model.setActivationFunction(cv.ml.ANN_MLP_SIGMOID_SYM,1,1)

                    # train mlp
                    model.train(features, cv.ml.ROW_SAMPLE, nnResponses)
                    model.save(self.modelDir + self.get_model_filename(i))

        if self.testImages:
            print("Testing images...")

            # check if majority voting or individual
            if self.majorityVoting:
                all_results = []
                for i in range(len(self.modelSizes)):
                    # load the model
                    exists = os.path.isfile(self.modelDir + self.get_model_filename(i)) # check that file exists
                    currentModel = None
                    if exists:
                        if self.modelTypes[i] == "svm":
                            currentModel = cv.ml.SVM_load(self.modelDir + self.get_model_filename(i))
                        elif self.modelTypes[i] == "rf":
                            currentModel = cv.ml.RTrees_load(self.modelDir + self.get_model_filename(i))
                        elif self.modelTypes[i] == "mlp":
                            currentModel = cv.ml.ANN_MLP_load(self.modelDir + self.get_model_filename(i))

                        # load the Features
                        features = self.load_features(self.bitSizes[i], self.modelSizes[i], self.testingSet)

                        # make predictions
                        if self.modelTypes[i] == "mlp":
                            # predict with mlp
                            predictions = currentModel.predict(features)[1]
                            binary_prediction = []
                            for k in range(len(predictions)):
                                if (predictions[k,0] > 0.8) and (predictions[k,1] < -0.8):
                                    binary_prediction.append(1)
                                elif (predictions[k,0] < -0.8) and (predictions[k,1] > 0.8):
                                    binary_prediction.append(0)
                            all_results.append(binary_prediction)
                        else:
                            # predict with svm or rf
                            predictions = currentModel.predict(features)[1]
                            # need to flatten this list
                            flat_prediction = [predict for sublist in predictions for predict in sublist]
                            all_results.append(flat_prediction)

                # perform majority voting
                overall_prediction = []
                for j in range(len(self.testingClass)):
                    # variables for majority decision
                    positive = 0
                    negative = 0

                    # loop through all all_results
                    for single_result in all_results:
                        if single_result[j] == 1:
                            positive += 1
                        else:
                            negative += 1

                    # vote
                    if (positive > negative):
                        overall_prediction.append(1)
                    elif (positive < negative):
                        overall_prediction.append(0)
                    else:
                        # assign random prediction
                        overall_prediction.append(random.randint(0,1))

                # determine accuracy
                total_incorrect = 0
                apcer = 0
                bpcer = 0
                num_bonafide = 0
                num_attack = 0

                for p in range(len(self.testingClass)):
                    if self.testingClass[p] == 1:
                        num_attack += 1
                    else:
                        num_bonafide += 1
                    if self.testingClass[p] != overall_prediction[p]:
                        total_incorrect += 1
                        if self.testingClass[p] == 1:
                            # attack presentation incorrectly identified as bona fide
                            apcer += 1
                        else:
                            # bona fide incorrectly identified as attack
                            bpcer += 1
                
                if num_bonafide == 0:
                    num_bonafide = 1
                if num_attack == 0:
                    num_attack = 1
                # calculate percentages
                ccr = 100.0 - float(total_incorrect)/float(len(self.testingSet)) * 100.0
                apcer = float(apcer)/float(num_attack) * 100.0
                bpcer = float(bpcer)/float(num_bonafide) * 100.0

                # print results
                print("Number of Models: " + str(len(self.modelSizes)))
                print("     CCR: " + str(ccr))
                print("     APCER: " + str(apcer))
                print("     BPCER: " + str(bpcer))

            else:
                # use each model separately

                for i in range(len(self.modelSizes)):
                    # load the model
                    exists = os.path.isfile(self.modelDir + self.get_model_filename(i)) # check that file exists
                    
                    if exists:
                        if self.modelTypes[i] == "svm":
                            currentModel = cv.ml.SVM_load(self.modelDir + self.get_model_filename(i))
                        elif self.modelTypes[i] == "rf":
                            currentModel = cv.ml.RTrees_load(self.modelDir + self.get_model_filename(i))
                        elif self.modelTypes[i] == "mp":
                            currentModel = cv.ml.ANN_MLP_load(self.modelDir + self.get_model_filename(i))

                        # load the Features
                        features = self.load_features(self.bitSizes[i], self.modelSizes[i], self.testingSet)
                        # test models
                        total_incorrect = 0
                        apcer = 0
                        bpcer = 0
                        num_bonafide = 0
                        num_attack = 0
                        
                        if self.modelTypes[i] == "mp":
                            # predict with mlp
                            predictions = currentModel.predict(features)[1]
                            binary_prediction = None
                            for k in range(len(predictions)):
                                if self.testingClass[k] == 1:
                                    num_attack += 1
                                else:
                                    num_bonafide += 1
                                if (predictions[k,0] > 0.8) and (predictions[k,1] < -0.8):
                                    binary_prediction = 1
                                elif (predictions[k,0] < -0.8) and (predictions[k,1] > 0.8):
                                    binary_prediction = 0
                                if self.testingClass[k] != binary_prediction:
                                    total_incorrect += 1
                                    if self.testingClass[k] == 1:
                                        # attack presentation incorrectly classified as bona fide
                                        apcer += 1
                                    else:
                                        # bona fide presentation incorrectly classified as attack
                                        bpcer += 1
                        else:
                            predictions = currentModel.predict(features)[1]

                        for k in range(len(predictions)):
                            if self.testingClass[k] == 1:
                                num_attack += 1
                            else:
                                num_bonafide += 1
                            if (float(self.testingClass[k]) != predictions[k][0]):
                                total_incorrect += 1
                                if self.testingClass[k] == 1:
                                    # attack presentation incorrectly classified as bona fide
                                    apcer += 1
                                else:
                                    # bona fide incorrectly classified as attack
                                    bpcer += 1
                    
                        if num_bonafide == 0:
                            num_bonafide = 1
                        if num_attack == 0:
                            num_attack = 1
                        
                        ccr = 100.0 - float(total_incorrect)/float(len(self.testingSet)) * 100.0
                        apcer = float(apcer)/float(num_attack) * 100.0
                        bpcer = float(bpcer)/float(num_bonafide) * 100.0

                        # print results
                        print("Model: " + self.get_model_filename(i))
                        print("     CCR: " + str(ccr))
                        print("     APCER: " + str(apcer))
                        print("     BPCER: " + str(bpcer))




    def load_features(self, bitsize, filtersize, fileSet):
        # Load files
        filename = self.extractionDir + self.extractionFilename + "_filter_" + str(filtersize) + "_" + str(filtersize) + "_" + str(bitsize) + ".hdf5"
        feature_file = h5py.File(filename, "r+")

        # create array (rows = number of samples, cols = size of histogram)
        all_features = np.zeros(shape=(len(fileSet), (2**bitsize)), dtype=np.float32)

        # loop through desired image features
        for i in range(len(fileSet)):
            name = fileSet[i]
            if name in feature_file:
                histogram = feature_file[name]
            
                # normalize
                mean = np.mean(histogram)
                stddev = np.std(histogram)
                histogram = (histogram - mean)/stddev
                # add to output
                all_features[i] = histogram
            else:
                print("Features not found for bitsize " + str(bitsize) + ", filtersize " + str(filtersize))
        feature_file.close()
        # return loaded Features
        return all_features


    def load_sets(self):
        # initialize parameters to hold sets
        self.trainingSet = []
        self.trainingClass = []
        self.testingSet = []
        self.testingClass = []

        # open training file as long as it is specified
        if self.trainFilename:
            trainFile = open((self.splitDir + self.trainFilename), 'r')
            # loop through lines
            for line in trainFile:
                if not (len(line.split(',')) == 2):
                    print("Incorrectly formatted set file.")
                self.trainingSet.append(line.split(',')[0])
                self.trainingClass.append(int(line.split(',')[1].strip()))
            self.trainingClass = np.asarray(self.trainingClass)

        # open testing file as long as it is specified
        if self.testFilename:
            testFile = open((self.splitDir + self.testFilename), 'r')
            # loop through lines
            for line in testFile:
                if not (len(line.split(',')) == 2):
                    print("Incorrectly formatted set file.")
                self.testingSet.append(line.split(',')[0])
                self.testingClass.append(int(line.split(',')[1].strip()))
            self.testingClass = np.asarray(self.testingClass)

    def get_model_filename(self, i):
        return ("BSIF-" + str(self.bitSizes[i]) + "-" + str(self.modelSizes[i]) + "-" + self.modelTypes[i] + "-" + self.segmentation + ".xml")



if __name__ == "__main__":
    newManager = manager()

    newManager.load_config("config.ini")
    newManager.show_config()
    newManager.run()
