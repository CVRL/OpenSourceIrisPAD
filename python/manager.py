"""
This file implements the manager class for extracting features, training, and testing iris PAD models.
"""
import os
import random
from typing import List

import bsif_filter as bsif_filter
import cv2 as cv
import h5py
import numpy as np


class Manager(object):
    def __init__(self):
        self.extract_features = False
        self.train_model = False
        self.test_images = False
        self.majority_voting = False
        self.segmentation_type = "wi"
        self.model_types = []
        self.image_dir = ""
        self.split_dir = ""
        self.train_filename = ""
        self.test_filename = ""
        self.all_filter_sizes = []
        self.all_num_filters = []
        self.extraction_filename = ""
        self.extraction_dir = ""
        self.model_dir = ""

    def load_config(self, filename: str):
        # open the file
        assert os.path.isfile(filename), "Config file not found: '{}'".format(filename)
        with open(filename, "r") as configFile:

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
                "Model directory": "",
            }

            # Loop on lines
            for line in configFile:

                # ignore if empty
                if line.strip():
                    # filter out comments
                    pos = line.find("#")
                    if not (pos == -1):
                        line = line[:pos]

                # ignore if empty now (if it was a comment)
                if line.strip():
                    # find equal sign
                    pos = line.find("=")
                    if not (pos == -1):
                        # trim key and value
                        key = line[:pos].strip()
                        value = line[(pos + 1) :].strip()
                        # look in dictionary
                        if key in config:
                            config[key] = value
                        else:
                            print(key)
                            print("Incorrect option")

        # create variables to store information
        self.extract_features = config["Extract features"].lower() == "yes"
        self.train_model = config["Train model"].lower() == "yes"
        self.test_images = config["Test images"].lower() == "yes"
        self.majority_voting = config["Majority voting"].lower() == "yes"
        self.segmentation_type = config["Segmentation"]
        self.model_types = config["Model type"].split(",")
        self.image_dir = config["Image directory"]
        self.split_dir = config["Split directory"]
        self.train_filename = config["Training set filename"]
        self.test_filename = config["Testing set filename"]
        self.all_filter_sizes = [int(i) for i in config["Training sizes"].split(",")]
        self.all_num_filters = [int(i) for i in config["Bitsizes"].split(",")]
        self.extraction_filename = config["Extraction destination filename"]
        self.extraction_dir = config["Extraction destination directory"]
        self.model_dir = config["Model directory"]

    def show_config(self):
        print("====================")
        print("Configuration")
        print("====================")

        # process
        print("Process: ")
        if self.extract_features:
            print("- Extract features")
        if self.train_model:
            print("-Train model")
        if self.test_images:
            print("-Test images")

        # Extraction parameters
        print("====================")
        if self.extract_features:
            print("- Features will be stored in directory: " + self.extraction_dir)
            print(
                "- Feature file names will be in format: "
                + self.extraction_filename
                + "_filtertype_size_size_bits.hdf5"
            )
            print("- Feature sets: (bits, size)")
            for filter_size in range(len(self.all_filter_sizes)):
                print(
                    "     "
                    + str(self.all_num_filters[filter_size])
                    + ","
                    + str(self.all_filter_sizes[filter_size])
                )
            print("====================")
        if self.train_model:
            print("- The training set will be: " + self.split_dir + self.train_filename)
            print("- Models to be trained: ")
            for filter_size in range(len(self.all_filter_sizes)):
                print("     " + self.get_model_filename(filter_size))
            print("====================")
        if self.test_images:
            if self.majority_voting:
                print(
                    "- Majority voting will be used to determine result from multiple models"
                )
                print("- In the case of a tie, a random decision will be made")
            else:
                print("- Models will be tested separately")
            print("- The testing set will be: " + self.split_dir + self.test_filename)
            print("- Models to be tested: ")
            for filter_size in range(len(self.all_filter_sizes)):
                print("     " + self.get_model_filename(filter_size))
            print("====================")

    def run(self):
        print("====================")
        print("Start processing...")
        print("====================")

        # load the sets for testing and training
        self.load_sets()

        # extract features
        if self.extract_features:
            # concatenate lists of files
            all_images = self.training_set + self.testing_set

            # extract features
            for num_filters, filter_size in zip(
                self.all_num_filters, self.all_filter_sizes
            ):
                print(
                    "Extracting filter set "
                    + str(num_filters)
                    + " of "
                    + str(len(self.all_num_filters))
                    + "..."
                )
                bsif_filter.extract_and_store(
                    image_location=self.image_dir,
                    file_names=all_images,
                    filter_size=filter_size,
                    num_filters=num_filters,
                    out_filename=self.extraction_filename,
                    out_location=self.extraction_dir,
                    segmentation_type=self.segmentation_type,
                )

        # train models

        if self.train_model:
            # loop through all model sizes requested
            for current_filter_size in range(len(self.all_filter_sizes)):
                print(
                    "Training model "
                    + str(current_filter_size + 1)
                    + " out of "
                    + str(len(self.all_filter_sizes))
                )
                print(
                    "     BSIF (bits,size) ("
                    + str(self.all_num_filters[current_filter_size])
                    + ","
                    + str(self.all_filter_sizes[current_filter_size])
                    + ") | "
                    + self.model_types[current_filter_size]
                )

                # load data for current size
                features = self.load_features(
                    self.all_num_filters[current_filter_size],
                    self.all_filter_sizes[current_filter_size],
                    self.training_set,
                )

                # train model
                if self.model_types[current_filter_size] == "svm":
                    # train a new model
                    model = cv.ml.SVM_create()
                    model.setType(cv.ml.SVM_C_SVC)
                    model.setKernel(cv.ml.SVM_RBF)
                    model.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

                    model.trainAuto(features, cv.ml.ROW_SAMPLE, self.training_classes)
                    # save the model
                    model.save(
                        self.model_dir + self.get_model_filename(current_filter_size)
                    )

                elif self.model_types[current_filter_size] == "rf":
                    model = cv.ml.RTrees_create()
                    model.train(features, cv.ml.ROW_SAMPLE, self.training_classes)
                    model.save(
                        self.model_dir + self.get_model_filename(current_filter_size)
                    )

                elif self.model_types[current_filter_size] == "mlp":
                    # need to change responses to (.8,-.8) and (-.8,.8)
                    nnResponses = np.zeros(
                        shape=(len(self.training_set), 2), dtype=np.float32
                    )
                    for j in range(len(self.training_set)):
                        if self.training_classes[j] == 1:
                            # textured
                            nnResponses[current_filter_size, 0] = 0.8
                            nnResponses[current_filter_size, 1] = -0.8
                        elif self.training_classes[j] == 0:
                            # clear or None
                            nnResponses[current_filter_size, 0] = -0.8
                            nnResponses[current_filter_size, 1] = 0.8

                    # define mlp parameters
                    layerSize = np.array(
                        [[features.shape[1]], [features.shape[1] * 2], [2]]
                    )
                    model = cv.ml.ANN_MLP_create()
                    model.setLayerSizes(layerSize)
                    model.setTrainMethod(cv.ml.ANN_MLP_BACKPROP)
                    model.setActivationFunction(cv.ml.ANN_MLP_SIGMOID_SYM, 1, 1)

                    # train mlp
                    model.train(features, cv.ml.ROW_SAMPLE, nnResponses)
                    model.save(
                        self.model_dir + self.get_model_filename(current_filter_size)
                    )

        if self.test_images:
            print("Testing images...")

            # check if majority voting or individual
            if self.majority_voting:
                all_results = []
                for current_filter_size in range(len(self.all_filter_sizes)):
                    # load the model
                    exists = os.path.isfile(
                        self.model_dir + self.get_model_filename(current_filter_size)
                    )  # check that file exists
                    current_model = None
                    if exists:
                        if self.model_types[current_filter_size] == "svm":
                            current_model = cv.ml.SVM_load(
                                self.model_dir
                                + self.get_model_filename(current_filter_size)
                            )
                        elif self.model_types[current_filter_size] == "rf":
                            current_model = cv.ml.RTrees_load(
                                self.model_dir
                                + self.get_model_filename(current_filter_size)
                            )
                        elif self.model_types[current_filter_size] == "mlp":
                            current_model = cv.ml.ANN_MLP_load(
                                self.model_dir
                                + self.get_model_filename(current_filter_size)
                            )

                        # load the Features
                        features = self.load_features(
                            self.all_num_filters[current_filter_size],
                            self.all_filter_sizes[current_filter_size],
                            self.testing_set,
                        )

                        # make predictions
                        if self.model_types[current_filter_size] == "mlp":
                            # predict with mlp
                            predictions = current_model.predict(features)[1]
                            binary_prediction = []
                            for k in range(len(predictions)):
                                if (predictions[k, 0] > 0.8) and (
                                    predictions[k, 1] < -0.8
                                ):
                                    binary_prediction.append(1)
                                elif (predictions[k, 0] < -0.8) and (
                                    predictions[k, 1] > 0.8
                                ):
                                    binary_prediction.append(0)
                            all_results.append(binary_prediction)
                        else:
                            # predict with svm or rf
                            assert current_model is not None and hasattr(
                                current_model, "predict"
                            ), "Model is not loaded or does not have a predict() method."
                            predictions = current_model.predict(features)[1]
                            # need to flatten this list
                            flat_prediction = [
                                predict
                                for sublist in predictions
                                for predict in sublist
                            ]
                            all_results.append(flat_prediction)

                # perform majority voting
                overall_prediction = []
                for j in range(len(self.testing_classes)):
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
                    if positive > negative:
                        overall_prediction.append(1)
                    elif positive < negative:
                        overall_prediction.append(0)
                    else:
                        # assign random prediction
                        overall_prediction.append(random.randint(0, 1))

                # evaluate the results
                total_incorrect = 0
                apcer = 0
                bpcer = 0
                num_bonafide = 0
                num_attack = 0

                for p in range(len(self.testing_classes)):
                    if self.testing_classes[p] == 1:
                        num_attack += 1
                    else:
                        num_bonafide += 1
                    if self.testing_classes[p] != overall_prediction[p]:
                        total_incorrect += 1
                        if self.testing_classes[p] == 1:
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
                ccr = (
                    100.0
                    - float(total_incorrect) / float(len(self.testing_set)) * 100.0
                )
                apcer = float(apcer) / float(num_attack) * 100.0
                bpcer = float(bpcer) / float(num_bonafide) * 100.0

                # print results
                print("Number of Models: " + str(len(self.all_filter_sizes)))
                print("     CCR: " + str(ccr))
                print("     APCER: " + str(apcer))
                print("     BPCER: " + str(bpcer))

            else:
                # use each model separately

                for current_filter_size in range(len(self.all_filter_sizes)):
                    # load the model
                    exists = os.path.isfile(
                        self.model_dir + self.get_model_filename(current_filter_size)
                    )  # check that file exists

                    if exists:
                        if self.model_types[current_filter_size] == "svm":
                            current_model = cv.ml.SVM_load(
                                self.model_dir
                                + self.get_model_filename(current_filter_size)
                            )
                        elif self.model_types[current_filter_size] == "rf":
                            current_model = cv.ml.RTrees_load(
                                self.model_dir
                                + self.get_model_filename(current_filter_size)
                            )
                        elif self.model_types[current_filter_size] == "mp":
                            current_model = cv.ml.ANN_MLP_load(
                                self.model_dir
                                + self.get_model_filename(current_filter_size)
                            )
                        else:
                            raise Exception(
                                "Model type not supported: "
                                + self.model_types[current_filter_size]
                            )

                        # load the Features
                        features = self.load_features(
                            self.all_num_filters[current_filter_size],
                            self.all_filter_sizes[current_filter_size],
                            self.testing_set,
                        )
                        # test models
                        total_incorrect = 0
                        apcer = 0
                        bpcer = 0
                        num_bonafide = 0
                        num_attack = 0

                        if self.model_types[current_filter_size] == "mp":
                            # predict with mlp
                            predictions = current_model.predict(features)[1]
                            binary_prediction = None
                            for k in range(len(predictions)):
                                if self.testing_classes[k] == 1:
                                    num_attack += 1
                                else:
                                    num_bonafide += 1
                                if (predictions[k, 0] > 0.8) and (
                                    predictions[k, 1] < -0.8
                                ):
                                    binary_prediction = 1
                                elif (predictions[k, 0] < -0.8) and (
                                    predictions[k, 1] > 0.8
                                ):
                                    binary_prediction = 0
                                if self.testing_classes[k] != binary_prediction:
                                    total_incorrect += 1
                                    if self.testing_classes[k] == 1:
                                        # attack presentation incorrectly classified as bona fide
                                        apcer += 1
                                    else:
                                        # bona fide presentation incorrectly classified as attack
                                        bpcer += 1
                        else:
                            predictions = current_model.predict(features)[1]

                        for k in range(len(predictions)):
                            if self.testing_classes[k] == 1:
                                num_attack += 1
                            else:
                                num_bonafide += 1
                            if float(self.testing_classes[k]) != predictions[k][0]:
                                total_incorrect += 1
                                if self.testing_classes[k] == 1:
                                    # attack presentation incorrectly classified as bona fide
                                    apcer += 1
                                else:
                                    # bona fide incorrectly classified as attack
                                    bpcer += 1

                        if num_bonafide == 0:
                            num_bonafide = 1
                        if num_attack == 0:
                            num_attack = 1

                        ccr = (
                            100.0
                            - float(total_incorrect)
                            / float(len(self.testing_set))
                            * 100.0
                        )
                        apcer = float(apcer) / float(num_attack) * 100.0
                        bpcer = float(bpcer) / float(num_bonafide) * 100.0

                        # print results
                        print("Model: " + self.get_model_filename(current_filter_size))
                        print("     CCR: " + str(ccr))
                        print("     APCER: " + str(apcer))
                        print("     BPCER: " + str(bpcer))

    def load_features(
        self, num_filters: int, filter_size: int, file_list: List[str]
    ) -> np.ndarray:
        # Load files
        filename = os.path.join(
            self.extraction_dir,
            "_".join(
                [
                    self.extraction_filename,
                    self.segmentation_type,
                    str(num_filters) + "filters",
                    str(filter_size) + "x" + str(filter_size),
                ]
            )
            + ".hdf5",
        )
        assert os.path.isfile(filename), "File not found: '{}'".format(filename)

        # create empty array (rows = number of image samples, cols = size of histogram)
        loaded_features = np.zeros(
            shape=(len(file_list), (2**num_filters)), dtype=np.float32
        )

        feature_file = h5py.File(filename, "r+")

        try:
            # loop through desired image features
            for i in range(len(file_list)):
                name = file_list[i]
                if name in feature_file:
                    histogram = feature_file[name]
                    # z-normalize histogram
                    mean = np.mean(histogram)
                    stddev = np.std(histogram)
                    histogram = (histogram - mean) / stddev
                    loaded_features[i] = histogram
                else:
                    print(
                        "Features not found for {} filters of size {}".format(
                            str(num_filters), str(filter_size)
                        )
                    )
        finally:
            feature_file.close()

        return loaded_features

    def load_sets(self):
        # initialize parameters to hold sets
        self.training_set = []
        self.training_classes = []
        self.testing_set = []
        self.testing_classes = []

        # open training file as long as it is specified
        if self.train_filename:
            trainFile = open((self.split_dir + self.train_filename), "r")
            # loop through lines
            for line in trainFile:
                if not (len(line.split(",")) == 2):
                    print("Incorrectly formatted set file.")
                self.training_set.append(line.split(",")[0])
                self.training_classes.append(int(line.split(",")[1].strip()))
            self.training_classes = np.asarray(self.training_classes)

        # open testing file as long as it is specified
        if self.test_filename:
            with open(os.path.join(self.split_dir, self.test_filename), "r") as fp:
                # read lines
                for line in fp:
                    if not (len(line.split(",")) == 2):
                        print("Incorrectly formatted set file.")
                    self.testing_set.append(line.split(",")[0])
                    self.testing_classes.append(int(line.split(",")[1].strip()))
            self.testing_classes = np.asarray(self.testing_classes)

    def get_model_filename(self, idx: int) -> str:
        return (
            "BSIF-"
            + str(self.all_num_filters[idx])
            + "-"
            + str(self.all_filter_sizes[idx])
            + "-"
            + self.model_types[idx]
            + "-"
            + self.segmentation_type
            + ".xml"
        )


if __name__ == "__main__":
    newManager = Manager()

    newManager.load_config("config.ini")
    newManager.show_config()
    newManager.run()
