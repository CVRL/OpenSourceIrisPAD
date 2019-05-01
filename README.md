# OpenSourceIrisPAD (v2 - 13 April 2019)


This repo contains the open-source implementation of iris PAD based on BSIF and a fusion of multiple classifiers, and is based on Jay Doyle's paper: "Robust Detection of Textured Contact Lenses in Iris Recognition Using BSIF", IEEE Access, 2015 (https://ieeexplore.ieee.org/document/7264974/).

The paper presenting this implementation is available at: https://arxiv.org/abs/1809.10172

## Updates

- Added random forest and multilayer perceptron models
- Changed feature file format to HDF5
- Added Python implementation

## Requirements

This iris PAD implementation is based on [OpenCV 3.4.1](https://opencv.org) and [HDF5 1.10.4](https://www.hdfgroup.org).

## Usage

TCL Detection comes with three built in capabilities: BSIF feature extraction, model training, and performance testing.  To select capabilities and set various parameters, edit the settings in the included configuration file (configuration.ini). To compile the program, use make.

For the Python implementation, simply run the manager.py file to start the program.


### BSIF Feature Extraction

If feature extraction is selected, the filenames specified in the training and testing csv files will be used as images for [BSIF](http://www.ee.oulu.fi/~jkannala/bsif/bsif.html) feature extraction.  This process involves filtering with previously defined BSIF filters and then producing a histogram characterizing the image in terms of these filters. For example, if 8 filters are used, as specified in this implementation, each pixel in the image will have an 8 bit integer where each binary position represents the pixel's response to a specific filter.  The histogram has counts for the number of pixels with each value, 1-256.

To use this feature, you must specify:
- The image directory
- The directory and filenames for the lists with the image filenames and classifications
- The desired output directory
- The desired output filename (outputs will be dir/filename_filter_size_size_bits.csv)
- The number of filters (bitsize) and scale to use

This process will produce one file for each set of bitsize and scale.  The main scales are 3,5,7,9,11,13,15, and 17.  The second set of 8 scales is produced by downsampling the images by 50%, effectively doubling the filter size and producing outputs 6,10,14,18,22,26,30, and 34. Available bitsizes are 5,6,7,8,9,10,11,12; however, scales 3 and 6 are only available for bitsizes 5,6,7,8.

The output file format is an HDF5 file with histograms indexed by the name of the image they represent.

### Model Training

If model training is selected, the desired model type will be created and trained on the data specified in the training set file. For SVM, the trainAuto function in OpenCV is used to select optimal parameters for each model.  For random forest and multilayer perceptron, a custom training function has been implemented to mimic the functionality of the SVM trainAuto function: 10 fold cross validation is used to select the best parameters for each model. Currently, the trainAuto functionality for random forest and multilayer perceptron is only available in the C++ version. In order to use the training functionality, the required BSIF features must already be extracted.

To use this feature, you must specify:
- The directory and filenames for the lists with the image filenames and classifications (only training required)
- The location and filename of the features that will be used for training (must run feature extraction prior to this step)
- The training sizes to use (for this, you may specify any of the 16 sizes as a comma separated list)
- The training bitsizes to use (for this, you may specify any of the 8 bitsizes as a comma separated list)
- The model types to use (for this, you may specify any of the 3 model types as a comma separated list)
- The desired model output directory

The training process will produce xml files for each model trained, allowing the models to be loaded in the future.

In this release, 360 (120 feature sets * 3 model types) models have been included that have been trained on the NDCLD15 database, which includes 5 brands of textured contact lenses (2500 images) and 4800 clear lens or no lens images. The following brands are represented in the database:
- CIBA Vision
- United Contact Lenses
- Clearlab
- Johnson&Johnson
- CooperVision

To access these models, go to https://notredame.box.com/v/OpenSourceIrisPADModels.

### Image testing

If image testing is selected, the models specified in the configuration file will be used to classify the images in the testing set.  You may choose to use majority voting to group the ensemble of models, or test each model individually.

To use this feature, you must specify:
- The directory and filenames for the lists with the image filenames and classifications (only testing required)
- The location and filename of the features that will be used for testing (must run feature extraction prior to this step)
- The BSIF sizes, bitsizes, and model types to use in testing (the models must already be trained and saved)
- The directory of the saved models

The results of the test will be printed to the console.
