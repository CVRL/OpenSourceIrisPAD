# OpenSourceIrisPAD


This repo contains the open-source (planned) implementation of iris PAD based on BSIF and a fusion of multiple classifiers, and is based on Jay Doyle's paper: "Robust Detection of Textured Contact Lenses in Iris Recognition Using BSIF", IEEE Access, 2015 (https://ieeexplore.ieee.org/document/7264974/).

## Requirements

This iris PAD implementation is based on [OpenCV 3.4.1](https://opencv.org).

## Usage

TCL Detection comes with three built in capabilities: BSIF feature extraction, SVM training, and performance testing.  To select capabilities and set various parameters, edit the settings in the included configuration file (configuration.ini). To compile the program, use make.


### BSIF Feature Extraction

If feature extraction is selected, the filenames specified in the training and testing csv files will be used as images for [BSIF](http://www.ee.oulu.fi/~jkannala/bsif/bsif.html) feature extraction.  This process involves filtering with previously defined BSIF filters and then producing a histogram characterizing the image in terms of these filters. For example, if 8 filters are used, as specified in this implementation, each pixel in the image will have an 8 bit integer where each binary position represents the pixel's response to a specific filter.  The histogram has counts for the number of pixels with each value, 1-256.

To use this feature, you must specify:
- image directory
- split directory and filenames
- output directory
- output filename (outputs will be dir/filename_filter_size_size_bits.csv)

This process will produce 16 files, one for each BSIF size.  The main sizes are 3,5,7,9,11,13,15, and 17.  The second set of 8 sizes is produced by downsampling the images by 50%, effectively doubling the filter size and producing outputs 6,10,14,18,22,26,30, and 34.

The output file format is a csv with image_filename,features on each line.

### SVM Training

If model training is selected, a SVM will be trained using the RBF (Gaussian) kernel specified in OpenCV.  The trainAuto function in OpenCV is used to select optimal parameters for each model.  In order to use this function, the required BSIF features must already be extracted.

To use this feature, you must specify:
- split directory and filenames
- feature extraction filename and directory
- training sizes to use (for this, you may specify any of the 16 sizes as a comma separated list)
- model output directory

The training process will produce xml files for each model trained, allowing the models to be loaded in the future.

In this release, models have been included that have been trained on the NDCLD15 database, which includes 5 brands of textured contact lenses (2500 images) and 4800 clear lens or no lens images.

### Image testing

If image testing is selected, the models specified in the training sizes list will be used to classify the images in the testing set.  You may choose to use majority voting to group the ensemble of models, or test each model individually.

To use this feature, you must specify:
- split directory and filenames
- feature extraction filename and directory
- training sizes to use
- model output directory (models must be trained to use this feature)

The results of the test will be printed to the console.
