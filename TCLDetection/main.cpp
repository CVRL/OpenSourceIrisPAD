//
//  main.cpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/11/18.
//  Copyright © 2018 Joseph McGrath. All rights reserved.
//

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>
#include <regex>
#include <fstream>
#include "CSVIterator.hpp"
#include "BSIFFilter.hpp"


using namespace std;
/* useful for identifying Mat object type
string type2str(int type) {
    string r;
    
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    
    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }
    
    r += "C";
    r += (chans+'0');
    
    return r;
}
*/

int main(int argc, char *argv[]) {
    if (argc != 5) {
        cout << "Usage: TCLD <mode> <file> <filter dimension> <filter bits>" << endl;
        cout << "Modes: " << endl;
        cout << "0 Extract features (.csv input)" << endl;
        cout << "1 Make decision (image input)" << endl;
        cout << "2 Train SVM" << endl;
        cout << "Example: bsifcpp 0 myimage.png 3 8" << endl;
        return 0;
    }
    
    // Set mode
    int mode = atoi(argv[1]);
    
    // Set filename
    char *filename = argv[2];
    
    // Set filter specifications
    int dimension = atoi(argv[3]);
    int bitsize = atoi(argv[4]);
    cout << "Filter size: " << dimension << endl << "Bit size: "  << bitsize << endl;
    // Declare BSIF filter
    BSIFFilter filter(dimension,bitsize);
    
    // Initialize histogram
    int histsize = pow(2,bitsize) + 1;
    std::vector<int> histogram(histsize, 0);
    
    
    switch(mode) {
        case 0: {
            // csv file mode: take in .csv with names of image files
            ifstream listOfImages(filename);
            CSVIterator imageList(listOfImages);
            size_t numCols = (*imageList).size();
            
            // Find locations of desired columns
            int i = 3; // number of desired columns
            size_t sequenceid = 0;
            size_t format = 0;
            size_t texture = 0;
            
            for (size_t k = 0; k < numCols; k++) {
                if ((*imageList)[k] == "sequenceid") {
                    sequenceid = k;
                    i--;
                } else if ((*imageList)[k] == "format") {
                    format = k;
                    i--;
                } else if ((*imageList)[k] == "contacts_texture") {
                    texture = k + 1; // need to add one because illuminant id will be read as two separate parts
                    i--;
                }
            }
            
            if (i == 0) {
                // If both parameters present
                // Outputting to a CSV file
                ofstream histfile;
                histfile.open("histogram.csv", ios::out | ios::trunc);
                
                // String to store the current file name
                string currentImage;
                
                imageList++; // Skip the header line of the .csv
                
                while (imageList != CSVIterator()) {
                    std::vector<int> histogram(histsize, 0); //reset histogram memory
                    currentImage = "./NDCLD15/TIFF/" + (*imageList)[sequenceid] + "." + (*imageList)[format]; //Make sure to include file path
                    cout << "Currently calculating features for: " << (*imageList)[sequenceid] + "." + (*imageList)[format] << endl;
                    cv::Mat image = cv::imread(currentImage, 0);
                    filter.generateHistogram(image, histogram);
                    histfile << (*imageList)[sequenceid] + "." + (*imageList)[format] + ", ";
                    if ((*imageList)[texture] == "No") {
                        histfile << "-1" << ", ";
                    } else {
                        histfile << "1" << ", ";
                    }
                    for (int i = 1; i < histsize; i++) histfile << histogram[i] << ", ";
                    histfile << endl;
                    imageList++;
                }
                
                histfile.close();
            } else {
                cout << "Error: incorrect .csv file format.  Need to have sequenceid and format available." << endl;
            }
            break;
        }
        case 1: {
            // For now will just generate features for a single image
            // Read image
            cv::Mat image = cv::imread(filename, 0);
            
            // Initialize output image
            cv::Mat imout;
            
            // Generate image
            filter.generateImage(image,imout);
            imwrite("new_output.png", imout);
            
            // output histogram to csv
            filter.generateHistogram(image, histogram);
            ofstream histfile;
            histfile.open("histogram_single.csv", ios::out | ios::trunc);
            for (int i = 1; i < (histogram.size() - 1); i++) histfile << histogram[i] << ", ";
            histfile << histogram[histogram.size()-1] << endl;
            histfile.close();
            break;
        }
        case 2: {
            // Set up SVM parameters
            cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
            svm->setType(cv::ml::SVM::C_SVC);
            svm->setKernel(cv::ml::SVM::LINEAR);
            svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
            
            // Import training data from CSV file
            cv::Ptr<cv::ml::TrainData> histData = cv::ml::TrainData::loadFromCSV("histogram.csv", 0, 1, -1); //Filename, lines to skip for header, label location, -1 means only one response variable
            
            // Test for error
            if (histData->getNSamples() == 0) {
                cout << "Error: Could not read training data file histogram.csv." << endl;
                break;
            }
            
            // Set split ratio
            histData -> setTrainTestSplit(7299);
            int n_train_samples = histData->getNTrainSamples();
            cout << "The number of training samples is " << n_train_samples << " out of 7300 total samples." << endl;
            // Train SVM
            svm->train(histData, 0);
            
            cv::Mat testData = histData->getTestSamples();
            cv::Mat testResponses = histData->getTestResponses();
            
            //int n_test_samples = histData->getNTestSamples();
            
            float result = svm->predict(testData);
            cout << "The prediction is " << result << " and the actual value is " << testResponses << "." << endl;
            break;
        }
        default: {
            cout << "Error: Please enter a valid mode of operation." << endl;
        }
    }
    
    
    return 0;
    
    
}
