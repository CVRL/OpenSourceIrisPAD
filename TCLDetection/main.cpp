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
#include <string>
#include <regex>
#include <fstream>
#include "CSVIterator.hpp"
#include "BSIFFilter.hpp"


using namespace std;


int main(int argc, char *argv[]) {
    if (argc != 5) {
        cout << "Usage: TCLD <mode> <file> <filter dimension> <filter bits>" << endl;
        cout << "Modes: " << endl << "0 Retrain Model (.csv input)" << endl << "1 Make Decision (image input)" << endl;
        cout << "Example: bsifcpp 0 myimage.png 3 8" << endl;
        return 0;
    }
    
    // Set mode
    char mode = atoi(argv[1]);
    
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
            int i = 2; // number of desired columns
            size_t sequenceid = 0;
            size_t format = 0;
            
            for (size_t k = 0; k < numCols; k++) {
              if ((*imageList)[k] == "sequenceid") {
                sequenceid = k;
                i--;
              } else if ((*imageList)[k] == "format") {
               format = k;
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
            break;
        }
        default: {
            cout << "Error: Please enter a valid mode of operation." << endl;
        }
    }
    
    
    return 0;
    
    
}
