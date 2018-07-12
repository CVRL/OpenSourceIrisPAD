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
    if (argc != 6) {
        cout << "Usage: bsifcpp <mode> <file> <filter height> <filter width> <filter depth>" << endl;
        cout << "Example: bsifcpp img myimage.png 3 3 8" << endl;
    }
    
    // Set filename
    char *filename = argv[1];
    
    // Read image
    cv::Mat image = cv::imread(filename, 0);
    
    // Declare BSIF filter
    BSIFFilter threeXthree(3,8);
    
    // Initialize histogram
    int histsize = pow(2,8) + 1;
    std::vector<int> histogram(histsize, 0);
    
    // Initialize output image
    cv::Mat imout;
    
    threeXthree.generateImage(image,imout);
    imwrite("new_output.png", imout);
    threeXthree.generateHistogram(image, histogram);
    
    ofstream histfile;
    histfile.open("histogram_single.csv", ios::out | ios::trunc);
    for (int i = 1; i < (histogram.size() - 1); i++) histfile << histogram[i] << ", ";
    histfile << histogram[histogram.size()-1] << endl;
        
    //cv::Mat im2 = cv::Mat(image.rows, image.cols, CV_8UC1);
    //cv::normalize(imout, im2, 0, 255, cv::NORM_MINMAX);
    //imwrite("new_output.png", im2);
    
        //} else if (strncmp(argv[1], "csv", 4) == 0) {
        // csv file mode
        //ifstream listOfImages(filename);
        //CSVIterator imageList(listOfImages);
        //size_t numCols = (*imageList).size();
        
        // Find locations of desired columns
        //int i = 2; // number of desired columns
        //size_t sequenceid = 0;
        //size_t format = 0;
        
        //for (size_t k = 0; k < numCols; k++) {
          //  if ((*imageList)[k] == "sequenceid") {
            //    sequenceid = k;
            //    i--;
            //} else if ((*imageList)[k] == "format") {
             //   format = k;
            //    i--;
            //}
        //}
        
       /* if (i == 0) {
            // If both parameters present
            // // Outputting to a CSV file
            ofstream histfile;
            histfile.open("histogram.csv", ios::out | ios::trunc);
            
            // String to store the current file name
            string currentImage;
            
            imageList++; // Skip the header line of the .csv
            
            while (imageList != CSVIterator()) {
                memset(histogram, 0, histsize*sizeof(int)); //reset histogram memory
                currentImage = "/Users/josephmcgrath/Desktop/SummerBiometricsResearch.nosync/NDCLD15/TIFF/" + (*imageList)[sequenceid] + "." + (*imageList)[format]; //Make sure to include file path
                cout << "Currently calculating features for: " << (*imageList)[sequenceid] + "." + (*imageList)[format] << endl;
                cv::Mat image = cv::imread(currentImage, 0);
                bsif_hist(image, &histogram[0], &dims[0], imout);
                histfile << (*imageList)[sequenceid] + "." + (*imageList)[format] + ", ";
                for (int i = 0; i < histsize; i++) histfile << histogram[i] << ", ";
                histfile << endl;
                imageList++;
            }
            
            histfile.close();
        } else {
            cout << "Error: incorrect .csv file format.  Need to have sequenceid and format available." << endl;
        }
    } else {
        cout << "Error: Please enter a valid mode of operation" << endl;
    }
    
    //cv::Mat imout;
    //bsif(image, imout, &dims[0]);
    
    //cv::Mat im2 = cv::Mat(image.rows, image.cols, CV_8UC1);
    //cv::normalize(imout, im2, 0, 255, cv::NORM_MINMAX);
    //imwrite("new_output.png", im2);
    */
    return 0;
    
    
}
