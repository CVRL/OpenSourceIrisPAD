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
#include "filtermap.hpp"
#include "CSVIterator.hpp"


using namespace std;

// convert linear indexing to subscript indexing
int s2i(int* dims, int i, int j, int k){
    // C++ and python use row-major order, so the last dimension is contiguous
    // in doubt, refer to https://en.wikipedia.org/wiki/Row-_and_column-major_order#Column-major_order
    return k + dims[2]*(j+dims[1]*i);
}



void bsif_hist(cv::Mat src, int* histogram, int* dims, cv::Mat& dst){
    // Returns bsif histogram for an image
    int numScl = (int) dims[2];
    
    //initializing matrix of 1s
    double codeImg[src.rows][src.cols];
    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++){
            codeImg[i][j] = 1;
        }
    }
    
    // creates the border around the image - it is wrapping
    int border = floor(dims[0]/2);
    cv::Mat imgWrap = src;
    cv::copyMakeBorder(src, imgWrap, border, border, border, border, cv::BORDER_WRAP);
    
    // load the hard-coded filters
    t_filtermap filters = build_filter_map();
    // here we retrieve a filter from the map
    char filtername[50];
    sprintf(filtername, "filter_%d_%d_%d", dims[0], dims[1], dims[2]);
    
    double* myFilter;
    myFilter = filters[filtername];
    
    // Loop over scales
    cv::Mat ci; // the textured image after filter
    double currentFilter[dims[0]*dims[1]];
    int itr = 0;
    
    // pull the data from the matfile into an array
    // the matlab file is in one long single array
    // we need to start w/ the last filter and work our way forward
    for (int filterNum = numScl - 1; filterNum >= 0; filterNum--){
        
        for (int row=0; row<dims[0]; row++){
            for (int column=0; column<dims[1]; column++){
                currentFilter[column+(row*dims[1])] = myFilter[s2i(dims, row, column, filterNum)];
            }
        }
        //convert the array into matlab object to use w/ filter
        cv::Mat tmpcv2 = cv::Mat(dims[0], dims[1], CV_64FC1, &currentFilter);
        
        // running the filter on the image w/ BORDER WRAP - equivalent to filter2 in matlab
        // filter2d will incidentally create another border - we do not want this extra border
        cv::filter2D(imgWrap, ci, CV_64FC1, tmpcv2, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
        
        // This will convert any positive values in the matrix
        // to 2^(i-1) as it did in the matlab software
        for (int j = 0; j < src.rows; j++){
            for (int k = 0; k < src.cols; k++){
                if (ci.at<double>(j+border,k+border) > (pow(10, -3))) {// ignore the extra border added on
                    codeImg[j][k] = codeImg[j][k] + pow(2,itr);
                }
            }
        }
        itr++;
    }
    
    // Creating the histogram
    for (int j = 0; j < src.rows; j++){
        for (int k = 0; k < src.cols; k++){
            histogram[(int)codeImg[j][k]]++;
        }
    }
    
    
    // // Outputting to a CSV file
    // ofstream histfile;
    // histfile.open("histogram.csv", ios::out | ios::trunc);
    // histfile << "Hist = [";
    // for (int i = 0; i < histsize; i++)
    //     histfile << histogram[i] << ", ";
    // histfile << "];";
    // histfile.close();
    
    // convert the calculated image to return it
    cv::Mat tmp = cv::Mat(src.rows, src.cols, CV_64FC1, &codeImg);
    dst = tmp.clone();
    
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        cout << "Usage: bsifcpp <mode> <file> <filter height> <filter width> <filter depth>" << endl;
        cout << "Example: bsifcpp img myimage.png 3 3 8" << endl;
    }
    
    // Set filename
    char *filename = argv[2];
    
    // when reading the filter, we have to know its dimensions
    int dims[3];
    dims[0] = atoi(argv[3]);
    dims[1] = atoi(argv[4]);
    dims[2] = atoi(argv[5]);
    
    
    // allocate memory for the histogram
    int histsize = pow(2, dims[2]) + 1; // account for inclusion of 0
    int histogram[histsize];
    memset(histogram, 0, histsize*sizeof(int));
    
    // create imout
    cv::Mat imout;
    
    if (strncmp(argv[1], "img", 4) == 0)  {
        // Single image mode
        // reading in the image with open cv - returns array of doubles
        cv::Mat image = cv::imread(filename, 0);
        bsif_hist(image, &histogram[0], &dims[0], imout);
        
        ofstream histfile;
        histfile.open("histogram_single.csv", ios::out | ios::trunc);
        for (int i = 1; i < (histsize - 1); i++) histfile << histogram[i] << ", ";
        histfile << histogram[histsize-1] << endl;
        
        cv::Mat im2 = cv::Mat(image.rows, image.cols, CV_8UC1);
        cv::normalize(imout, im2, 0, 255, cv::NORM_MINMAX);
        imwrite("new_output.png", im2);
    } else if (strncmp(argv[1], "csv", 4) == 0) {
        // csv file mode
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
    
    return 0;
    
    
}
