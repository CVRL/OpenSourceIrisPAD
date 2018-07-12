//
//  BSIFFilter.cpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/11/18.


#include "BSIFFilter.hpp"
BSIFFilter::BSIFFilter(int dimension, int bitlength) {size = dimension; bits = bitlength; }

void BSIFFilter::generateImage(cv::Mat src, cv::Mat& dst) {
    //initializing matrix of 1s
    double codeImg[src.rows][src.cols];
    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++){
            codeImg[i][j] = 1;
        }
    }
    
    // creates the border around the image - it is wrapping
    int border = floor(size/2);
    cv::Mat imgWrap = src;
    cv::copyMakeBorder(src, imgWrap, border, border, border, border, cv::BORDER_WRAP);
    
    // load the hard-coded filters
    t_filtermap filters = build_filter_map();
    // here we retrieve a filter from the map
    char filtername[50];
    sprintf(filtername, "filter_%d_%d_%d", size, size, bits);
    
    double* myFilter;
    myFilter = filters[filtername];
    
    // Loop over scales
    cv::Mat ci; // the textured image after filter
    double currentFilter[size * size];
    int itr = 0;
    
    // pull the data from the matfile into an array
    // the matlab file is in one long single array
    // we need to start w/ the last filter and work our way forward
    for (int filterNum = bits - 1; filterNum >= 0; filterNum--){
        
        for (int row=0; row<size; row++){
            for (int column=0; column<size; column++){
                currentFilter[column+(row*size)] = myFilter[s2i(size, bits, row, column, filterNum)];
            }
        }
        //convert the array into matlab object to use w/ filter
        cv::Mat tmpcv2 = cv::Mat(size, size, CV_64FC1, &currentFilter);
        
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
    
    cv::Mat tmp = cv::Mat(src.rows, src.cols, CV_64FC1, &codeImg);
    cv::Mat im2 = cv::Mat(src.rows, src.cols, CV_8UC1);
    cv::normalize(tmp, im2, 0, 255, cv::NORM_MINMAX);
    dst = im2;
}

void BSIFFilter::generateHistogram(cv::Mat src, std::vector<int>& histogram) {
    //initializing matrix of 1s
    double codeImg[src.rows][src.cols];
    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++){
            codeImg[i][j] = 1;
        }
    }
    
    // creates the border around the image - it is wrapping
    int border = floor(size/2);
    cv::Mat imgWrap = src;
    cv::copyMakeBorder(src, imgWrap, border, border, border, border, cv::BORDER_WRAP);
    
    // load the hard-coded filters
    t_filtermap filters = build_filter_map();
    // here we retrieve a filter from the map
    char filtername[50];
    sprintf(filtername, "filter_%d_%d_%d", size, size, bits);
    
    double* myFilter;
    myFilter = filters[filtername];
    
    // Loop over scales
    cv::Mat ci; // the textured image after filter
    double currentFilter[size * size];
    int itr = 0;
    
    // pull the data from the matfile into an array
    // the matlab file is in one long single array
    // we need to start w/ the last filter and work our way forward
    for (int filterNum = bits - 1; filterNum >= 0; filterNum--){
        
        for (int row=0; row<size; row++){
            for (int column=0; column<size; column++){
                currentFilter[column+(row*size)] = myFilter[s2i(size, bits, row, column, filterNum)];
            }
        }
        //convert the array into matlab object to use w/ filter
        cv::Mat tmpcv2 = cv::Mat(size, size, CV_64FC1, &currentFilter);
        
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
}

// convert linear indexing to subscript indexing
int s2i(int size, int bits, int i, int j, int k){
    // C++ and python use row-major order, so the last dimension is contiguous
    // in doubt, refer to https://en.wikipedia.org/wiki/Row-_and_column-major_order#Column-major_order
    return k + bits*(j+size*i);
}
