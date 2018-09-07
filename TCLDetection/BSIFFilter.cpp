//
//  BSIFFilter.cpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/11/18.


#include "BSIFFilter.hpp"

#include "filters.h"


BSIFFilter::BSIFFilter(void) {}

void BSIFFilter::loadFilter(int dimension, int bitlength) {
    size = dimension; bits = bitlength;
    
    // Set the filter name
    std::stringstream nameStream;
    nameStream << "filter_" << size << "_" << size << "_" << bits;
    filtername = nameStream.str();
    
    // Set the downsampled filter name
    nameStream.str(std::string());
    nameStream << "filter_" << (size * 2) << "_" << (size * 2) << "_" << bits;
    downFiltername = nameStream.str();
    
    // load the hard-coded filters
    t_filtermap filters = build_filter_map();
    
    // Retrieve filter from filtermap
    double* myFilter;
    myFilter = filters[filtername];
}

void BSIFFilter::generateImage(cv::Mat src, cv::Mat& dst) {
    //initializing matrix of 1s
    cv::Mat codeImg = cv::Mat::ones(src.rows, src.cols, CV_64FC1);
    
    
    // creates the border around the image - it is wrapping
    int border = floor(size/2);
    cv::Mat imgWrap = src;
    cv::copyMakeBorder(src, imgWrap, border, border, border, border, cv::BORDER_WRAP);
    
    // the textured image after filter
    cv::Mat ci;
    
    cv::Mat currentFilter = cv::Mat(size, size, CV_64FC1);
    
    int itr = 0;
    
    // pull the data from the matfile into an array
    // the matlab file is in one long single array
    // we need to start w/ the last filter and work our way forward
    for (int filterNum = bits - 1; filterNum >= 0; filterNum--){
        
        for (int row=0; row<size; row++){
            for (int column=0; column<size; column++){
                currentFilter.at<double>(row,column) = myFilter[s2i(size, bits, row, column, filterNum)];
            }
        }
        
        
        // running the filter on the image w/ BORDER WRAP - equivalent to filter2 in matlab
        // filter2d will incidentally create another border - we do not want this extra border
        cv::filter2D(imgWrap, ci, CV_64FC1, currentFilter, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
        
        // This will convert any positive values in the matrix
        // to 2^(i-1) as it did in the matlab software
        for (int j = 0; j < src.rows; j++){
            for (int k = 0; k < src.cols; k++){
                if (ci.at<double>(j+border,k+border) > (pow(10, -3))) {// ignore the extra border added on
                    codeImg.at<double>(j,k) = codeImg.at<double>(j,k) + pow(2,itr);
                }
            }
        }
        itr++;
    }
    
    
    cv::Mat im2 = cv::Mat(src.rows, src.cols, CV_8UC1);
    cv::normalize(codeImg, im2, 0, 255, cv::NORM_MINMAX);
    dst = im2;
}

void BSIFFilter::generateHistogram(cv::Mat src, std::vector<int>& histogram) {
    //initializing matrix of 1s
    cv::Mat codeImg = cv::Mat::ones(src.rows, src.cols, CV_64FC1);
    
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
    
    // the textured image after filter
    cv::Mat ci;
    
    cv::Mat currentFilter = cv::Mat(size, size, CV_64FC1);
    
    int itr = 0;
    
    // pull the data from the matfile into an array
    // the matlab file is in one long single array
    // we need to start w/ the last filter and work our way forward
    for (int filterNum = bits - 1; filterNum >= 0; filterNum--){
        
        for (int row=0; row<size; row++){
            for (int column=0; column<size; column++){
                currentFilter.at<double>(row,column) = myFilter[s2i(size, bits, row, column, filterNum)];
            }
        }
        
        
        // running the filter on the image w/ BORDER WRAP - equivalent to filter2 in matlab
        // filter2d will incidentally create another border - we do not want this extra border
        cv::filter2D(imgWrap, ci, CV_64FC1, currentFilter, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
        
        // This will convert any positive values in the matrix
        // to 2^(i-1) as it did in the matlab software
        for (int j = 0; j < src.rows; j++){
            for (int k = 0; k < src.cols; k++){
                if (ci.at<double>(j+border,k+border) > (pow(10, -3))) {// ignore the extra border added on
                    codeImg.at<double>(j,k) = codeImg.at<double>(j,k) + pow(2,itr);
                }
            }
        }
        itr++;
    }
    
    // Creating the histogram
    for (int j = 0; j < src.rows; j++){
        for (int k = 0; k < src.cols; k++){
            histogram[(int)codeImg.at<double>(j,k)]++;
        }
    }
}

// convert linear indexing to subscript indexing
int s2i(int size, int bits, int i, int j, int k){
    // C++ and python use row-major order, so the last dimension is contiguous
    // in doubt, refer to https://en.wikipedia.org/wiki/Row-_and_column-major_order#Column-major_order
    return k + bits*(j+size*i);
}




// build a map containing all the hard-coded ICA Filters, used to load the filters
t_filtermap build_filter_map(){
    // this is the map of filters
    t_filtermap filters;
    
    // we have to add all filters to the map
    // 3x3 filters
    std::string filterName = "filter_3_3_5";
    double* theFilter = &filter_3_3_5[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_3_3_6";
    theFilter = &filter_3_3_6[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_3_3_7";
    theFilter = &filter_3_3_7[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_3_3_8";
    theFilter = &filter_3_3_8[0];
    filters.insert(t_filterpair(filterName, theFilter));
    
    // 5x5 filters
    filterName = "filter_5_5_5";
    theFilter = &filter_5_5_5[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_5_5_6";
    theFilter = &filter_5_5_6[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_5_5_7";
    theFilter = &filter_5_5_7[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_5_5_8";
    theFilter = &filter_5_5_8[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_5_5_9";
    theFilter = &filter_5_5_9[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_5_5_10";
    theFilter = &filter_5_5_10[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_5_5_11";
    theFilter = &filter_5_5_11[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_5_5_12";
    theFilter = &filter_5_5_12[0];
    filters.insert(t_filterpair(filterName, theFilter));
    
    // 7x7 filters
    filterName = "filter_7_7_5";
    theFilter = &filter_7_7_5[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_7_7_6";
    theFilter = &filter_7_7_6[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_7_7_7";
    theFilter = &filter_7_7_7[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_7_7_8";
    theFilter = &filter_7_7_8[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_7_7_9";
    theFilter = &filter_7_7_9[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_7_7_10";
    theFilter = &filter_7_7_10[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_7_7_11";
    theFilter = &filter_7_7_11[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_7_7_12";
    theFilter = &filter_7_7_12[0];
    filters.insert(t_filterpair(filterName, theFilter));
    
    // 9x9 filters
    filterName = "filter_9_9_5";
    theFilter = &filter_9_9_5[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_9_9_6";
    theFilter = &filter_9_9_6[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_9_9_7";
    theFilter = &filter_9_9_7[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_9_9_8";
    theFilter = &filter_9_9_8[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_9_9_9";
    theFilter = &filter_9_9_9[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_9_9_10";
    theFilter = &filter_9_9_10[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_9_9_11";
    theFilter = &filter_9_9_11[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_9_9_12";
    theFilter = &filter_9_9_12[0];
    filters.insert(t_filterpair(filterName, theFilter));
    
    // 11x11 filters
    filterName = "filter_11_11_5";
    theFilter = &filter_11_11_5[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_11_11_6";
    theFilter = &filter_11_11_6[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_11_11_7";
    theFilter = &filter_11_11_7[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_11_11_8";
    theFilter = &filter_11_11_8[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_11_11_9";
    theFilter = &filter_11_11_9[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_11_11_10";
    theFilter = &filter_11_11_10[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_11_11_11";
    theFilter = &filter_11_11_11[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_11_11_12";
    theFilter = &filter_11_11_12[0];
    filters.insert(t_filterpair(filterName, theFilter));
    
    // 13x13 filters
    filterName = "filter_13_13_5";
    theFilter = &filter_13_13_5[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_13_13_6";
    theFilter = &filter_13_13_6[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_13_13_7";
    theFilter = &filter_13_13_7[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_13_13_8";
    theFilter = &filter_13_13_8[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_13_13_9";
    theFilter = &filter_13_13_9[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_13_13_10";
    theFilter = &filter_13_13_10[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_13_13_11";
    theFilter = &filter_13_13_11[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_13_13_12";
    theFilter = &filter_13_13_12[0];
    filters.insert(t_filterpair(filterName, theFilter));
    
    // 15x15 filters
    filterName = "filter_15_15_5";
    theFilter = &filter_15_15_5[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_15_15_6";
    theFilter = &filter_15_15_6[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_15_15_7";
    theFilter = &filter_15_15_7[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_15_15_8";
    theFilter = &filter_15_15_8[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_15_15_9";
    theFilter = &filter_15_15_9[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_15_15_10";
    theFilter = &filter_15_15_10[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_15_15_11";
    theFilter = &filter_15_15_11[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_15_15_12";
    theFilter = &filter_15_15_12[0];
    filters.insert(t_filterpair(filterName, theFilter));
    
    // 17x17 filters
    filterName = "filter_17_17_5";
    theFilter = &filter_17_17_5[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_17_17_6";
    theFilter = &filter_17_17_6[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_17_17_7";
    theFilter = &filter_17_17_7[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_17_17_8";
    theFilter = &filter_17_17_8[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_17_17_9";
    theFilter = &filter_17_17_9[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_17_17_10";
    theFilter = &filter_17_17_10[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_17_17_11";
    theFilter = &filter_17_17_11[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_17_17_12";
    theFilter = &filter_17_17_12[0];
    filters.insert(t_filterpair(filterName, theFilter));
    
    return filters;
}

