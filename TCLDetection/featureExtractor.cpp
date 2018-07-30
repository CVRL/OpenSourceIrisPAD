//
//  featureExtractor.cpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/27/18.


#include "featureExtractor.hpp"


featureExtractor::featureExtractor(int bits) : bitsize(bits){}
    
void featureExtractor::extract(std::string& outDir, std::string& outName, std::string& imageDir, sampleDatabase& database) {
    outputLocation = outDir + outName;
    imageLocation = imageDir;
    
    // Filter with 8 different sizes
    filter(3, database);
    filter(5, database);
    filter(7, database);
    filter(9, database);
    filter(11, database);
    filter(13, database);
    filter(15, database);
    filter(17, database);
}
    
// Function produces features for filter size and its double (through downsampling)
void featureExtractor::filter(int filterSize, sampleDatabase& db) {
    // Load filter
    BSIFFilter currentFilter;
    currentFilter.loadFilter(filterSize, bitsize);
    
    // Open files
    ofstream histOut;
    histOut.open((outputLocation + "_" + currentFilter.filtername + ".csv"), ios::out | ios::trunc);
    
    ofstream downHistOut;
    downHistOut.open((outputLocation + "_" + currentFilter.downFiltername + ".csv"), ios::out | ios::trunc);
    
    // Initialize histogram
    int histsize = pow(2,bitsize) + 1;
    std::vector<int> histogram(histsize, 0);
    
    // Loop through images
    for (int i = 0; i < (*db).size(); i++) {
        // Display progress
        std::cout << "Processing subject " << (i + 1) << " out of " << (*db).size() << endl;
        
        // Load current subject
        subject currentSubject = (*db)[i];
        
        // Loop through samples for each subject
        for (int j = 0; j < (*currentSubject).size(); j++) {
            // Display progress
            std::cout << "  Image " << (j + 1) << " out of " << (*currentSubject).size() << endl;
            
            // Load current sample
            irisSample currentSample = (*currentSubject)[j];
            
            // Load image from file
            cv::Mat image = cv::imread((imageLocation + currentSample.filename), 0);
            
            // Save image information
            histOut << currentSample.filename << ", ";
            downHistOut << currentSample.filename << ", ";
            
            // Calculate histograms for full sized image
            currentFilter.generateHistogram(image, histogram);
            for (int i = 1; i < histsize; i++) histOut << histogram[i] << ", ";
            histOut << std::endl;
            std::fill(histogram.begin(), histogram.end(), 0);
            
            // Downsample image by 50% in either direction
            cv::Mat downImage;
            cv::pyrDown(image, downImage, cv::Size(image.cols / 2, image.rows / 2));
            
            // Run filter on downsampled image (simulates doubling of BSIF kernel size)
            currentFilter.generateHistogram(downImage, histogram);
            for (int i = 1; i < histsize; i++) downHistOut << histogram[i] << ", ";
            downHistOut << std::endl;
            std::fill(histogram.begin(), histogram.end(), 0);
        }
    }
    
    // Close files
    histOut.close();
    downHistOut.close();
}
