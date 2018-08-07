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
    cout << "Filtering with size 3..." << endl;
    filter(3, database);
    cout << "Filtering with size 5..." << endl;
    filter(5, database);
    cout << "Filtering with size 7..." << endl;
    filter(7, database);
    cout << "Filtering with size 9..." << endl;
    filter(9, database);
    cout << "Filtering with size 11..." << endl;
    filter(11, database);
    cout << "Filtering with size 13..." << endl;
    filter(13, database);
    cout << "Filtering with size 15..." << endl;
    filter(15, database);
    cout << "Filtering with size 17..." << endl;
    filter(17, database);
    cout << "Done generating features." << endl;
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
        //std::cout << "Processing subject " << (i + 1) << " out of " << (*db).size() << endl;
        
        // Load current subject
        subject currentSubject = (*db)[i];
        
        // Loop through samples for each subject
        for (int j = 0; j < (*currentSubject).size(); j++) {
            // Display progress
            //std::cout << "  Image " << (j + 1) << " out of " << (*currentSubject).size() << endl;
            
            // Load current sample
            irisSample currentSample = (*currentSubject)[j];
            
            // Load image from file
            cv::Mat image = cv::imread((imageLocation + currentSample.filename), 0);
            
            // Save image information
            histOut << currentSample.filename << ",";
            downHistOut << currentSample.filename << ",";
            
            // Calculate histograms for full sized image
            currentFilter.generateHistogram(image, histogram);
            
            // Ignore 0 position in histogram (image initialized to 1s in BSIFfilter so no 0s will be present)
            for (int i = 1; i < (histsize - 1); i++) histOut << histogram[i] << ",";
            // Need to output endl after last column instead of ","
            histOut << histogram[histsize - 1] << std::endl;
            std::fill(histogram.begin(), histogram.end(), 0);
           
            // Downsample image by 50% in either direction
            cv::Mat downImage;
            cv::pyrDown(image, downImage, cv::Size(image.cols / 2, image.rows / 2));
            
            // Run filter on downsampled image (simulates doubling of BSIF kernel size)
            currentFilter.generateHistogram(downImage, histogram);
            for (int i = 1; i < (histsize - 1); i++) downHistOut << histogram[i] << ", ";
            // Need to output endl after last column instead of ","
            downHistOut << histogram[histsize - 1] << std::endl;
            std::fill(histogram.begin(), histogram.end(), 0);
        }
    }
    
    // Close files
    histOut.close();
    downHistOut.close();
}
