//
//  featureExtractor.cpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/27/18.


#include "featureExtractor.hpp"


using namespace std;

featureExtractor::featureExtractor(int bits) : bitsize(bits){}
    
void featureExtractor::extract(std::string& outDir, std::string& outName, std::string& imageDir, vector<string>& training, vector<string>& testing) {
    outputLocation = outDir + outName;
    imageLocation = imageDir;
    
    // Combine both sets into a single vector
    filenames.insert(filenames.end(), training.begin(), training.end());
    filenames.insert(filenames.end(), testing.begin(), testing.end());
    
    // Filter with 8 different sizes
    cout << "Filtering with size 3..." << endl;
    filter(3);
    cout << "Filtering with size 5..." << endl;
    filter(5);
    cout << "Filtering with size 7..." << endl;
    filter(7);
    cout << "Filtering with size 9..." << endl;
    filter(9);
    cout << "Filtering with size 11..." << endl;
    filter(11);
    cout << "Filtering with size 13..." << endl;
    filter(13);
    cout << "Filtering with size 15..." << endl;
    filter(15);
    cout << "Filtering with size 17..." << endl;
    filter(17);
    cout << "Done generating features." << endl;
}
    
// Function produces features for filter size and its double (through downsampling)
void featureExtractor::filter(int filterSize) {
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
    for (int i = 0; i < filenames.size(); i++) {
        // Display progress
        std::cout << "Processing sample " << (i + 1) << " out of " << filenames.size() << endl;
        cout << filenames[i] << endl;
        // Load image from file
        cv::Mat image = cv::imread((imageLocation + filenames[i]), 0);
        
        // Save image information
        histOut << filenames[i] << ",";
        downHistOut << filenames[i] << ",";
        
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

    
    // Close files
    histOut.close();
    downHistOut.close();
}
