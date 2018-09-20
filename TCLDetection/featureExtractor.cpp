//
//  featureExtractor.cpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/27/18.


#include "featureExtractor.hpp"


using namespace std;

featureExtractor::featureExtractor(int bits, vector<string>& inFilenames) : bitsize(bits), filenames(inFilenames){}
    
void featureExtractor::extract(std::string& outDir, std::string& outName, std::string& imageDir)
{
    
    outputLocation = outDir + outName;
    imageLocation = imageDir;
    
    // Filter with 8 different sizes
    cout << "Filtering with size 3..." << endl;
    try
    {
        filter(3);
    }
    catch (runtime_error& e)
    {
        throw e;
    }
    
    
    cout << "Filtering with size 5..." << endl;
    try
    {
        filter(5);
    }
    catch (runtime_error& e)
    {
        throw e;
    }
    
    
    cout << "Filtering with size 7..." << endl;
    try
    {
        filter(7);
    }
    catch (runtime_error& e)
    {
        throw e;
    }
    
    
    cout << "Filtering with size 9..." << endl;
    try
    {
        filter(9);
    }
    catch (runtime_error& e)
    {
        throw e;
    }
    
    
    cout << "Filtering with size 11..." << endl;
    try
    {
        filter(11);
    }
    catch (runtime_error& e)
    {
        throw e;
    }
    
    
    cout << "Filtering with size 13..." << endl;
    try
    {
        filter(13);
    }
    catch (runtime_error& e)
    {
        throw e;
    }
    
    
    cout << "Filtering with size 15..." << endl;
    try
    {
        filter(15);
    }
    catch (runtime_error& e)
    {
        throw e;
    }
    
    
    cout << "Filtering with size 17..." << endl;
    try
    {
        filter(17);
    }
    catch (runtime_error& e)
    {
        throw e;
    }
    
    
    cout << "Done generating features." << endl;
}





// Function produces features for filter size and its double (through downsampling)
void featureExtractor::filter(int filterSize)
{
    
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
    for (int i = 0; i < (int)filenames.size(); i++)
    {
        // Display progress
        std::cout << "Processing sample " << (i + 1) << " out of " << filenames.size() << endl;
        cout << filenames[i] << endl;
        
        // Load image from file
        cv::Mat image = cv::imread((imageLocation + filenames[i]), 0);
        
        
        
        
        if ( image.empty() )
        {
            throw runtime_error("Error: unable to read image " + filenames[i] + " for feature extraction.");
        }
        
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
