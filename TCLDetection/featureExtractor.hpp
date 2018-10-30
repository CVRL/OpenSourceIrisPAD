//
//  featureExtractor.hpp
//  TCLDetection



#ifndef featureExtractor_hpp
#define featureExtractor_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include "BSIFFilter.hpp"


class featureExtractor
{
public:
    featureExtractor(int bits, std::vector<std::string>& inFilenames, std::string& segmentationType);
    
    void extract(std::string& outDir, std::string& outName, std::string& imageDir);
    
    void extract(std::string& outDir, std::string& outName, std::string& imageDir, int filtersize);
    
private:
    // Filter information
    int bitsize;
    
    // Segmentation information
    std::string segmentation;
    
    // Output information
    std::string outputLocation;
    
    // Image Locations
    std::string imageLocation;
    
    // List of filenames
    std::vector<std::string>& filenames;
    
    // Function produces features for filter size and its double (through downsampling)
    void filter(int filterSize);
};


#endif /* featureExtractor_hpp */
