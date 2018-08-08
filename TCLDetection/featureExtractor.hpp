//
//  featureExtractor.hpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/27/18.


#ifndef featureExtractor_hpp
#define featureExtractor_hpp

#include <stdio.h>
#include "BSIFFilter.hpp"
#include "sampleDatabase.hpp"

class featureExtractor {
public:
    featureExtractor(int bits);
    
    void extract(std::string& outDir, std::string& outName, std::string& imageDir, vector<string>& training, vector<string>& testing);
    
private:
    // Filter information
    int bitsize;
    
    // Output information
    std::string outputLocation;
    
    // Image Locations
    std::string imageLocation;
    
    // List of filenames
    std::vector<string> filenames;
    
    // Function produces features for filter size and its double (through downsampling)
    void filter(int filterSize);
};


#endif /* featureExtractor_hpp */
