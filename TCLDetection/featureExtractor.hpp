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
    
    void extract(std::string& outDir, std::string& outName, std::string& imageDir, sampleDatabase& database);
    
private:
    // Filter information
    int bitsize;
    
    // Output information
    std::string outputLocation;
    
    // Image Locations
    std::string imageLocation;
    
    
    // Function produces features for filter size and its double (through downsampling)
    void filter(int filterSize, sampleDatabase& db);
};


#endif /* featureExtractor_hpp */
