//
//  featureExtractor.cpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/27/18.


#include "featureExtractor.hpp"
#include "BSIFFilter.hpp"
#include "sampleDatabase.cpp"

class featureExtractor {
public:
    featureExtractor(int bits) : bitsize(bits){
        // Load filter data
        b3.loadFilter(3, bitsize);
        b5.loadFilter(5, bitsize);
        b7.loadFilter(7, bitsize);
        b9.loadFilter(9, bitsize);
        b11.loadFilter(11, bitsize);
        b13.loadFilter(13, bitsize);
        b15.loadFilter(15, bitsize);
        b17.loadFilter(17, bitsize);
    }
    
    void extract(std::string& outDir, std::string& outName, std::string& imageDir, sampleDatabase& database) {
        
    }
    
private:
    // Filter information
    int bitsize;
    BSIFFilter b3;
    BSIFFilter b5;
    BSIFFilter b7;
    BSIFFilter b9;
    BSIFFilter b11;
    BSIFFilter b13;
    BSIFFilter b15;
    BSIFFilter b17;
};
