//
//  TCLManager.h
//  TCLDetection
//
//  Created by Joseph McGrath on 9/6/18.


#ifndef TCLManager_h
#define TCLManager_h

#include <map>
#include <opencv2/ml/ml.hpp>
#include "featureExtractor.hpp"

#define TRAIN 0
#define VALID 1
#define TEST 2

class TCLManager
{
public:
    TCLManager(void);
    
    void loadConfig(const std::string& Filename);
    
    void showConfig(void);
    
    void run(void);
    
private:
    // Commands
    bool extractFeatures;
    bool trainModel;
    bool testImages;
    bool majorityVoting;
    
    
    // Inputs
    std::string imageDir;
    std::string splitDir;
    std::string trainingSetFilename;
    std::string testingSetFilename;
    std::string validationSetFilename;
    std::string trainingSizes;
    std::vector<int> modelSizes;
    std::string kernelType;
    std::string parameterString;
    std::vector<double> modelParameter;
    
    
    // Outputs
    std::string outputExtractionFilename;
    std::string outputExtractionDir;
    std::string modelOutputDir;
    
    // Parameters
    int bitsize;
    
    // Maps to associate a string (config file) to a variable (pointer)
    std::map<std::string,bool*> mapBool;
    std::map<std::string,int*> mapInt;
    std::map<std::string,std::string*> mapString;
    
    // List of filenames for each set
    std::vector<std::string> trainingSet;
    std::vector<std::string> testingSet;
    std::vector<std::string> validationSet;
    
    // List of classifications for each set
    std::vector<int> trainingClass;
    std::vector<int> testingClass;
    std::vector<int> validationClass;
    
    void initConfig(void);
    
    void loadSets(void);
    
    void loadFeatures(cv::Mat& outputFeatures, cv::Mat& outputLabels, int filtersize, int setType);
    
    std::string generateFilename(int i);
    
};

#endif /* TCLManager_h */
