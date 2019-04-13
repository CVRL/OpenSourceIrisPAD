//
//  TCLManager.h
//  TCLDetection



#ifndef TCLManager_h
#define TCLManager_h

#include <map>
#include "opencv2/ml.hpp"
#include "featureExtractor.hpp"
#include "opencv2/core.hpp"
#include "hdf5.h"

#define TRAIN 0
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
    std::string segmentationType;
    std::string modelString;
    std::vector<std::string> modelTypes;
    
    
    // Inputs
    std::string imageDir;
    std::string splitDir;
    std::string trainingSetFilename;
    std::string testingSetFilename;
    std::string trainingSizes;
    std::vector<int> modelSizes;
    std::string bitString;
    std::vector<int> bitSizes;
    
    
    // Outputs
    std::string outputExtractionFilename;
    std::string outputExtractionDir;
    std::string modelOutputDir;
    
    // Maps to associate a string (config file) to a variable (pointer)
    std::map<std::string,bool*> mapBool;
    std::map<std::string,int*> mapInt;
    std::map<std::string,std::string*> mapString;
    
    // List of filenames for each set
    std::vector<std::string> trainingSet;
    std::vector<std::string> testingSet;
    
    // List of classifications for each set
    std::vector<int> trainingClass;
    std::vector<int> testingClass;
    
    void initConfig(void);
    
    void loadSets(void);
    
    void trainAuto_rf(cv::Ptr<cv::ml::TrainData>& trainData, cv::Ptr<cv::ml::RTrees> model);
    
    void trainAuto_mlp(cv::Ptr<cv::ml::TrainData>& data, cv::Ptr<cv::ml::ANN_MLP> model);
    
    void loadFeatures(cv::Mat& outputFeatures, cv::Mat& outputLabels, int filtersize, int setType, int bitType);
    
    std::string generateFilename(int i);
    
};

static inline void setRangeVector(std::vector<int>& vec, int n);

#endif /* TCLManager_h */
