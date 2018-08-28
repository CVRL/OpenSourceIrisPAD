//
//  TCLManager.cpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/17/18.

#include <stdio.h>
#include <map>
#include <opencv2/ml/ml.hpp>
#include "OsiStringUtils.h"
#include "featureExtractor.hpp"
#include "CSVIterator.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

class TCLManager {
public:
    TCLManager(void) {
        // Associate lines of config file to attributes
        mapBool["Extract features"] = &extractFeatures;
        mapBool["Train model"] = &trainModel;
        mapBool["Test images"] = &testImages;
        mapBool["Majority voting"] = &majorityVoting;
        
        mapString["Split directory"] = &splitDir;
        mapString["Training set filename"] = &trainingSetFilename;
        mapString["Testing set filename"] = &testingSetFilename;
        mapString["Database image directory"] = &databaseImageDir;
        mapString["Training sizes"] = &trainingSizes;
        mapString["Kernel type"] = &kernelType;
        mapString["Parameters"] = &parameterString;
        
        
        mapString["Feature extraction destination file"] = &outputExtractionFilename;
        mapString["Feature extraction destination directory"] = &outputExtractionDir;
        mapString["Model output directory"] = &modelOutputDir;
        
        
        mapInt["Bitsize"] = &bitsize;
        
        // Initialize
        initConfig();
    }
    
    void loadConfig(const std::string& Filename) {
        // Open the file
        std::ifstream configFile(Filename, std::ifstream::in);
        
        if ( ! configFile.good() )
            std::cout << "Error: Cannot read configuration file." << std::endl;
            //throw runtime_error("Cannot read configuration file " + rFilename ) ;
        
        // String utilities
        osiris::OsiStringUtils osu;
        
        // Loop on lines
        while ( configFile.good() && ! configFile.eof() )
        {
            // Get the new line
            string line ;
            getline(configFile,line) ;
            
            // Filter out comments
            if ( ! line.empty() )
            {
                int pos = (int)line.find('#') ;
                if ( pos != string::npos )
                    line = line.substr(0,pos) ;
            }
            
            // Split line into key and value
            if ( ! line.empty() )
            {
                int pos = (int)line.find("=") ;
                
                if ( pos != string::npos )
                {
                    // Trim key and value
                    string key = osu.trim(line.substr(0,pos)) ;
                    string value = osu.trim(line.substr(pos+1)) ;
                    
                    if ( ! key.empty() && ! value.empty() )
                    {
                        // Option is type bool
                        if ( mapBool.find(key) != mapBool.end() )
                            *mapBool[key] = osu.fromString<bool>(value) ;
                        
                        // Option is type ints
                        else if ( mapInt.find(key) != mapInt.end() )
                            *mapInt[key] = osu.fromString<int>(value) ;
                        
                        // Option is type string
                        else if ( mapString.find(key) != mapString.end() )
                            *mapString[key] = osu.convertSlashes(value) ;
                        
                        // Option is not stored in any mMap
                        else
                            cout << "Unknown option in configuration file : " << line << endl ;
                    }
                }
            }
        }
        
        // Determine models from trainingSizes string
        std::stringstream modelStream(trainingSizes);
        
        std::string currentNum;
        
        while (getline(modelStream, currentNum, ',')) {
            modelSizes.push_back(stoi(currentNum));
        }
        
        // Determine model parameters from parameterString
        std::stringstream parameterStream(parameterString);
        
        while (getline(parameterStream, currentNum, ',')) {
            modelParameter.push_back(stod(currentNum));
        }
        
    }
    
    void showConfig(void) {
        cout << "=============" << endl ;
        cout << "Configuration" << endl ;
        cout << "=============" << endl ;
        cout << endl;
        
        
        cout << "- Process : " ;
        if (extractFeatures) {
            cout << "| Extract features | ";
        }
        if (trainModel) {
            cout << "| Train model | ";
        }
        if (testImages) {
            cout << "| Test images | ";
        }
        cout << endl;
        
        
        if (extractFeatures) {
            cout << "=============" << endl;
            cout << "- Features will be stored in directory: " << outputExtractionDir << endl;
            cout << "- Feature filenames will be in format: " << outputExtractionFilename + "_filter_size_size_bits.csv" << endl;
            cout << "=============" << endl;
        }
        
        if (trainModel) {
            cout << "=============" << endl;
            cout << "-Models to be trained: " << endl;
            for (int i = 0; i < modelSizes.size(); i++) {
                cout << "   " << generateFilename(i) << endl;
            }
            cout << "=============" << endl;
        }
        
        if (testImages) {
            cout << "=============" << endl;
            if (majorityVoting) {
                cout << "- Majority voting will be used to determine result from multiple models" << endl;
            } else {
                cout << "- Each model will be tested separately" << endl;
            }
            cout << "-Models to be tested: " << endl;
            for (int i = 0; i < modelSizes.size(); i++) {
                cout << "   " << generateFilename(i) << endl;
            }
            cout << "=============" << endl;
        }
    }
    
    void run(void) {
        cout << endl ;
        cout << "================" << endl ;
        cout << "Start processing" << endl ;
        cout << "================" << endl ;
        cout << endl ;
        
        loadSets();
       
        if (extractFeatures) {
            std::cout << "Extracting features..." << std::endl;
            
            // Declare new feature extractor
            featureExtractor newExtractor(bitsize);
            
            // Extract
            newExtractor.extract(outputExtractionDir, outputExtractionFilename, databaseImageDir, trainingSet, testingSet);
            
        }
        
        
        if (trainModel) {
            std::cout << "Training SVM..." << endl << endl;
          
            // Loop through modelSizes vector to train required models
            for (int i = 0; i < modelSizes.size(); i++) {
                std::cout << "Training model " << (i + 1) << " out of " << modelSizes.size() << "..." << endl;
                std::cout << "  BSIF size " << modelSizes[i] << endl;
                std::cout << "  Model type " << kernelType << endl;
                std::cout << "  Model parameter " << modelParameter[i] << endl;
                
                // Load training data for current size
                cv::Mat featuresTrain;
                cv::Mat classesTrain;
                
                loadTraining(featuresTrain, classesTrain, modelSizes[i]);
                
                
                // Create new SVM and train
                if (kernelType == "POLY") {
                    Ptr<SVM> svmPoly = SVM::create();
                    
                    // Set parameters
                    svmPoly->setType(SVM::C_SVC);
                    svmPoly->setKernel(SVM::POLY);
                    svmPoly->setDegree(modelParameter[i]);
                    svmPoly->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
                    
                    svmPoly->train(featuresTrain, ROW_SAMPLE, classesTrain);
                    
                    svmPoly->save(modelOutputDir + generateFilename(i));
                } else if (kernelType == "GAUSS") {
                    Ptr<SVM> svmGauss = SVM::create();
                    
                    // Set parameters
                    svmGauss->setType(SVM::C_SVC);
                    svmGauss->setKernel(SVM::RBF);
                    svmGauss->setGamma(modelParameter[i]);
                    svmGauss->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
                    
                    svmGauss->train(featuresTrain, ROW_SAMPLE, classesTrain);
                    
                    svmGauss->save(modelOutputDir + generateFilename(i));
                }
            }
        }
        
        if (testImages) {
            std::cout << "Testing images..." << endl << endl;
            
            // Need to load all models into vector
            std::vector<Ptr<SVM>> testingModels;
            
            // Loop through modelSize vector
            for (int i = 0; i < modelSizes.size(); i++) {
                // Check if file exists
                ifstream nextFile(modelOutputDir + generateFilename(i));
                
                if (nextFile.good()) {
                    // Load SVM
                    Ptr<SVM> currentSVM = Algorithm::load<SVM>(modelOutputDir + generateFilename(i));
                    
                    // Add to list of models
                    testingModels.push_back(currentSVM);
                    
                } else {
                    std::cout << "Model \"" << generateFilename(i) << "\" not found." << endl;
                    std::cout << "Please make sure all models have been trained." << endl;
                }
            }
            
            if (majorityVoting) {
                // Test performance (majority voting)
                // Mat objects for testing data
                cv::Mat featuresTest;
                cv::Mat classesTest;
                
                // Results vector
                vector<cv::Mat> results;
                
                for (int i = 0; i < testingModels.size(); i++) {
                    // Load testing features
                    loadTesting(featuresTest, classesTest, modelSizes[i]);
                    
                    // New Mat for results
                    cv::Mat individualResults(classesTest.rows, classesTest.cols, CV_32FC1);
                    
                    // Predict using model
                    testingModels[i]->predict(featuresTest, individualResults);
                    
                    // Add results to results vector
                    results.push_back(individualResults);
                }
                
                // Perform majority voting
                vector<int> overallResult;
                
                int numIncorrect = 0;
                
                for (int i = 0; i < testingClass.size(); i++) {
                    // Variables to count number of votes for or against
                    int textured = 0;
                    int other = 0;
                    
                    // Loop through results vector
                    for (int j = 0; j < results.size(); j++) {
                        if (results[j].at<float>(i,0) == 1) {
                            textured++;
                        } else {
                            other++;
                        }
                    }
                    
                    // Determine result by majority voting
                    if (textured > other) {
                        overallResult.push_back(1);
                    } else if (other > textured){
                        overallResult.push_back(0);
                    } else {
                        overallResult.push_back(1);
                    }
                    
                    // Determine if incorrect
                    if (overallResult[i] != testingClass[i]) {
                        numIncorrect++;
                    }
                }
                
                // Output accuracy
                cout << "The total number incorrect is: " << numIncorrect << endl;
                
                float ccr = 100 - ((float)numIncorrect / testingClass.size()) * 100;
                cout << "CCR: " << ccr << endl;
            } else {
                // Use each model separately
                // Mat objects for testing data
                cv::Mat featuresTest;
                cv::Mat classesTest;
                
                for (int i = 0; i < testingModels.size(); i++) {
                    // Load testing features
                    loadTesting(featuresTest, classesTest, modelSizes[i]);
                    
                    // New Mat for results
                    cv::Mat individualResults(classesTest.rows, classesTest.cols, CV_32FC1);
                    
                    // Predict using model
                    testingModels[i]->predict(featuresTest, individualResults);
                    
                    // Determine number incorrect
                    int numIncorrect = 0;
                    
                    for (int j = 0; j < testingClass.size(); j++) {
                        if (individualResults.at<float>(j,0) != testingClass[j]) {
                            numIncorrect++;
                        }
                    }
                    
                    // Output accuracy
                    cout << "Model: " << generateFilename(i) << endl;
                    cout << "The total number incorrect is: " << numIncorrect << endl;
                    
                    float ccr = 100 - ((float)numIncorrect / testingClass.size()) * 100;
                    cout << "CCR: " << ccr << endl << endl;
                    
                }
            }
         }
    }
    
private:
    
    // Commands
    bool extractFeatures;
    bool trainModel;
    bool testImages;
    bool majorityVoting;
    
    
    // Inputs
    std::string databaseImageDir;
    std::string splitDir;
    std::string trainingSetFilename;
    std::string testingSetFilename;
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
    
    // Maps to associate a string (conf file) to a variable (pointer)
    std::map<std::string,bool*> mapBool;
    std::map<std::string,int*> mapInt;
    std::map<std::string,std::string*> mapString;
    
    // List of filenames for each set
    vector<string> trainingSet;
    vector<string> testingSet;
    
    // List of classifications for each set
    vector<int> trainingClass;
    vector<int> testingClass;
    
    
    // Initialize all parameters
    void initConfig(void) {
        // Commands
        extractFeatures = false;
        trainModel = false;
        testImages = false;
        majorityVoting = false;
        
        // Inputs
        databaseImageDir = "";
        splitDir = "";
        trainingSetFilename = "";
        testingSetFilename = "";
        trainingSizes = "";
        kernelType = "";
        parameterString = "";
        
        // Outputs
        outputExtractionFilename = "";
        outputExtractionDir = "";
        modelOutputDir = "";
        
        // Parameters
        bitsize = 8;
    }
    
    // Loads the training/testing image names indicated in the config file
    void loadSets(void) {
        string currentName;
        size_t location;
        
        // Training sets
        ifstream train;
        train.open(splitDir + trainingSetFilename);
        
        while (getline(train, currentName)) {
            location = currentName.find(",");
            trainingSet.push_back(currentName.substr(0, location));
            trainingClass.push_back(stoi(currentName.substr((location + 1))));
        }
        train.close();
        
        // Testing sets
        ifstream test;
        test.open(splitDir + testingSetFilename);
        currentName = "";
        
        while (getline(test, currentName)) {
            location = currentName.find(",");
            testingSet.push_back(currentName.substr(0, location));
            testingClass.push_back(stoi(currentName.substr((location + 1))));
        }
        test.close();
        
    }
    
    
    // Loads training features and labels into a Mat object
    void loadTraining(cv::Mat& outputFeatures, cv::Mat& outputLabels, int filtersize) {
        
            // Allocate storage for the output features (rows = number of samples, columns = size of histogram)
            outputFeatures.create((int)trainingSet.size(), pow(2,bitsize), CV_32FC1);
            
            // Load features
            stringstream featureFilename;
            featureFilename << outputExtractionDir << outputExtractionFilename << "_filter_" << filtersize << "_" << filtersize << "_" << bitsize << ".csv";
            ifstream featureFile(featureFilename.str());
            CSVIterator featureCSV(featureFile);
            
            int i = 0;
            
            while (featureCSV != CSVIterator()) {
                // If the current line includes a file from the testing set
                if (find(trainingSet.begin(), trainingSet.end(),(*featureCSV)[0]) != trainingSet.end()) {
                    // Start at 1 to ignore filename column
                    for (int j = 1; j < (*featureCSV).size(); j++) {
                        outputFeatures.at<float>(i, (j-1)) = stoi((*featureCSV)[j]);
                    }
                    // Normalize
                    cv::Scalar mean;
                    cv::Scalar stddev;
                    
                    meanStdDev(outputFeatures.row(i), mean, stddev);
                    
                    for (int j = 1; j < (*featureCSV).size(); j++) {
                        outputFeatures.at<float>(i, (j-1)) = (outputFeatures.at<float>(i, (j-1)) - mean[0]) / stddev[0];
                    }
                    
                    // Increment the output features counter
                    i++;
                }
                // Increment the CSV line
                featureCSV++;
            }
        
            // Load classifications
            outputLabels.create((int)trainingClass.size(), 1, CV_32SC1);
            
            for (int j = 0; j < trainingClass.size(); j++) {
                outputLabels.at<int>(j, 0) = trainingClass[j];
            }
            
        }
    
    
    // Loads testing features and labels into a Mat object
    void loadTesting(cv::Mat& outputFeatures, cv::Mat& outputLabels, int filtersize) {
        
        // Allocate storage for the output features (rows = number of samples, columns = size of histogram)
        outputFeatures.create((int)testingSet.size(), pow(2,bitsize), CV_32FC1);
        
        // Load features
        stringstream featureFilename;
        featureFilename << outputExtractionDir << outputExtractionFilename << "_filter_" << filtersize << "_" << filtersize << "_" << bitsize << ".csv";
        ifstream featureFile(featureFilename.str());
        CSVIterator featureCSV(featureFile);
        
        int i = 0;
        
        while (featureCSV != CSVIterator()) {
            // If the current line includes a file from the testing set
            if (find(testingSet.begin(), testingSet.end(),(*featureCSV)[0]) != testingSet.end()) {
                // Start at 1 to ignore filename column
                for (int j = 1; j < (*featureCSV).size(); j++) {
                    outputFeatures.at<float>(i, (j-1)) = stoi((*featureCSV)[j]);
                }
                // Normalize
                cv::Scalar mean;
                cv::Scalar stddev;
                
                meanStdDev(outputFeatures.row(i), mean, stddev);
                
                for (int j = 1; j < (*featureCSV).size(); j++) {
                    outputFeatures.at<float>(i, (j-1)) = (outputFeatures.at<float>(i, (j-1)) - mean[0]) / stddev[0];
                }
                
                // Increment the output features counter
                i++;
            }
            
            // Increment the CSV line
            featureCSV++;
        }
        
        // Load classifications
        outputLabels.create((int)testingClass.size(), 1, CV_32SC1);
        
        for (int j = 0; j < testingClass.size(); j++) {
            outputLabels.at<int>(j, 0) = testingClass[j];
        }
        
    }
    
    std::string generateFilename(int i) {
        // Initialize new filename stringstream
        std::stringstream newFilename;
        
        if (kernelType == "POLY") {
            newFilename <<  "svm-BSIF-" << modelSizes[i] << "-" << kernelType << "-" << "degree" << "-" << modelParameter[i] << ".xml";
        } else if (kernelType == "GAUSS") {
            newFilename <<  "svm-BSIF-" << modelSizes[i] << "-" << kernelType << "-" << "gamma" << "-" << modelParameter[i] << ".xml";
        }
        
        return newFilename.str();
    }
};

