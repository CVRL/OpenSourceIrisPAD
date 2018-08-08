//
//  TCLManager.cpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/17/18.

#include <stdio.h>
#include <map>
#include <opencv2/ml/ml.hpp>
#include "OsiStringUtils.h"
#include "sampleDatabase.hpp"
#include "featureExtractor.hpp"
#include "testSeparator.cpp"

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
        
        mapString["Split directory"] = &splitDir;
        mapString["Training set filename"] = &trainingSetFilename;
        mapString["Testing set filename"] = &testingSetFilename;
        mapString["Database image directory"] = &databaseImageDir;
        
        mapString["Feature extraction destination file"] = &outputExtractionFilename;
        mapString["Feature extraction destination directory"] = &outputExtractionDir;
        mapString["Histogram output file"] = &outputHistFilename;
        mapString["Histogram output directory"] = &outputHistDir;
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
            cout << "- Features will be stored in " << outputExtractionFilename << endl;
            cout << "=============" << endl;
        }
        if (testImages) {
            cout << "- Testing features will be stored in " << outputHistFilename  + ".csv" << endl;
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
            std::cout << "Training SVM..." << std::endl;
            
            // Load training data
            cv::Mat featuresTrain;
            cv::Mat classesTrain;
            
            loadTraining(featuresTrain, classesTrain, 3);
           
            // Load test data
            cv::Mat featuresTest;
            cv::Mat classesTest;
            
            loadTesting(featuresTest, classesTest, 3);
            
            // Create new SVM and train
            Ptr<SVM> svmPoly2 = SVM::create();
            svmPoly2->setType(SVM::C_SVC);
            svmPoly2->setKernel(SVM::POLY);
            svmPoly2->setDegree(2);
            svmPoly2->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
            svmPoly2->train(featuresTrain, ROW_SAMPLE, classesTrain);
            
            // Predict with SVM
            cv::Mat predictions2(classesTest.rows, classesTest.cols, CV_32SC1);
            float accuracy = 0;
            
            svmPoly2->predict(featuresTest, predictions2);
            for (int i = 0; i < classesTest.rows; i++) {
                if ((int)predictions2.at<float>(i,0) != classesTest.at<int>(i,0)) {
                    accuracy++;
                }
            }
            
            cout << "POLY-2" << endl;
            cout << "Number incorrect: " << accuracy << endl;
            accuracy /= classesTest.rows;
            cout << "Percentage incorrect: " << (accuracy * 100) << endl;
            
            
            //Create new SVM and train
            Ptr<SVM> svmPoly3 = SVM::create();
            svmPoly3->setType(SVM::C_SVC);
            svmPoly3->setKernel(SVM::POLY);
            svmPoly3->setDegree(3);
            svmPoly3->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
            svmPoly3->train(featuresTrain, ROW_SAMPLE, classesTrain);
            
            
            // Predict with SVM
            cv::Mat predictions3(classesTest.rows, classesTest.cols, CV_32SC1);
            accuracy = 0;
            
            svmPoly3->predict(featuresTest, predictions3);
            for (int i = 0; i < classesTest.rows; i++) {
                if ((int)predictions3.at<float>(i,0) != classesTest.at<int>(i,0)) {
                    accuracy++;
                }
            }
            cout << "POLY-3" << endl;
            cout << "Number incorrect: " << accuracy << endl;
            accuracy /= classesTest.rows;
            cout << "Percentage incorrect: " << (accuracy * 100) << endl;
            
            //Create new SVM and train
            Ptr<SVM> svmPoly4 = SVM::create();
            svmPoly4->setType(SVM::C_SVC);
            svmPoly4->setKernel(SVM::POLY);
            svmPoly4->setDegree(4);
            svmPoly4->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
            svmPoly4->train(featuresTrain, ROW_SAMPLE, classesTrain);
            svmPoly4->save(modelOutputDir + "savedSVM");
            
            // Predict with SVM
            cv::Mat predictions4(classesTest.rows, classesTest.cols, CV_32SC1);
            accuracy = 0;
            
            svmPoly4->predict(featuresTest, predictions4);
            for (int i = 0; i < classesTest.rows; i++) {
                if ((int)predictions4.at<float>(i,0) != classesTest.at<int>(i,0)) {
                    accuracy++;
                }
            }
            cout << "POLY-4" << endl;
            cout << "Number incorrect: " << accuracy << endl;
            accuracy /= classesTest.rows;
            cout << "Percentage incorrect: " << (accuracy * 100) << endl;
            
            //Create new SVM and train
            Ptr<SVM> svmPoly5 = SVM::create();
            svmPoly5->setType(SVM::C_SVC);
            svmPoly5->setKernel(SVM::POLY);
            svmPoly5->setDegree(5);
            svmPoly5->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
            svmPoly5->train(featuresTrain, ROW_SAMPLE, classesTrain);
            
            
            // Predict with SVM
            cv::Mat predictions5(classesTest.rows, classesTest.cols, CV_32SC1);
            accuracy = 0;
            
            svmPoly5->predict(featuresTest, predictions5);
            for (int i = 0; i < classesTest.rows; i++) {
                if ((int)predictions5.at<float>(i,0) != classesTest.at<int>(i,0)) {
                    accuracy++;
                }
            }
            cout << "POLY-5" << endl;
            cout << "Number incorrect: " << accuracy << endl;
            accuracy /= classesTest.rows;
            cout << "Percentage incorrect: " << (accuracy * 100) << endl;
            
            //Create new SVM and train
            Ptr<SVM> svmGauss = SVM::create();
            svmGauss->setType(SVM::C_SVC);
            svmGauss->setKernel(SVM::RBF);
            svmGauss->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
            svmGauss->train(featuresTrain, ROW_SAMPLE, classesTrain);
            
            
            // Predict with SVM
            cv::Mat predictions6(classesTest.rows, classesTest.cols, CV_32FC1);
            accuracy = 0;
            
            svmGauss->predict(featuresTest, predictions6);
            for (int i = 0; i < classesTest.rows; i++) {
                if ((int)predictions6.at<float>(i,0) != classesTest.at<int>(i,0)) {
                    accuracy++;
                }
            }
            cout << "RBF" << endl;
            cout << "Number incorrect: " << accuracy << endl;
            accuracy /= classesTest.rows;
            cout << "Percentage incorrect: " << (accuracy * 100) << endl;
            
        }
        
        if (testImages) {
            std::cout << "Testing images..." << std::endl;
        }
    }
    
private:
    
    // Commands
    bool extractFeatures;
    bool trainModel;
    bool testImages;
    
    
    // Inputs
    std::string databaseImageDir;
    std::string splitDir;
    std::string trainingSetFilename;
    std::string testingSetFilename;
    
    // Outputs
    std::string outputExtractionFilename;
    std::string outputExtractionDir;
    std::string outputHistFilename;
    std::string outputHistDir;
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
        
        // Inputs
        databaseImageDir = "";
        splitDir = "";
        trainingSetFilename = "";
        testingSetFilename = "";
        
        // Outputs
        outputExtractionFilename = "";
        outputExtractionDir = "";
        outputHistFilename = "";
        outputHistDir = "";
        modelOutputDir = "";
        
        // Parameters
        bitsize = 8;
    }
    
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
                // Increment the output features counter
                i++;
            }
            // Increment the CSV line
            featureCSV++;
        }
        
        // Load classifications
        outputLabels.create((int)testingClass.size(), 1, CV_32SC1);
        
        for (int j = 0; j < testingClass.size(); j++) {
            outputLabels.at<int>(j, 0) = trainingClass[j];
        }
        
    }
};

