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
        
        mapString["Split directory"] = &splitDir;
        mapString["Training set filename"] = &trainingSetFilename;
        mapString["Testing set filename"] = &testingSetFilename;
        mapString["Database image directory"] = &databaseImageDir;
        mapString["Training sizes"] = &trainingSizes;
        mapString["Kernel type"] = &kernelType;
        
        
        mapString["Feature extraction destination file"] = &outputExtractionFilename;
        mapString["Feature extraction destination directory"] = &outputExtractionDir;
        mapString["Histogram output file"] = &outputHistFilename;
        mapString["Histogram output directory"] = &outputHistDir;
        mapString["Model output directory"] = &modelOutputDir;
        
        
        mapInt["Bitsize"] = &bitsize;
        
        mapDouble["Polynomial degree"] = &degree;
        mapDouble["Gamma"] = &gamma;
        
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
                        
                        // Option is type double
                        else if ( mapDouble.find(key) != mapDouble.end() )
                            *mapDouble[key] = osu.fromString<double>(value) ;
                        
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
        //if (testImages) {
        //    cout << "- Testing features will be stored in " << outputHistFilename  + ".csv" << endl;
        //}
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
          
            // Loop through modelSizes vector to train required models
            for (int i = 0; i < modelSizes.size(); i++) {
                std::cout << "BSIF size " << modelSizes[i] << "..." << endl;
                
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
                    svmPoly->setDegree(degree);
                    svmPoly->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
                    
                    svmPoly->train(featuresTrain, ROW_SAMPLE, classesTrain);
                    
                    svmPoly->save(modelOutputDir + generateFilename(modelSizes[i]));
                } else if (kernelType == "GAUSS") {
                    Ptr<SVM> svmGauss = SVM::create();
                    
                    // Set parameters
                    svmGauss->setType(SVM::C_SVC);
                    svmGauss->setKernel(SVM::RBF);
                    svmGauss->setGamma(gamma);
                    svmGauss->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
                    
                    svmGauss->train(featuresTrain, ROW_SAMPLE, classesTrain);
                    
                    svmGauss->save(modelOutputDir + generateFilename(modelSizes[i]));
                }
            }
            
            // Load from previously saved split (currently need to initialize with database, so create temp one)
            sampleDatabase noData(outputExtractionDir);
            testSeparator loadSets(noData);
            
            cv::Mat featuresTrain;
            cv::Mat classesTrain;
            string combined = outputExtractionDir + outputExtractionFilename;
            
            // Load features and classes from split
            loadSets.loadTraining(featuresTrain, classesTrain, setOutputDir, combined, bitsize, 3);
            
            
            // Create new SVM and train
            Ptr<SVM> svmPoly2 = SVM::create();
            svmPoly2->setType(SVM::C_SVC);
            svmPoly2->setKernel(SVM::POLY);
            svmPoly2->setDegree(2);
            svmPoly2->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
            svmPoly2->train(featuresTrain, ROW_SAMPLE, classesTrain);
            
            
            // Load test data
            cv::Mat featuresTest;
            cv::Mat classesTest;
            
            loadSets.loadTesting(featuresTest, classesTest, setOutputDir, combined, bitsize, 3);
            
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
            */
        }
        
        if (testImages) {
            std::cout << "Testing images..." << std::endl;
            
            // Need to load all models into vector
            std::vector<Ptr<SVM>> testingModels;
            
            // Loop through modelSize vector
            for (int i = 0; i < modelSizes.size(); i++) {
                // Check if file exists
                ifstream nextFile(modelOutputDir + generateFilename(modelSizes[i]));
                
                if (nextFile.good()) {
                    // Load SVM
                    Ptr<SVM> currentSVM = Algorithm::load<SVM>(modelOutputDir + generateFilename(modelSizes[i]));
                    
                    // Add to list of models
                    testingModels.push_back(currentSVM);
                    
                } else {
                    std::cout << "Model \"" << generateFilename(modelSizes[i]) << "\" not found." << endl;
                    std::cout << "Please make sure all models have been trained." << endl;
                }
            }
            
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
                if (textured > 0) {
                    overallResult.push_back(1);
                } else {
                    overallResult.push_back(0);
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
    std::string trainingSizes;
    std::vector<int> modelSizes;
    std::string kernelType;
    
    // Outputs
    std::string outputExtractionFilename;
    std::string outputExtractionDir;
    std::string outputHistFilename;
    std::string outputHistDir;
    std::string modelOutputDir;
    
    // Parameters
    int bitsize;
    double gamma;
    double degree;
    
    // Maps to associate a string (conf file) to a variable (pointer)
    std::map<std::string,bool*> mapBool;
    std::map<std::string,int*> mapInt;
    std::map<std::string,std::string*> mapString;
    std::map<std::string, double*> mapDouble;
    
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
        trainingSizes = "";
        kernelType = "";
        
        // Outputs
        outputExtractionFilename = "";
        outputExtractionDir = "";
        outputHistFilename = "";
        outputHistDir = "";
        modelOutputDir = "";
        
        // Parameters
        bitsize = 8;
        gamma = .25;
        degree = 2;
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
    
    std::string generateFilename(int size) {
        // Initialize new filename stringstream
        std::stringstream newFilename;
        
        if (kernelType == "POLY") {
            newFilename <<  "svm-BSIF-" << size << "-" << kernelType << "-" << "degree" << "-" << degree << ".xml";
        } else if (kernelType == "GAUSS") {
            newFilename <<  "svm-BSIF-" << size << "-" << kernelType << "-" << "gamma" << "-" << gamma << ".xml";
        }
        
        return newFilename.str();
    }
};

