//
//  TCLManager.cpp
//  TCLDetection


#include "TCLManager.hpp"
#include "tclUtil.h"


using namespace std;
using namespace cv;
using namespace cv::ml;


/* Manager structure based on OSIRIS: The method for reading in a configuration file and the basic flow from configuration to showing configuration to running are both based on a similar structure within OSIRIS. */

// Default constructor
TCLManager::TCLManager(void)
{
    
    // Associate lines of config file to attributes
    mapBool["Extract features"] = &extractFeatures;
    mapBool["Train model"] = &trainModel;
    mapBool["Test images"] = &testImages;
    mapBool["Majority voting"] = &majorityVoting;
    
    mapString["Image directory"] = &imageDir;
    mapString["CSV directory"] = &splitDir;
    mapString["Training set filename"] = &trainingSetFilename;
    mapString["Testing set filename"] = &testingSetFilename;
    mapString["Sizes"] = &trainingSizes;
    
    
    mapString["Feature extraction destination file"] = &outputExtractionFilename;
    mapString["Feature extraction destination directory"] = &outputExtractionDir;
    mapString["Model directory"] = &modelOutputDir;
    
    
    mapInt["Bitsize"] = &bitsize;
    
    // Initialize
    initConfig();
    
}





// Load configuration from config file
void TCLManager::loadConfig(const std::string& Filename)
{
    
    // Open the file
    std::ifstream configFile(Filename, std::ifstream::in);
    
    if ( ! configFile.good() )
        
        throw runtime_error("Error: Cannot read configuration file " + Filename ) ;
    
    // String utilities
    tclStringUtil tsu;
    
    // Loop on lines
    while ( configFile.good() && ! configFile.eof() )
    {
        // Get the new line
        string line ;
        getline(configFile,line) ;
        
        // Filter out comments
        if ( ! line.empty() )
        {
            unsigned long pos = line.find('#') ;
            if ( pos != string::npos )
                line = line.substr(0,pos) ;
        }
        
        // Split line into key and value
        if ( ! line.empty() )
        {
            unsigned long pos = (int)line.find("=") ;
            
            if ( pos != string::npos )
            {
                // Trim key and value
                string key = tsu.trim(line.substr(0,pos)) ;
                string value = tsu.trim(line.substr(pos+1)) ;
                
                if ( ! key.empty() && ! value.empty() )
                {
                    // Option is type bool
                    if ( mapBool.find(key) != mapBool.end() )
                        *mapBool[key] = tsu.fromString<bool>(value) ;
                    
                    // Option is type ints
                    else if ( mapInt.find(key) != mapInt.end() )
                        *mapInt[key] = tsu.fromString<int>(value) ;
                    
                    // Option is type string
                    else if ( mapString.find(key) != mapString.end() )
                        *mapString[key] = value ;
                    
                    // Option is not stored in any mMap
                    else
                        throw runtime_error("Error: Unknown option " + line + " in configuration file");
                }
            }
        }
    }
    
    // Determine models from trainingSizes string
    std::stringstream modelStream(trainingSizes);
    
    std::string currentNum;
    
    while (getline(modelStream, currentNum, ','))
    {
        modelSizes.push_back(stoi(currentNum));
    }
    
}





// Display the configuration to the user
void TCLManager::showConfig(void)
{
    
    cout << "=============" << endl ;
    cout << "Configuration" << endl ;
    cout << "=============" << endl ;
    cout << endl;
    
    
    cout << "- Process : " ;
    if (extractFeatures)
    {
        cout << "| Extract features | ";
    }
    if (trainModel)
    {
        cout << "| Train model | ";
    }
    if (testImages)
    {
        cout << "| Test images | ";
    }
    cout << endl;
    
    
    if (extractFeatures)
    {
        cout << "=============" << endl;
        cout << "- Features will be stored in directory: " << outputExtractionDir << endl;
        cout << "- Feature filenames will be in format: " << outputExtractionFilename + "_filter_size_size_bits.csv" << endl;
        cout << "=============" << endl;
    }
    
    if (trainModel)
    {
        cout << "=============" << endl;
        cout << "- The training set will be: " << splitDir << trainingSetFilename << endl;
        cout << "- Models to be trained: " << endl;
        for (int i = 0; i < (int)modelSizes.size(); i++)
        {
            cout << "   " << generateFilename(i) << endl;
        }
        cout << "=============" << endl;
    }
    
    if (testImages)
    {
        cout << "=============" << endl;
        if (majorityVoting)
        {
            cout << "- Majority voting will be used to determine result from multiple models" << endl;
            cout << "- In the case of a tie, a random decision will be made" << endl;
        }
        else
        {
            cout << "- Models will be tested separately" << endl;
        }
        cout << "- The testing set will be: " << splitDir << testingSetFilename << endl;
        cout << "- Models to be tested: " << endl;
        for (int i = 0; i < (int)modelSizes.size(); i++)
        {
            cout << "   " << generateFilename(i) << endl;
        }
        cout << "=============" << endl;
    }
    
}





// Perform functions according to settings in config file
void TCLManager::run(void)
{
    
    cout << endl ;
    cout << "================" << endl ;
    cout << "Start processing" << endl ;
    cout << "================" << endl ;
    cout << endl ;
    
    try
    {
        loadSets();
    }
    catch (runtime_error& e)
    {
        throw e;
    }


    if (extractFeatures)
    {
        
        std::cout << "Extracting features..." << std::endl;
        
        // Verify bitsize (program has only been tested with bitsize 8 as of now)
        if (bitsize != 8)
        {
            throw runtime_error("Error: Please use bitsize 8 for tested results.");
        }
        
        // Concatenate lists of files
        std::vector<std::string> extractionFilenames;
        extractionFilenames.insert(extractionFilenames.end(), trainingSet.begin(), trainingSet.end());
        extractionFilenames.insert(extractionFilenames.end(), testingSet.begin(), testingSet.end());
        
        // Declare new feature extractor
        featureExtractor newExtractor(bitsize, extractionFilenames);
        
        // Extract
        try
        {
            newExtractor.extract(outputExtractionDir, outputExtractionFilename, imageDir);
        }
        catch (runtime_error& e)
        {
            throw e;
        }
        
    }
    
    
    if (trainModel)
    {
        
        std::cout << "Training SVM..." << endl << endl;
        
        // Loop through modelSizes vector to train required models
        for (int i = 0; i < (int)modelSizes.size(); i++)
        {
            std::cout << "Training model " << (i + 1) << " out of " << modelSizes.size() << "..." << endl;
            std::cout << "  BSIF size " << modelSizes[i] << " | Kernel RBF " << endl;
            
            // Load training data for current size
            cv::Mat featuresTrain;
            cv::Mat classesTrain;
            
            try
            {
                loadFeatures(featuresTrain, classesTrain, modelSizes[i], TRAIN);
            }
            catch (runtime_error& e)
            {
                throw e;
            }
            
            
            // Place into trainData
            Ptr<TrainData> trainingData = TrainData::create(featuresTrain, ROW_SAMPLE, classesTrain);
            
            
            // Create new SVM
            Ptr<SVM> svmGauss = SVM::create();
            
            // Set parameters
            svmGauss->setType(SVM::C_SVC);
            svmGauss->setKernel(SVM::RBF);
            
            
            // Train model and save
            svmGauss->trainAuto(trainingData);
            svmGauss->save(modelOutputDir + generateFilename(i));
            
        }
    }
    
    if (testImages)
    {
        std::cout << "Testing images..." << endl << endl;
        
        // Need to load all models into vector
        std::vector< Ptr<SVM> > testingModels;
        
        // Loop through modelSize vector
        for (int i = 0; i < (int)modelSizes.size(); i++)
        {
            // Check if file exists
            ifstream nextFile(modelOutputDir + generateFilename(i));
            
            if (nextFile.good())
            {
                // Load SVM
                Ptr<SVM> currentSVM = Algorithm::load<SVM>(modelOutputDir + generateFilename(i));
                
                // Add to list of models
                testingModels.push_back(currentSVM);
                
            }
            else
            {
                throw runtime_error("Error: Model \"" + generateFilename(i) + "\" not found.");
            }
        }
        
        if (majorityVoting)
        {
            // Test performance (majority voting)
            
            // Mat objects for testing data
            cv::Mat featuresTest;
            cv::Mat classesTest;
            
            // Results vector
            vector<cv::Mat> results;
            
            for (int i = 0; i < (int)testingModels.size(); i++)
            {
                // Load testing features
                try
                {
                loadFeatures(featuresTest, classesTest, modelSizes[i], TEST);
                }
                catch (runtime_error& e)
                {
                    throw e;
                }
                
                
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
            
            for (int i = 0; i < (int)testingClass.size(); i++)
            {
                // Variables for majority voting division
                int inFavor = 0;
                int against = 0;
                
                for (int j = 0; j < (int)testingModels.size(); j++)
                {
                    if (results[j].at<float>(i,0) == 1) {
                        inFavor++;
                    } else {
                        against++;
                    }
                }
                
                if (inFavor > against)
                {
                    overallResult.push_back(1);
                }
                else if (against > inFavor)
                {
                    overallResult.push_back(0);
                }
                else
                {
                    // In the case of a tie, choose randomly (0 or 1)
                    overallResult.push_back((rand() % 2));
                }
                
                // Determine if incorrect
                if (overallResult[i] != classesTest.at<int>(i,0))
                {
                    numIncorrect++;
                }
            }
            
            // Output accuracy
            float ccr = 100 - (((float)numIncorrect / classesTest.rows) * 100);
            
            cout << "The total number incorrect is: " << numIncorrect << endl;
            cout << "CCR: " << ccr << endl;
            
        }
        else
        {
            // Use each model separately
            // Mat objects for testing data
            cv::Mat featuresTest;
            cv::Mat classesTest;
            
            for (int i = 0; i < (int)testingModels.size(); i++)
            {
                // Load testing features
                try
                {
                    loadFeatures(featuresTest, classesTest, modelSizes[i], TEST);
                }
                catch (runtime_error& e)
                {
                    throw e;
                }
                
                // New Mat for results
                cv::Mat individualResults(classesTest.rows, classesTest.cols, CV_32FC1);
                
                // Predict using model
                testingModels[i]->predict(featuresTest, individualResults);
                
                // Determine number incorrect
                int numIncorrect = 0;
                
                for (int j = 0; j < classesTest.rows; j++)
                {
                    if (individualResults.at<float>(j,0) != classesTest.at<int>(j,0))
                    {
                        numIncorrect++;
                    }
                }
                
                // Output accuracy
                float ccr = 100 - ((float)numIncorrect / classesTest.rows) * 100;
                
                cout << "Model: " << generateFilename(i) << endl;
                cout << "The total number incorrect is: " << numIncorrect << endl;
                cout << "CCR: " << ccr << endl << endl;
                
            }
        }
    }
}





// Initialize all parameters
void TCLManager::initConfig(void)
{
    
    // Commands
    extractFeatures = false;
    trainModel = false;
    testImages = false;
    majorityVoting = false;
    
    // Inputs
    imageDir = "";
    splitDir = "";
    trainingSetFilename = "";
    testingSetFilename = "";
    trainingSizes = "";
    
    // Outputs
    outputExtractionFilename = "";
    outputExtractionDir = "";
    modelOutputDir = "";
    
    // Parameters
    bitsize = 8;
    
}





// Loads the training/testing image names indicated in the config file
void TCLManager::loadSets(void)
{
    string currentName;
    size_t location;
    
    // Training sets
    if (trainingSetFilename != "")
    {
        
        ifstream train;
        train.open(splitDir + trainingSetFilename);
        
        if (! train.good())
        {
            throw runtime_error("Error: training split not found in " + trainingSetFilename);
        }
        
        while (getline(train, currentName))
        {
            location = currentName.find(",");
            trainingSet.push_back(currentName.substr(0, location));
            trainingClass.push_back(stoi(currentName.substr((location + 1))));
        }
        
        train.close();
        
    } else if (trainModel)
    {
        // if model training is requested but no file is given
        throw runtime_error("Error: please specify a list of images for training (training set filename)");
    }
    
    // Testing sets
    if (testingSetFilename != "")
    {
        
        ifstream test;
        test.open(splitDir + testingSetFilename);
        currentName = "";
        
        if (! test.good())
        {
            throw runtime_error("Error: testing split not found in " + testingSetFilename);
        }
        
        while (getline(test, currentName))
        {
            location = currentName.find(",");
            testingSet.push_back(currentName.substr(0, location));
            testingClass.push_back(stoi(currentName.substr((location + 1))));
        }
        test.close();
        
    } else if (testImages)
    {
        // if image testing is requested but no file is given
        throw runtime_error("Error: please specify a list of images for testing (testing set filename)");
    }
    
}





// Loads features for training or testing sets into Mat objects
void TCLManager::loadFeatures(cv::Mat& outputFeatures, cv::Mat& outputLabels, int filtersize, int setType)
{
    
    // Determine set types
    vector<string>* fileSet;
    vector<int>* classSet;
    
    switch (setType)
    {
        case TRAIN:
            fileSet = &trainingSet;
            classSet = &trainingClass;
            break;
        case TEST:
            fileSet = &testingSet;
            classSet = &testingClass;
            break;
        default:
            fileSet = NULL;
            classSet = NULL;
            throw runtime_error("Error: Invalid set type");
    }
    
    // Allocate storage for the output features (rows = number of samples, columns = size of histogram) and for output labels
    outputFeatures.create((int)(*fileSet).size(), pow(2,bitsize), CV_32FC1);
    outputLabels.create((int)(*classSet).size(), 1, CV_32SC1);
    
    // Load features
    stringstream featureFilename;
    featureFilename << outputExtractionDir << outputExtractionFilename << "_filter_" << filtersize << "_" << filtersize << "_" << bitsize << ".csv";
    ifstream featureFile(featureFilename.str());
    CSVIterator featureCSV(featureFile);
    
    int i = 0;
    
    if (featureCSV == CSVIterator())
    {
        throw runtime_error("Error: Unable to load feature files");
    }
    
    while (featureCSV != CSVIterator())
    {
        // If the current line includes a file from the testing set
        if (find((*fileSet).begin(), (*fileSet).end(),(*featureCSV)[0]) != (*fileSet).end())
        {
            
            // Start at 1 to ignore filename column
            for (int j = 1; j < (int)(*featureCSV).size(); j++)
            {
                outputFeatures.at<float>(i, (j-1)) = stof((*featureCSV)[j]);
            }
            
            
            // Normalize
            cv::Scalar mean;
            cv::Scalar stddev;
            
            meanStdDev(outputFeatures.row(i), mean, stddev);
            
            for (int j = 1; j < (int)(*featureCSV).size(); j++)
            {
                outputFeatures.at<float>(i, (j-1)) = (outputFeatures.at<float>(i, (j-1)) - mean[0]) / stddev[0];
            }
            
            
            /* Load class into Mat
            Since we are adding the features in the order they appear in the feature csv, and not necessarily in the order of the file set, we need to make sure that the classes are added in the same order.
             */
            
            
            // Find index where current file is
            int idx = (int)(find((*fileSet).begin(), (*fileSet).end(), (*featureCSV)[0]) - (*fileSet).begin());
            
            outputLabels.at<int>(i,0) = (*classSet)[idx];
            
            // Increment the output features counter
            i++;
        }
        
        // Increment the CSV line
        featureCSV++;
    }
    
    // Check to ensure that the number of feature vectors retrieved is equal to the number requested
    if (i != (*classSet).size())
    {
        throw runtime_error("Error: unable to locate all features.");
    }
    
}






std::string TCLManager::generateFilename(int i)
{
    // Initialize new filename stringstream
    std::stringstream newFilename;
    newFilename <<  "svm-BSIF-" << modelSizes[i] << "-" << "RBF" << ".xml";
    
    return newFilename.str();
}

