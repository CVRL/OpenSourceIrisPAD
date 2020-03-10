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
    mapBool["Test list has base truth"] = &hasBaseTruth;
    mapBool["Majority voting"] = &majorityVoting;
    mapString["Segmentation"] = &segmentationType;
    mapString["Model type"] = &modelString;
    mapString["Bitsizes"] = &bitString;

    mapString["Image directory"] = &imageDir;
    mapString["CSV directory"] = &splitDir;
    mapString["Training set filename"] = &trainingSetFilename;
    mapString["Testing set filename"] = &testingSetFilename;
    mapString["Sizes"] = &trainingSizes;


    mapString["Feature extraction destination file"] = &outputExtractionFilename;
    mapString["Feature extraction destination directory"] = &outputExtractionDir;
    mapString["Model directory"] = &modelOutputDir;
    mapString["Classification filename"] = &classificationFilename;
    mapString["Classification file directory"] = &classificationDirectory;


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

    // Determine model sizes from trainingSizes string
    std::stringstream sizesStream(trainingSizes);

    std::string currentString;

    while (getline(sizesStream, currentString, ','))
    {
        modelSizes.push_back(stoi(currentString));
    }

    // Determine model types from modelString string
    std::stringstream modelStream(modelString);

    while (getline(modelStream, currentString, ','))
    {
        modelTypes.push_back(currentString);
    }

    // Determine bit sizes from modelString string
    std::stringstream bitStream(bitString);

    while (getline(bitStream, currentString, ','))
    {
        bitSizes.push_back(stoi(currentString));
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
    if (hasBaseTruth)
    {
        cout << endl << "- True classifications included for test set";
    }
    cout << endl;


    if (extractFeatures)
    {
        cout << "=============" << endl;
        cout << "- Features will be stored in directory: " << outputExtractionDir << endl;
        cout << "- Feature filenames will be in format: " << outputExtractionFilename + "_filter_size_size_bits.hdf5" << endl;
        cout << "- Segmentation type: " << segmentationType << endl;
        cout << "- Feature sets: " << endl;
        for (int i = 0; i < (int)modelSizes.size(); i++)
        {
            cout << bitSizes[i] << "," << modelSizes[i] << endl;
        }
        cout << "=============" << endl;
    }

    if (trainModel)
    {
        cout << "=============" << endl;
        cout << "- The training set will be: " << splitDir << trainingSetFilename << endl;
        cout << "- Models to be trained: " << endl;
        for (int i = 0; i < (int)modelSizes.size(); i++)
        {
            cout << "   " << generateFilename(i) << "  |  Model type: " << modelTypes[i] << endl;
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
            cout << "   " << generateFilename(i) << "  |  Model type: " << modelTypes[i] << endl;
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

        // Concatenate lists of files
        std::vector<std::string> extractionFilenames;
        extractionFilenames.insert(extractionFilenames.end(), trainingSet.begin(), trainingSet.end());
        extractionFilenames.insert(extractionFilenames.end(), testingSet.begin(), testingSet.end());

        for (int i = 0; i < (int)bitSizes.size(); i++)
        {
            cout << "Extracting..." << bitSizes[i] << "," << modelSizes[i] << endl;
            // Declare new feature extractor
            featureExtractor newExtractor(bitSizes[i], extractionFilenames, segmentationType);

            // Extract
            try
            {
                newExtractor.extract(outputExtractionDir, outputExtractionFilename, imageDir, modelSizes[i]);
            }
            catch (runtime_error& e)
            {
                throw e;
            }
        }

    }

    
    if (trainModel)
    {
        
        // Loop through modelSizes vector to train required models
        for (int i = 0; i < (int)modelSizes.size(); i++)
        {
            std::cout << "Training model " << (i + 1) << " out of " << modelSizes.size() << "..." << endl;


            // Load training data for current size
            cv::Mat featuresTrain;
            cv::Mat classesTrain;

            try
            {
                loadFeatures(featuresTrain, classesTrain, modelSizes[i], TRAIN, bitSizes[i]);
            }
            catch (runtime_error& e)
            {
                throw e;
            }


            if (modelTypes[i] == "svm")
            {
                std::cout << "  BSIF size " << modelSizes[i] << " Bit size " << bitSizes[i] << " | SVM with Kernel RBF " << endl;

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
            else if (modelTypes[i] == "rf")
            {
                std::cout << "  BSIF size " << modelSizes[i] << " Bit Sizes " << bitSizes[i] << " | Random Forest " << endl;

                // Place into trainData
                Ptr<TrainData> trainingData = TrainData::create(featuresTrain, ROW_SAMPLE, classesTrain);

                // Create new Random Forest
                Ptr<RTrees> rForest = RTrees::create();
                
                // Train model and save
                trainAuto_rf(trainingData, rForest);
                
                rForest->save(modelOutputDir + generateFilename(i));
                
                
            }
            else if (modelTypes[i] == "mp")
            {
               std::cout << "  BSIF size " << modelSizes[i] << " bit size " << bitSizes[i] << " | Multilayer Perceptron" << endl;

                // Place into trainData (need two responses (.8,-.8) and (-.8,.8))
                cv::Mat nnResponses(classesTrain.rows, 2, CV_32FC1);

                for (int i = 0; i < classesTrain.rows; i++)
                {
                    if (classesTrain.at<int>(i,0) == 1)
                    {
                        // Textured
                        nnResponses.at<float>(i,0) = 0.8;
                        nnResponses.at<float>(i,1) = -0.8;
                    }
                    else if (classesTrain.at<int>(i,0) == 0)
                    {
                        // Clear or none
                        nnResponses.at<float>(i,0) = -0.8;
                        nnResponses.at<float>(i,1) = 0.8;
                    }
                }
                Ptr<TrainData> trainingData = TrainData::create(featuresTrain, ROW_SAMPLE, nnResponses);
                
                

                // Train MLP and save
                
                Ptr<ANN_MLP> autoMLP = ANN_MLP::create();
                trainAuto_mlp(trainingData, autoMLP);
                autoMLP->save(modelOutputDir + generateFilename(i));
                
            }

        }
    }
    
    if (testImages)
    {
        std::cout << "Testing images..." << endl << endl;
        
        // Mat objects for testing data
        cv::Mat featuresTest;
        cv::Mat classesTest;
        
        // Results vector
        vector<cv::Mat> results;
        
        // determine individual results
        for (int i = 0; i < (int)modelSizes.size(); i++)
        {
            // Check if file exists
            ifstream nextFile(modelOutputDir + generateFilename(i));
            Ptr<StatModel> currentModel;
            if (nextFile.good())
            {
                if (modelTypes[i] == "svm")
                {
                    // Load SVM
                    currentModel = Algorithm::load<SVM>(modelOutputDir + generateFilename(i));
                }
                else if (modelTypes[i] == "rf")
                {
                    currentModel = Algorithm::load<RTrees>(modelOutputDir + generateFilename(i));
                }
                else if (modelTypes[i] == "mp")
                {
                    currentModel = Algorithm::load<ANN_MLP>(modelOutputDir + generateFilename(i));
                }
                
            }
            else
            {
                throw runtime_error("Error: Model \"" + generateFilename(i) + "\" not found.");
            }
            
            // Load testing features
            try
            {
                loadFeatures(featuresTest, classesTest, modelSizes[i], TEST, bitSizes[i]);
            }
            catch (runtime_error& e)
            {
                throw e;
            }
            
            
            if (modelTypes[i] == "mp")
            {
                // New mat for results
                cv::Mat individualResults(classesTest.rows, 2, CV_32FC1);
                
                // Predict using model
                currentModel->predict(featuresTest, individualResults);
                
                // Convert back to 0 or 1
                for (int j = 0; j < classesTest.rows; j++)
                {
                    if ((individualResults.at<float>(j,0) > 0.8) && (individualResults.at<float>(j,1) < -0.8))
                    {
                        individualResults.at<float>(j,0) = 1;
                    }
                    else if ((individualResults.at<float>(j,0) < -0.8) && (individualResults.at<float>(j,1) > 0.8))
                    {
                        individualResults.at<float>(j,0) = 0;
                    }
                    else
                    {
                        // If uncertain, assign randomly
                        individualResults.at<float>(j,0) = (rand() % 2);
                    }
                }
                
                // Add results to results vector
                results.push_back(individualResults);
            }
            else
            {
                // New Mat for results
                cv::Mat individualResults(classesTest.rows, classesTest.cols, CV_32FC1);
                
                // Predict using model
                currentModel->predict(featuresTest, individualResults);
                
                // Add results to results vector
                results.push_back(individualResults);
            }
            
        }
        
        // open file to output classifications
        ofstream classifications;
        classifications.open(classificationDirectory + classificationFilename);
        
        if (majorityVoting) {
            // Perform majority voting
            vector<int> overallResult;
            
            
            for (int i = 0; i < (int)testingSet.size(); i++)
            {
                // Variables for majority voting division
                int inFavor = 0;
                int against = 0;
                
                for (int j = 0; j < (int)modelSizes.size(); j++)
                {
                    if (results[j].at<float>(i,0) == 1)
                    {
                        inFavor++;
                    }
                    else
                    {
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
            }
            // Add the result to the output file
            addToClassificationFile(overallResult, classifications);
            
            // gets truth and compares to the result we saw here
            if (hasBaseTruth) {
                cout << "Number of models in the ensemble: " << modelSizes.size() << endl;
                outputStats(classesTest, overallResult);
            }
        }
        else {
            vector<int> overallResult;
            // loop through models
            for (int i = 0; i < modelSizes.size(); ++i) {
                // convert to vector
                overallResult.clear();
                for (int j = 0; j < testingSet.size(); ++j) {
                    overallResult.emplace_back(results[i].at<float>(j,0));
                }
                // add result to output file
                addToClassificationFile(overallResult, classifications);
                classifications << "------------------" << endl;
                
                // get stats
                if (hasBaseTruth) {
                    cout << "Model: " << generateFilename(i) << endl;
                    outputStats(classesTest, overallResult);
                }
            }
        }
        classifications.close();
    }
}



// Determines the number correct/incorrect if we have the base truth
// outputs to console
void TCLManager::outputStats(cv::Mat classesTest, vector<int>& result)
{
    int numIncorrect = 0;
    float apcer = 0;
    float bpcer = 0;
    int num_bonafide = 0;
    int num_attack = 0;
    
    // load the base truth
    for (int j = 0; j < classesTest.rows; j++)
    {
        if (classesTest.at<int>(j,0) == 0)
        {
            num_bonafide++;
        }
        else
        {
            num_attack++;
        }
    }
    
    // check against the model results
    for (int i = 0; i < testingClass.size(); ++i) {
        // Determine if incorrect
        if (result[i] != classesTest.at<int>(i,0))
        {
            numIncorrect++;
            if (classesTest.at<int>(i,0) == 1)
            {
                // positive misidentified as negative
                apcer++;
            }
            else
            {
                // negative misidentified as positive
                bpcer++;
            }
        }
    }
    
    // Output accuracy
    float ccr = 100 - ((float)numIncorrect / classesTest.rows) * 100;
    
    apcer = apcer / (float)num_attack * 100;
    bpcer = bpcer / (float)num_bonafide * 100;
    
    cout << "CCR: " << ccr << endl;
    cout << "APCER: " << apcer << endl;
    cout << "BPCER: " << bpcer << endl << endl;
}

// logs a list of filenames and the classifications determined by the selected models or majority voting combination of them
void TCLManager::addToClassificationFile(vector<int>& result, ofstream& file) {
    for (int i = 0; i < testingSet.size(); ++i) {
        // add each result to the file
        file << testingSet[i] << "," << result[i] << endl;
    }
}

// Initializes all parameters
void TCLManager::initConfig(void)
{

    // Commands
    extractFeatures = false;
    trainModel = false;
    testImages = false;
    majorityVoting = false;
    segmentationType = "wi";

    // Inputs
    imageDir = "";
    splitDir = "";
    trainingSetFilename = "";
    testingSetFilename = "";
    trainingSizes = "";
    modelString = "";
    bitString = "";

    // Outputs
    outputExtractionFilename = "";
    outputExtractionDir = "";
    modelOutputDir = "";
}





// Loads the training/testing image names indicated in the config file
void TCLManager::loadSets(void)
{
    string currentName;
    size_t location;

    // Clear
    trainingSet.clear();
    trainingClass.clear();
    testingSet.clear();
    testingClass.clear();

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
            if (hasBaseTruth) testingClass.push_back(stoi(currentName.substr((location + 1))));
        }
        test.close();

    } else if (testImages)
    {
        // if image testing is requested but no file is given
        throw runtime_error("Error: please specify a list of images for testing (testing set filename)");
    }

}





// Loads features for training or testing sets into Mat objects
void TCLManager::loadFeatures(cv::Mat& outputFeatures, cv::Mat& outputLabels, int filtersize, int setType, int bitType)
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
    outputFeatures.create((int)(*fileSet).size(), pow(2,bitType), CV_32FC1);
    outputLabels.create((int)(*classSet).size(), 1, CV_32SC1);
    
    std::vector<int> histogram(pow(2,bitType), 0);

    
    // HDF5 Version
    
    // Load files
    stringstream featureFilename;
    featureFilename << outputExtractionDir << outputExtractionFilename << "_filter_" << filtersize << "_" << filtersize << "_" << bitType << ".hdf5";
    string featureName = featureFilename.str();
    const char* file_to_open = featureName.c_str();
    
    hid_t       file_id;   /* file identifier */
    hid_t     dataset_id;
    herr_t      status;
    
    
    
    // Open existing file
    try {
        file_id = H5Fopen(file_to_open, H5F_ACC_RDWR, H5P_DEFAULT);
    } catch (runtime_error e) {
        throw runtime_error("Error: no features found in given directory");
    }
    
    // Loop through file set
    for (int i = 0; i < (int)(*fileSet).size(); i++)
    {
        const char* dataset_to_open = (*fileSet)[i].c_str();
        // open dataset for specific image
        dataset_id = H5Dopen2(file_id, dataset_to_open, H5P_DEFAULT); // returns negative value if unsuccessful (image features not found)
        
        if (dataset_id >= 0) // features were found
        {
            status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &histogram[0]);
        }
        else
        {
            status = H5Dclose(dataset_id);
            status = H5Fclose(file_id);
            throw runtime_error("Error: features not found in given directory");
            //TODO: extract new features
        }
        
        // Close dataset
        status = H5Dclose(dataset_id);
        
        // Store histogram in Mat object
        for (int j = 0; j < (int)(histogram).size(); j++)
        {
            outputFeatures.at<float>(i, j) = (float)histogram[j];
        }
        
        
        // Normalize
        cv::Scalar mean;
        cv::Scalar stddev;
        
        meanStdDev(outputFeatures.row(i), mean, stddev);
        
        for (int j = 0; j < (int)(histogram).size(); j++)
        {
            outputFeatures.at<float>(i, (j)) = (outputFeatures.at<float>(i, j) - mean[0]) / stddev[0];
        }
        
        
        // Load class into Mat
        // don't load labels for the test set if it doesn't have any
        if (hasBaseTruth || (setType == TRAIN)) {
            outputLabels.at<int>(i,0) = (*classSet)[i];
        }

    }
    
    // Close file
    status = H5Fclose(file_id);

}


// finds the best parameters for a random forest model
void TCLManager::trainAuto_rf(Ptr<TrainData>& data, Ptr<RTrees> model)
{
    // Define parameters
    int k_folds = 10;
    
    // 10 folds times 20 trials = 200 total training cycles
    vector<int> training_depth = {1, 5, 10, 15, 20, 25};
    vector<float> percent_of_traindata = {1, 1.5, 2, 2.5};
    
    // get training data
    Mat samples = data->getTrainSamples();
    Mat responses = data->getTrainResponses();
    //cout << responses << endl;
    int sample_count = samples.rows;
    
    vector<int> sidx;
    setRangeVector(sidx, sample_count);
    RNG rng = RNG();
    
    // randomly permute training samples
    for( int i = 0; i < sample_count; i++ )
    {
        int i1 = rng.uniform(0, sample_count);
        int i2 = rng.uniform(0, sample_count);
        std::swap(sidx[i1], sidx[i2]);
    }
    
    // reshuffle the training set in such a way that
    // instances of each class are divided more or less evenly
    // between the k_fold parts.
    vector<int> sidx0, sidx1;
    
    // separate pos and neg samples
    for( int i = 0; i < sample_count; i++ )
    {
        if( responses.at<int>(sidx[i],0) == 0 )
        {
            sidx0.push_back(sidx[i]);
        }
        else
        {
           sidx1.push_back(sidx[i]);
        }
    }
    
    int n0 = (int)sidx0.size(), n1 = (int)sidx1.size();
    int a0 = 0, a1 = 0;
    vector<vector<int>> k_idx;
    
    for( int k = 0; k < k_folds; k++ )
    {
        sidx.clear();
        int b0 = ((k+1)*n0 + k_folds/2)/k_folds, b1 = ((k+1)*n1 + k_folds/2)/k_folds;
        int a = (int)sidx.size(), b = a + (b0 - a0) + (b1 - a1); // b gives end position of samples
        // a0 is startpoint and b0 is endpoint
        for( int i = a0; i < b0; i++ )
            sidx.push_back(sidx0[i]);
        for( int i = a1; i < b1; i++ )
            sidx.push_back(sidx1[i]);
        for( int i = 0; i < (b - a); i++ ) // for the number of samples
        {
            int i1 = rng.uniform(a, b); // swap samples between a and b (the ones that have been added this cycle)
            int i2 = rng.uniform(a, b);
            std::swap(sidx[i1], sidx[i2]);
        }
        // start at previous ending positions
        a0 = b0; a1 = b1;
        // add k set to vector
        k_idx.push_back(sidx);
    }
    
    // calculate number of samples per fold
    int number_per_fold = ((n0 + k_folds/2)/k_folds + (n1 + k_folds/2)/k_folds);
    
    // testing parameters
    int best_depth = 0;
    int best_count = 0;
    float best_ccr = 0.0;
    
    for (int i = 0; i < (int)training_depth.size(); i++) // all training depths
    {
        for (int j = 0; j < (int)percent_of_traindata.size(); j++) // all min_sample_count
        {
            // create vector to track performance over all folds
            vector<float> ccr;
            
            // Loop through 10 folds
            for (int k = 0; k < k_folds; k++)
            {
                // load folds minus k
                Mat train_samples(((k_folds - 1)* number_per_fold), samples.cols, CV_32FC1);
                Mat train_classes(((k_folds - 1)* number_per_fold), responses.cols, CV_32SC1);
                int samples_added = 0;
                for (int l = 0; l < k_folds; l++)
                {
                    if (l != k) // skip current fold to use for testing
                    {
                        // load indices for the current fold
                        vector<int> current_fold = k_idx[l];
                        // add samples to mat objects
                        for (int sample_num = 0; sample_num < (int)current_fold.size(); sample_num++)
                        {
                            samples.row(current_fold[sample_num]).copyTo(train_samples.row(samples_added));
                            train_classes.at<int>(samples_added,0) = responses.at<int>(current_fold[sample_num],0);
                            samples_added++;
                        }
                    }
                }
                
                // Place into trainData
                Ptr<TrainData> trainingData = TrainData::create(train_samples, ROW_SAMPLE, train_classes);
                
                // Create new Random Forest
                Ptr<RTrees> rForest_tmp = RTrees::create();
                rForest_tmp->setMaxDepth(training_depth[i]);
                rForest_tmp->setMinSampleCount((int)(sample_count * percent_of_traindata[j] / 100));
                
                // Train model and save
                rForest_tmp->train(trainingData);
                
                // Load final fold to test
                vector<int> final_fold = k_idx[k];
                Mat test_samples(number_per_fold, samples.cols, CV_32FC1);
                Mat test_classes(number_per_fold, responses.cols, CV_32SC1);
                // add samples to mat objects
                for (int sample_num = 0; sample_num < (int)final_fold.size(); sample_num++)
                {
                    samples.row(final_fold[sample_num]).copyTo(test_samples.row(sample_num));
                    test_classes.at<int>(sample_num,0) = responses.at<int>(final_fold[sample_num],0);
                    samples_added++;
                }
                
                // Test model
                cv::Mat predictions(test_classes.rows, test_classes.cols, CV_32FC1); // new mat for results
                rForest_tmp->predict(test_samples, predictions);
                // Determine number incorrect
                int numIncorrect = 0;
                for (int pnum = 0; pnum < test_classes.rows; pnum++)
                {
                    if (predictions.at<float>(pnum,0) != test_classes.at<int>(pnum,0))
                    {
                        numIncorrect++;
                    }
                }
            
                // Output accuracy
                float ccr_single = 100 - ((float)numIncorrect / test_classes.rows) * 100;
                
                ccr.push_back(ccr_single);
            }
            // determine average performance over folds
            float total = 0;
            for (int k = 0; k < (int)ccr.size(); k++)
            {
                total += ccr[k];
            }
            float mean_performance = total / (float)ccr.size();
            
            // check if this is better than other models
            if (mean_performance > best_ccr)
            {
                best_ccr = mean_performance;
                best_depth = training_depth[i];
                best_count = percent_of_traindata[j];
            }
        }
    }
    
    // train with best parameters
    model->setMaxDepth(best_depth);
    model->setMinSampleCount((int)(sample_count * best_count / 100));
    
    model->train(data);
    
}

// finds the best parameters for a multilayer perceptron model
void TCLManager::trainAuto_mlp(Ptr<TrainData>& data, Ptr<ANN_MLP> model)
{
    // Define parameters
    int k_folds = 10;
    
    // 10 folds times 4 trials = 40 total training cycles
    vector<int> hidden_layer_multiplier = {1, 2, 4};
    
    // get training data
    Mat samples = data->getTrainSamples();
    Mat responses = data->getTrainResponses();
    
    /* CREATE K-FOLDS */
    int sample_count = samples.rows;
    
    vector<int> sidx;
    setRangeVector(sidx, sample_count);
    RNG rng = RNG();
    
    // randomly permute training samples
    for( int i = 0; i < sample_count; i++ )
    {
        int i1 = rng.uniform(0, sample_count);
        int i2 = rng.uniform(0, sample_count);
        std::swap(sidx[i1], sidx[i2]);
    }
    
    // reshuffle the training set in such a way that
    // instances of each class are divided more or less evenly
    // between the k_fold parts.
    vector<int> sidx0, sidx1;
    
    // separate pos and neg samples
    for( int i = 0; i < sample_count; i++ )
    {
        // 0 for MLP is -0.8 and 0.8
        if((responses.at<float>(sidx[i],0)) == -0.8 && (responses.at<float>(sidx[i],1) == 0.8))
        {
            sidx0.push_back(sidx[i]);
        }
        else // must be a 1
        {
           sidx1.push_back(sidx[i]);
        }
    }
    
    int n0 = (int)sidx0.size(), n1 = (int)sidx1.size();
    int a0 = 0, a1 = 0;
    vector<vector<int>> k_idx;
    
    for( int k = 0; k < k_folds; k++ )
    {
        sidx.clear();
        int b0 = ((k+1)*n0 + k_folds/2)/k_folds, b1 = ((k+1)*n1 + k_folds/2)/k_folds;
        int a = (int)sidx.size(), b = a + (b0 - a0) + (b1 - a1); // b gives end position of samples
        // a0 is startpoint and b0 is endpoint
        for( int i = a0; i < b0; i++ )
            sidx.push_back(sidx0[i]);
        for( int i = a1; i < b1; i++ )
            sidx.push_back(sidx1[i]);
        for( int i = 0; i < (b - a); i++ ) // for the number of samples
        {
            int i1 = rng.uniform(a, b); // swap samples between a and b (the ones that have been added this cycle)
            int i2 = rng.uniform(a, b);
            std::swap(sidx[i1], sidx[i2]);
        }
        // start at previous ending positions
        a0 = b0; a1 = b1;
        // add k set to vector
        k_idx.push_back(sidx);
    }
    
    // calculate number of samples per fold
    int number_per_fold = ((n0 + k_folds/2)/k_folds + (n1 + k_folds/2)/k_folds);
    
    // testing parameters
    int best_multiplier = 0;
    float best_ccr = 0.0;
    
    /* PERFORM K-FOLD ANALYSIS */
    for (int i = 0; i < (int)hidden_layer_multiplier.size(); i++) // all hidden layer sizes
    {
        
        // create vector to track performance over all folds
        vector<float> ccr;
        
        // Loop through 10 folds
        for (int k = 0; k < k_folds; k++)
        {
            // load folds minus k
            Mat train_samples(((k_folds - 1)* number_per_fold), samples.cols, CV_32FC1);
            Mat train_classes(((k_folds - 1)* number_per_fold), responses.cols, CV_32FC1);
            int samples_added = 0;
            for (int l = 0; l < k_folds; l++)
            {
                if (l != k) // skip current fold to use for testing
                {
                    // load indices for the current fold
                    vector<int> current_fold = k_idx[l];
                    // add samples to mat objects
                    for (int sample_num = 0; sample_num < (int)current_fold.size(); sample_num++)
                    {
                        samples.row(current_fold[sample_num]).copyTo(train_samples.row(samples_added));
                        train_classes.at<float>(samples_added,0) = responses.at<float>(current_fold[sample_num],0);
                        train_classes.at<float>(samples_added,1) = responses.at<float>(current_fold[sample_num],1);
                        samples_added++;
                    }
                }
            }
            
            // Place into trainData
            Ptr<TrainData> trainingData = TrainData::create(train_samples, ROW_SAMPLE, train_classes);
            
            // Define multilayer perceptron parameters
            cv::Mat layerSize(3, 1, CV_32SC1);
            layerSize.at<int>(0,0) = train_samples.cols;
            layerSize.at<int>(1,0) = train_samples.cols * hidden_layer_multiplier[i];
            layerSize.at<int>(2,0) = train_classes.cols;
            
            // Create MLP
            Ptr<ANN_MLP> tmp_MLP = ANN_MLP::create();
            tmp_MLP->setLayerSizes(layerSize);
            tmp_MLP->setTrainMethod(ANN_MLP::BACKPROP);
            tmp_MLP->setActivationFunction(ANN_MLP::SIGMOID_SYM,1,1);
            
            // Train MLP
            tmp_MLP->train(trainingData);
            
            // Load final fold to test
            vector<int> final_fold = k_idx[k];
            Mat test_samples(number_per_fold, samples.cols, CV_32FC1);
            Mat test_classes(number_per_fold, responses.cols, CV_32FC1);
            // add samples to mat objects
            for (int sample_num = 0; sample_num < (int)final_fold.size(); sample_num++)
            {
                samples.row(final_fold[sample_num]).copyTo(test_samples.row(sample_num));
                test_classes.at<float>(sample_num,0) = responses.at<float>(final_fold[sample_num],0);
                test_classes.at<float>(sample_num,1) = responses.at<float>(final_fold[sample_num],1);
                samples_added++;
            }
            
            // Test model
            cv::Mat predictions(test_classes.rows, test_classes.cols, CV_32FC1); // new mat for results
            tmp_MLP->predict(test_samples, predictions);
            
            // Determine number incorrect
            int numIncorrect = 0;
            for (int pnum = 0; pnum < test_classes.rows; pnum++)
            {
                if ((predictions.at<float>(pnum,0) > 0.8) && (predictions.at<float>(pnum,1) < -0.8))
                {
                    // Falsely identified as textured
                    if (test_classes.at<float>(pnum,0) == -0.8)
                    {
                       numIncorrect++;
                    }
                }
                else if ((predictions.at<float>(pnum,0) < -0.8) && (predictions.at<float>(pnum,1) > 0.8))
                {
                    // Falsely identified as not textured
                    if (test_classes.at<float>(pnum,0) == 0.8)
                    {
                        numIncorrect++;
                    }
                }
                else
                {
                    // no prediction
                    numIncorrect++;
                }
            }
            
            // Output accuracy
            float ccr_single = 100 - ((float)numIncorrect / test_classes.rows) * 100;
            
            ccr.push_back(ccr_single);
        }
        // determine average performance over folds
        float total = 0;
        for (int k = 0; k < (int)ccr.size(); k++)
        {
            total += ccr[k];
        }
        float mean_performance = total / (float)ccr.size();
        
        // check if this is better than other models
        if (mean_performance > best_ccr)
        {
            best_ccr = mean_performance;
            best_multiplier = hidden_layer_multiplier[i];
        }
    }
    
    /* TRAIN BEST MODEL */
    // Define multilayer perceptron parameters
    cv::Mat layerSize(3, 1, CV_32SC1);
    layerSize.at<int>(0,0) = samples.cols;
    layerSize.at<int>(1,0) = samples.cols * best_multiplier;
    layerSize.at<int>(2,0) = responses.cols;
    
    // Create MLP
    model->setLayerSizes(layerSize);
    model->setTrainMethod(ANN_MLP::BACKPROP);
    model->setActivationFunction(ANN_MLP::SIGMOID_SYM,1,1);
    
    model->train(data);
    
}


// creates the filename for a specific model
std::string TCLManager::generateFilename(int i)
{
    // Initialize new filename stringstream
    std::stringstream newFilename;
    newFilename <<  "BSIF-" << bitSizes[i] << "-" << modelSizes[i] << "-" << modelTypes[i] << "-" << segmentationType << ".xml";

    return newFilename.str();
}

// function to create a vector from 1:n with entries equivalent to locations
static inline void setRangeVector(std::vector<int>& vec, int n)
{
    vec.resize(n);
    for( int i = 0; i < n; i++ )
        vec[i] = i;
}
