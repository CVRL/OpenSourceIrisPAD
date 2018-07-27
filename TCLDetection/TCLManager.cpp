//
//  TCLManager.cpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/17/18.

#include <stdio.h>
//#include <iostream>
//#include <fstream>
//#include <string>
#include <map>
#include "CSVIterator.hpp"
#include "BSIFFilter.hpp"
#include "OsiStringUtils.h"
#include "sampleDatabase.cpp"

using namespace std;
/*
// Class to hold sample information
class irisSample {
public:
    std::string filename;
    std::string classification;
    std::string manufacturer;
    std::string sensor;
    
    void clear(void) {
        filename = "";
        classification = "";
        manufacturer = "";
        sensor = "";
    }
};

// Class to hold subject information
class subject {
public:
    std::vector<irisSample> mSamples;
    int numNone;
    int numClear;
    int numTextured;
    std::string ID;
};
*/
class TCLManager {
public:
    TCLManager(void) {
        // Associate lines of config file to attributes
        mapBool["Extract features"] = &extractFeatures;
        mapBool["Train model"] = &trainModel;
        mapBool["Test images"] = &testImages;
        
        mapString["Database file"] = &inputDatabaseFilename;
        mapString["Database directory"] = &inputDatabaseDir;
        mapString["Database image directory"] = &databaseImageDir;
        mapString["Feature extraction destination file"] = &outputExtractionFilename;
        mapString["Feature extraction destination directory"] = &outputExtractionDir;
        mapString["Histogram output file"] = &outputHistFilename;
        mapString["Histogram output directory"] = &outputHistDir;
        
        mapString["SequenceID column name"] = &sequenceIDColumnName;
        mapString["Format column name"] = &formatColumnName;
        mapString["Subject column name"] = &subjectColumnName;
        mapString["Texture column name"] = &textureColumnName;
        mapString["Contacts column name"] = &contactsColumnName;
        mapString["Tags column name"] = &tagsColumnName;
        mapString["Manufacturer tag"] = &manufacturerTag;

        
        
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
            cout << "- The database for feature extraction will be " << inputDatabaseFilename << endl;
            cout << "- Features will be stored in " << outputExtractionFilename << endl;
            cout << "=============" << endl;
            
            cout << "Database Parameters" << endl;
            cout << "=============" << endl;
            cout << "SequenceID: " << sequenceIDColumnName << endl;
            cout << "Format: " << formatColumnName << endl;
            cout << "Texture: "  << textureColumnName << endl;
            cout << "Contacts: " << contactsColumnName << endl;
            cout << "Tags: " << tagsColumnName << endl;
            cout << "Manufacturer Tag: " << manufacturerTag << endl;
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
        
        if (extractFeatures) {
            std::cout << "Extracting features..." << std::endl;
            //parseDatabaseCSV();
            sampleDatabase inputDatabase(inputDatabaseDir, inputDatabaseFilename);
            inputDatabase.parseDatabaseCSV(sequenceIDColumnName, formatColumnName, subjectColumnName, textureColumnName, contactsColumnName, tagsColumnName, manufacturerTag);
            inputDatabase.save(outputExtractionDir);
            
            sampleDatabase loadData(outputExtractionDir);
            loadData.load();
            
            extractHist();
        }
        
        if (trainModel) {
            std::cout << "Training SVM..." << std::endl;
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
    std::string inputDatabaseFilename;
    std::string inputDatabaseDir;
    std::string databaseImageDir;
    std::vector<subject> listOfSubjects;
    
    // Outputs
    std::string outputExtractionFilename;
    std::string outputExtractionDir;
    std::string outputHistFilename;
    std::string outputHistDir;
    
    
    // Parameters
    int bitsize;
    std::string sequenceIDColumnName;
    std::string formatColumnName;
    std::string subjectColumnName;
    std::string textureColumnName;
    std::string contactsColumnName;
    std::string tagsColumnName;
    std::string manufacturerTag;
    
    // Maps to associate a string (conf file) to a variable (pointer)
    std::map<std::string,bool*> mapBool;
    std::map<std::string,int*> mapInt;
    std::map<std::string,std::string*> mapString;
    
    // Initialize all parameters
    void initConfig(void) {
        // Commands
        extractFeatures = false;
        trainModel = false;
        testImages = false;
        
        // Inputs
        inputDatabaseFilename = "";
        inputDatabaseDir = "";
        databaseImageDir = "";
        listOfSubjects.clear();
        
        // Outputs
        outputExtractionFilename = "";
        outputExtractionDir = "";
        outputHistFilename = "";
        outputHistDir = "";
        
        // Parameters
        bitsize = 8;
        
        sequenceIDColumnName = "";
        formatColumnName = "";
        subjectColumnName = "";
        textureColumnName = "";
        contactsColumnName = "";
        tagsColumnName = "";
        manufacturerTag = "";
    }
    
    void parseDatabaseCSV(void) {
        // Set up file input and CSVIterator
        ifstream databaseFile(inputDatabaseDir + inputDatabaseFilename);
        CSVIterator databaseCSV(databaseFile);
        
        // Find the indices of the desired columns
        int numDesiredColumns = 6;
        int sequenceidIdx = 0;
        int formatIdx = 0;
        int subjectIdx = 0;
        int textureIdx = 0;
        int contactsIdx = 0;
        int tagsIdx = 0;
        
        for (int i = 0; i < (*databaseCSV).size(); i++) {
            if ((*databaseCSV)[i] == sequenceIDColumnName) {
                sequenceidIdx = i;
                numDesiredColumns--;
            } else if ((*databaseCSV)[i] == formatColumnName) {
                formatIdx = i;
                numDesiredColumns--;
            } else if ((*databaseCSV)[i] == subjectColumnName) {
                subjectIdx = i;
                numDesiredColumns--;
            } else if ((*databaseCSV)[i] == textureColumnName) {
                textureIdx = i;
                numDesiredColumns--;
            } else if ((*databaseCSV)[i] == contactsColumnName) {
                contactsIdx = i;
                numDesiredColumns--;
            } else if ((*databaseCSV)[i] == tagsColumnName) {
                tagsIdx = i;
                numDesiredColumns--;
            }
        }
        
        if (numDesiredColumns == 0) {
            // Increment databaseCSV to skip header line
            databaseCSV++;
            
            // Step through database and add to list of samples
            while (databaseCSV != CSVIterator()) {
                irisSample currentSample;
               
                bool contacts = false;
                bool textured = false;
                
                if ((*databaseCSV)[contactsIdx] == "Yes") {contacts = true;} // Maybe use tobool string class from osiris
                if ((*databaseCSV)[textureIdx] == "Yes") {textured = true;}
                
                currentSample.filename = (*databaseCSV)[sequenceidIdx] + "." + (*databaseCSV)[formatIdx];
                
                if (contacts && textured) {
                    currentSample.classification = "textured";
                } else if (contacts) {
                    currentSample.classification = "clear";
                } else {
                    currentSample.classification = "none";
                }
                
                // Look through tags to find manufacturer
                std::stringstream tagStream((*databaseCSV)[tagsIdx]);
                std::string tag;
                
                // Delim is \ as given in NDCLD15 database tag column
                while (std::getline(tagStream, tag, '\\')) {
                    // Remove extra commas
                    size_t commaIdx;
                    while ((commaIdx = tag.find(",")) != std::string::npos) {
                        tag.erase(commaIdx,1);
                    }
                    // Remove quotation marks
                    size_t quoteIdx;
                    while ((quoteIdx = tag.find("\"")) != std::string::npos) {
                        tag.erase(quoteIdx,1);
                    }

                    
                    // Look for manufacturer tag
                    if (tag.substr(0, manufacturerTag.length()) == "contacts-manufacturer") {
                        currentSample.manufacturer = tag.substr(manufacturerTag.length() + 1);
                    }
                }
                
                // Find subject identifier
                std::string subjectID = (*databaseCSV)[subjectIdx];
                
                // Determine if subject has already been created
                int locationInList = -1;
                for (int j = 0; j < listOfSubjects.size(); j++) {
                    if (listOfSubjects.at(j).ID == subjectID) {locationInList = j;}
                }
                
                // If this is a new subject, create and add to vector
                if (locationInList == -1) {
                    subject newSubject;
                    newSubject.ID = subjectID;
                    newSubject.numNone = 0;
                    newSubject.numClear = 0;
                    newSubject.numTextured = 0;
                    newSubject.mSamples.push_back(currentSample);
                    if (currentSample.classification == "none") {
                        newSubject.numNone++;
                    } else if (currentSample.classification == "clear") {
                        newSubject.numClear++;
                    } else if (currentSample.classification == "textured") {
                        newSubject.numTextured++;
                    }
                    listOfSubjects.push_back(newSubject);
                } else {
                    listOfSubjects.at(locationInList).mSamples.push_back(currentSample);
                    if (currentSample.classification == "none") {
                        listOfSubjects.at(locationInList).numNone++;
                    } else if (currentSample.classification == "clear") {
                        listOfSubjects.at(locationInList).numClear++;
                    } else if (currentSample.classification == "textured") {
                        listOfSubjects.at(locationInList).numTextured++;
                    }
                }
                // Increment
                databaseCSV++;
                    
            }
        } else {
            cout << "Error: incorrect .csv file format.  Need to have sequenceid and format available." << endl;
        }
        
        // Store list of subjects (separate method)
        ofstream subjectList;
        subjectList.open((outputExtractionDir + "subjectList.csv"), ios::out | ios::trunc);
        for (int i = 0; i < listOfSubjects.size(); i++) {
            subject currentSubject = listOfSubjects.at(i);
            subjectList << currentSubject.ID << ", " << currentSubject.numNone << ", " << currentSubject.numClear << ", " << currentSubject.numTextured;
            for (int j = 0; j < currentSubject.mSamples.size(); j++) {
                irisSample currentSample = currentSubject.mSamples.at(j);
                subjectList << ", " << currentSample.filename;
            }
            subjectList << endl;
        }
        subjectList.close();
    }
    
    void extractHist(void) {
        // Load filters
        BSIFFilter bsifThree(3,bitsize);
        BSIFFilter bsifFive(5,bitsize);
        BSIFFilter bsifSeven(7,bitsize);
        BSIFFilter bsifNine(9,bitsize);
        BSIFFilter bsifEleven(11,bitsize);
        BSIFFilter bsifThirteen(13,bitsize);
        BSIFFilter bsifFifteen(15,bitsize);
        BSIFFilter bsifSeventeen(17,bitsize);
        
        // Initialize output files
        // Full sized outputs
        ofstream histThree;
        histThree.open((outputExtractionDir + outputExtractionFilename + "_" + bsifThree.filtername + ".csv"), ios::out | ios::trunc);
        ofstream histFive;
        histFive.open((outputExtractionDir + outputExtractionFilename + "_" + bsifFive.filtername + ".csv"), ios::out | ios::trunc);
        ofstream histSeven;
        histSeven.open((outputExtractionDir + outputExtractionFilename + "_" + bsifSeven.filtername + ".csv"), ios::out | ios::trunc);
        ofstream histNine;
        histNine.open((outputExtractionDir + outputExtractionFilename + "_" + bsifNine.filtername + ".csv"), ios::out | ios::trunc);
        ofstream histEleven;
        histEleven.open((outputExtractionDir + outputExtractionFilename + "_" + bsifEleven.filtername + ".csv"), ios::out | ios::trunc);
        ofstream histThirteen;
        histThirteen.open((outputExtractionDir + outputExtractionFilename + "_" + bsifThirteen.filtername + ".csv"), ios::out | ios::trunc);
        ofstream histFifteen;
        histFifteen.open((outputExtractionDir + outputExtractionFilename + "_" + bsifFifteen.filtername + ".csv"), ios::out | ios::trunc);
        ofstream histSeventeen;
        histSeventeen.open((outputExtractionDir + outputExtractionFilename + "_" + bsifSeventeen.filtername + ".csv"), ios::out | ios::trunc);
        // Downsampled outputs
        ofstream histSix;
        histSix.open((outputExtractionDir + outputExtractionFilename + "_" + bsifThree.downFiltername + ".csv"), ios::out | ios::trunc);
        ofstream histTen;
        histTen.open((outputExtractionDir + outputExtractionFilename + "_" + bsifFive.downFiltername + ".csv"), ios::out | ios::trunc);
        ofstream histFourteen;
        histFourteen.open((outputExtractionDir + outputExtractionFilename + "_" + bsifSeven.downFiltername + ".csv"), ios::out | ios::trunc);
        ofstream histEighteen;
        histEighteen.open((outputExtractionDir + outputExtractionFilename + "_" + bsifNine.downFiltername + ".csv"), ios::out | ios::trunc);
        ofstream histTwentyTwo;
        histTwentyTwo.open((outputExtractionDir + outputExtractionFilename + "_" + bsifEleven.downFiltername + ".csv"), ios::out | ios::trunc);
        ofstream histTwentySix;
        histTwentySix.open((outputExtractionDir + outputExtractionFilename + "_" + bsifThirteen.downFiltername + ".csv"), ios::out | ios::trunc);
        ofstream histThirty;
        histThirty.open((outputExtractionDir + outputExtractionFilename + "_" + bsifFifteen.downFiltername + ".csv"), ios::out | ios::trunc);
        ofstream histThirtyFour;
        histThirtyFour.open((outputExtractionDir + outputExtractionFilename + "_" + bsifSeventeen.downFiltername + ".csv"), ios::out | ios::trunc);
        
        // Initialize histogram
        int histsize = pow(2,bitsize) + 1;
        std::vector<int> histogram(histsize, 0);
       
        // Loop through images stored in listOfSubjects (results from same subject will be stored in close proximity)
        for (int i = 0; i < listOfSubjects.size(); i++) {
            // Display progress
            std::cout << "Processing subject " << (i + 1) << " out of " << listOfSubjects.size() << endl;
            // Load current subject
            subject currentSubject = listOfSubjects.at(i);
            
            // Loop through samples for a single subject
            for (int j = 0; j < currentSubject.mSamples.size(); j++) {
                // Display progress
                std::cout << "  Image " << (j + 1) << " out of " << currentSubject.mSamples.size() << endl;
                
                // Load current sample
                irisSample currentSample = currentSubject.mSamples.at(j);
                
                // Load image from file
                cv::Mat image = cv::imread((databaseImageDir + currentSample.filename), 0);
                
                // Store image information
                histThree << currentSample.filename << ", " << currentSample.filename << ", " << currentSample.classification << ", " << currentSample.manufacturer << ", " << currentSample.sensor << ", ";
                histFive << currentSample.filename << ", " << currentSample.classification << ", " << currentSample.manufacturer << ", " << currentSample.sensor << ", ";
                histSeven << currentSample.filename << ", " << currentSample.classification << ", " << currentSample.manufacturer << ", " << currentSample.sensor << ", ";
                histNine << currentSample.filename << ", " << currentSample.classification << ", " << currentSample.manufacturer << ", " << currentSample.sensor << ", ";
                histEleven << currentSample.filename << ", " << currentSample.classification << ", " << currentSample.manufacturer << ", " << currentSample.sensor << ", ";
                histThirteen << currentSample.filename << ", " << currentSample.classification << ", " << currentSample.manufacturer << ", " << currentSample.sensor << ", ";
                histFifteen << currentSample.filename << ", " << currentSample.classification << ", " << currentSample.manufacturer << ", " << currentSample.sensor << ", ";
                histSeventeen << currentSample.filename << ", " << currentSample.classification << ", " << currentSample.manufacturer << ", " << currentSample.sensor << ", ";
                histSix << currentSample.filename << ", " << currentSample.classification << ", " << currentSample.manufacturer << ", " << currentSample.sensor << ", ";
                histTen << currentSample.filename << ", " << currentSample.classification << ", " << currentSample.manufacturer << ", " << currentSample.sensor << ", ";
                histFourteen << currentSample.filename << ", " << currentSample.classification << ", " << currentSample.manufacturer << ", " << currentSample.sensor << ", ";
                histEighteen << currentSample.filename << ", " << currentSample.classification << ", " << currentSample.manufacturer << ", " << currentSample.sensor << ", ";
                histTwentyTwo << currentSample.filename << ", " << currentSample.classification << ", " << currentSample.manufacturer << ", " << currentSample.sensor << ", ";
                histTwentySix << currentSample.filename << ", " << currentSample.classification << ", " << currentSample.manufacturer << ", " << currentSample.sensor << ", ";
                histThirty << currentSample.filename << ", " << currentSample.classification << ", " << currentSample.manufacturer << ", " << currentSample.sensor << ", ";
                histThirtyFour << currentSample.filename << ", " << currentSample.classification << ", " << currentSample.manufacturer << ", " << currentSample.sensor << ", ";
                
                
                // Calculate histograms for full sized images
                bsifThree.generateHistogram(image, histogram);
                for (int i = 1; i < histsize; i++) histThree << histogram[i] << ", ";
                histThree << std::endl;
                std::fill(histogram.begin(), histogram.end(), 0);
                
                bsifFive.generateHistogram(image, histogram);
                for (int i = 1; i < histsize; i++) histFive << histogram[i] << ", ";
                histFive << std::endl;
                std::fill(histogram.begin(), histogram.end(), 0);
                
                bsifSeven.generateHistogram(image, histogram);
                for (int i = 1; i < histsize; i++) histSeven << histogram[i] << ", ";
                histSeven << std::endl;
                std::fill(histogram.begin(), histogram.end(), 0);
                
                bsifNine.generateHistogram(image, histogram);
                for (int i = 1; i < histsize; i++) histNine << histogram[i] << ", ";
                histNine << std::endl;
                std::fill(histogram.begin(), histogram.end(), 0);
                
                bsifEleven.generateHistogram(image, histogram);
                for (int i = 1; i < histsize; i++) histEleven << histogram[i] << ", ";
                histEleven << std::endl;
                std::fill(histogram.begin(), histogram.end(), 0);
                
                bsifThirteen.generateHistogram(image, histogram);
                for (int i = 1; i < histsize; i++) histThirteen << histogram[i] << ", ";
                histThirteen << std::endl;
                std::fill(histogram.begin(), histogram.end(), 0);
                
                bsifFifteen.generateHistogram(image, histogram);
                for (int i = 1; i < histsize; i++) histFifteen << histogram[i] << ", ";
                histFifteen << std::endl;
                std::fill(histogram.begin(), histogram.end(), 0);
                
                bsifSeventeen.generateHistogram(image, histogram);
                for (int i = 1; i < histsize; i++) histSeventeen << histogram[i] << ", ";
                histSeventeen << std::endl;
                std::fill(histogram.begin(), histogram.end(), 0);
                
                // Downsample image by 50% in either direction
                cv::Mat downImage;
                cv::pyrDown(image, downImage, cv::Size(image.cols / 2, image.rows / 2));
                
                // Run filters on downsampled image (simulates doubling of BSIF kernel sizes)
                bsifThree.generateHistogram(downImage, histogram);
                for (int i = 1; i < histsize; i++) histSix << histogram[i] << ", ";
                histSix << std::endl;
                std::fill(histogram.begin(), histogram.end(), 0);
                
                bsifFive.generateHistogram(downImage, histogram);
                for (int i = 1; i < histsize; i++) histTen << histogram[i] << ", ";
                histTen << std::endl;
                std::fill(histogram.begin(), histogram.end(), 0);
                
                bsifSeven.generateHistogram(downImage, histogram);
                for (int i = 1; i < histsize; i++) histFourteen << histogram[i] << ", ";
                histFourteen << std::endl;
                std::fill(histogram.begin(), histogram.end(), 0);
                
                bsifNine.generateHistogram(downImage, histogram);
                for (int i = 1; i < histsize; i++) histEighteen << histogram[i] << ", ";
                histEighteen << std::endl;
                std::fill(histogram.begin(), histogram.end(), 0);
                
                bsifEleven.generateHistogram(downImage, histogram);
                for (int i = 1; i < histsize; i++) histTwentyTwo << histogram[i] << ", ";
                histTwentyTwo << std::endl;
                std::fill(histogram.begin(), histogram.end(), 0);
                
                bsifThirteen.generateHistogram(downImage, histogram);
                for (int i = 1; i < histsize; i++) histTwentySix << histogram[i] << ", ";
                histTwentySix << std::endl;
                std::fill(histogram.begin(), histogram.end(), 0);
                
                bsifFifteen.generateHistogram(downImage, histogram);
                for (int i = 1; i < histsize; i++) histThirty << histogram[i] << ", ";
                histThirty << std::endl;
                std::fill(histogram.begin(), histogram.end(), 0);
                
                bsifSeventeen.generateHistogram(downImage, histogram);
                for (int i = 1; i < histsize; i++) histThirtyFour << histogram[i] << ", ";
                histThirtyFour << std::endl;
                std::fill(histogram.begin(), histogram.end(), 0);
            }
        }
        // Close csv files
        histThree.close();
        histFive.close();
        histSeven.close();
        histNine.close();
        histEleven.close();
        histThirteen.close();
        histFifteen.close();
        histSeventeen.close();
        
        histSix.close();
        histTen.close();
        histFourteen.close();
        histEighteen.close();
        histTwentyTwo.close();
        histTwentySix.close();
        histThirty.close();
        histThirtyFour.close();
    }
};

