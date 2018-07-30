//
//  TCLManager.cpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/17/18.

#include <stdio.h>
#include <map>
#include "OsiStringUtils.h"
#include "sampleDatabase.hpp"
#include "featureExtractor.hpp"
#include "testSeparator.cpp"

using namespace std;

class TCLManager {
public:
    TCLManager(void) {
        // Associate lines of config file to attributes
        mapBool["Extract features"] = &extractFeatures;
        mapBool["Separate sets"] = &separateSets;
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
        mapString["Sensor column name"] = &sensorColumnName;

        mapString["Set output directory"] = &setOutputDir;
        
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
        if (separateSets) {
            cout << "| Create training set |";
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
            sampleDatabase inputDatabase(inputDatabaseDir, inputDatabaseFilename);
            inputDatabase.parseDatabaseCSV(sequenceIDColumnName, formatColumnName, subjectColumnName, textureColumnName, contactsColumnName, tagsColumnName, manufacturerTag, sensorColumnName);
            inputDatabase.save(outputExtractionDir);
            
            featureExtractor newExtractor(bitsize);
            newExtractor.extract(outputExtractionDir, outputExtractionFilename, databaseImageDir, inputDatabase);
            
            // Separate the sets if required
            if (separateSets) {
                std::cout << "Creating training set..." << std::endl;
                // No need to reload inputDatabase, can use the one defined above
                
                // Separate, not subject disjoint
                //testSeparator newSets(loadDatabase);
                //newSets.separate(false, 6000, setOutputDir);
                
                // Separate as subject disjoint
                //testSeparator newSets(inputDatabase);
                //newSets.separate(true, 6000, setOutputDir);
                
                // Separate with specific requirements
                //testSeparator newSets(inputDatabase);
                
            }
        }
        
        if (separateSets && !extractFeatures) {
            std::cout << "Creating training set..." << std::endl;
            
            // Need to reload the inputDatabase
            sampleDatabase loadDatabase(outputExtractionDir);
            loadDatabase.load();
            
            // Separate, not subject disjoint
            //testSeparator newSets(loadDatabase);
            //newSets.separate(false, 6000, setOutputDir);
            
            // Separate as subject disjoint
            //testSeparator newSets(loadDatabase);
            //newSets.separate(true, 6000, setOutputDir);
            
            // Separate with specific requirements
            testSeparator newSets(loadDatabase);
            newSets.separate("exclude-m", "CibaVision,Coopervision", 6000, setOutputDir);
            
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
    bool separateSets;
    bool trainModel;
    bool testImages;
    
    
    // Inputs
    std::string inputDatabaseFilename;
    std::string inputDatabaseDir;
    std::string databaseImageDir;
    
    // Outputs
    std::string outputExtractionFilename;
    std::string outputExtractionDir;
    std::string outputHistFilename;
    std::string outputHistDir;
    std::string setOutputDir;
    
    
    // Parameters
    int bitsize;
    std::string sequenceIDColumnName;
    std::string formatColumnName;
    std::string subjectColumnName;
    std::string textureColumnName;
    std::string contactsColumnName;
    std::string tagsColumnName;
    std::string manufacturerTag;
    std::string sensorColumnName;
    
    // Maps to associate a string (conf file) to a variable (pointer)
    std::map<std::string,bool*> mapBool;
    std::map<std::string,int*> mapInt;
    std::map<std::string,std::string*> mapString;
    
    // Initialize all parameters
    void initConfig(void) {
        // Commands
        extractFeatures = false;
        separateSets = false;
        trainModel = false;
        testImages = false;
        
        // Inputs
        inputDatabaseFilename = "";
        inputDatabaseDir = "";
        databaseImageDir = "";
        
        // Outputs
        outputExtractionFilename = "";
        outputExtractionDir = "";
        outputHistFilename = "";
        outputHistDir = "";
        setOutputDir = "";
        
        // Parameters
        bitsize = 8;
        
        sequenceIDColumnName = "";
        formatColumnName = "";
        subjectColumnName = "";
        textureColumnName = "";
        contactsColumnName = "";
        tagsColumnName = "";
        manufacturerTag = "";
        sensorColumnName = "";
    }
};

