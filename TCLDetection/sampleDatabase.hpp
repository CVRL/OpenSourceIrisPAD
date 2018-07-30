//
//  sampleDatabase.hpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/27/18.


#ifndef sampleDatabase_hpp
#define sampleDatabase_hpp

#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include "CSVIterator.hpp"
#include <fstream>

using namespace std;
// Class to hold sample information
class irisSample {
public:
    std::string filename;
    std::string classification;
    std::string manufacturer;
    std::string sensor;
    
    void clear(void);
};

// Class to hold subject information
class subject {
public:
    subject();
    
    subject(string name, string none, string clear, string textured);
    
    std::vector<irisSample> mSamples;
    int numNone;
    int numClear;
    int numTextured;
    int size;
    std::string ID;
    
    std::vector<irisSample>& operator*();
};

    
class sampleDatabase {
public:
    // Create with filename for initial database parse, without for loading previously saved database
    sampleDatabase(string& Dir, string& Filename);
    
    // Create without filename for loading previously saved database
    sampleDatabase(string& Dir);
    
    // Use to store reduced database information
    void save(string& Dir);
    
    // Use to parse a user defined database
    void parseDatabaseCSV(string& sequenceIDColumnName, string& formatColumnName, string& subjectColumnName, string& textureColumnName, string& contactsColumnName, string& tagsColumnName, string& manufacturerTag, string& sensorColumnName);
    
    // Use to load previously saved database
    void load(void);
    
    // Use to access subject vector
    vector<subject>& operator*();
    
private:
    vector<subject> listOfSubjects;
    string Directory;
    string Name;
};

#endif /* sampleDatabase_hpp */
