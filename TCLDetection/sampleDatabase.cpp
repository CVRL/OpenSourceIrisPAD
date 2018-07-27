//
//  sampleDatabase.cpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/27/18.
//  Copyright © 2018 Joseph McGrath. All rights reserved.
//

#include "sampleDatabase.hpp"
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
    subject() {};
    
    subject(string name, string none, string clear, string textured) : numNone(stoi(none)), numClear(stoi(clear)), numTextured(stoi(textured)), ID(name) {};
    
    std::vector<irisSample> mSamples;
    int numNone;
    int numClear;
    int numTextured;
    std::string ID;
            
    int size(void) {
        return (int)mSamples.size();
    }
};



class sampleDatabase {
public:
    // Create with filename for initial database parse, without for loading previously saved database
    sampleDatabase(string& Dir, string& Filename) : Directory(Dir), Name(Filename) {}
    
    sampleDatabase(string& Dir) : Directory(Dir), Name("subjectList.csv"){}
    
    // Use to store reduced database information
    void save(string& Dir) {
        // Store list of subjects (separate method)
        ofstream subjectList;
        subjectList.open((Dir + "subjectList.csv"), ios::out | ios::trunc);
        for (int i = 0; i < listOfSubjects.size(); i++) {
            subject cs = listOfSubjects[i];
            subjectList << cs.ID << ", " << cs.numNone << ", " << cs.numClear << ", " << cs.numTextured;
            for (int j = 0; j < cs.size(); j++) {
                irisSample cSam = cs.mSamples.at(j);
                subjectList << ", " << "\"" << cSam.filename << "\\,"<< cSam.classification << "\\," << cSam.manufacturer << "\\," << cSam.sensor << "\"";
            }
            subjectList << endl;
        }
        subjectList.close();
    }
    
    // Use to parse a user defined database
    void parseDatabaseCSV(string& sequenceIDColumnName, string& formatColumnName, string& subjectColumnName, string& textureColumnName, string& contactsColumnName, string& tagsColumnName, string& manufacturerTag) {
        // Set up file input and CSVIterator
        ifstream databaseFile(Directory + Name);
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
                
                if ((*databaseCSV)[contactsIdx] == "Yes") {contacts = true;}
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
    }
    
    // Use to load previously saved database
    void load(void) {
        // Set up file input and CSVIterator
        ifstream databaseFile(Directory + Name);
        CSVIterator databaseCSV(databaseFile);
        
        // Loop through lines to read in data
        while (databaseCSV != CSVIterator()) {
            subject newSubject((*databaseCSV)[0], (*databaseCSV)[1], (*databaseCSV)[2], (*databaseCSV)[3]);
            for (int i = 4; i < (*databaseCSV).size(); i++) {
                // Extract information from each sample
                std::stringstream sampleStream((*databaseCSV)[i]);
                string tag;
                
                irisSample newSample;
                std::getline(sampleStream, newSample.filename, '\\');
                std::getline(sampleStream, newSample.classification, '\\');
                std::getline(sampleStream, newSample.manufacturer, '\\');
                std::getline(sampleStream, newSample.sensor, '\\');
                newSubject.mSamples.push_back(newSample);
            }
            listOfSubjects.push_back(newSubject);
            databaseCSV++;
        }
        
    }
    
private:
    vector<subject> listOfSubjects;
    string Directory;
    string Name;
};

