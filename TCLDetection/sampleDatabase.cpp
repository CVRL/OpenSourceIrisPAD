//
//  sampleDatabase.cpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/27/18.


#include "sampleDatabase.hpp"

using namespace std;

// Class to hold sample information
void irisSample::clear(void) {
    filename = "";
    classification = "";
    manufacturer = "";
    sensor = "";
}


// Class to hold subject information
subject::subject() {};
    
subject::subject(string name, string none, string clear, string textured) : numNone(stoi(none)), numClear(stoi(clear)), numTextured(stoi(textured)), ID(name) {
    size = numNone + numClear + numTextured;
};

    
vector<irisSample>& subject::operator*() {return mSamples;}



// Database object definitions

// Create with filename for initial database parse, without for loading previously saved database
sampleDatabase::sampleDatabase(string& Dir, string& Filename) : Directory(Dir), Name(Filename) {}
    
sampleDatabase::sampleDatabase(string& Dir) : Directory(Dir), Name("subjectList.csv"){}
    
// Use to store reduced database information
void sampleDatabase::save(string& Dir) {
    // Store list of subjects (separate method)
    ofstream subjectList;
    subjectList.open((Dir + "subjectList.csv"), ios::out | ios::trunc);
    for (int i = 0; i < listOfSubjects.size(); i++) {
        subject cs = listOfSubjects[i];
        subjectList << cs.ID << ", " << cs.numNone << "," << cs.numClear << "," << cs.numTextured;
        for (int j = 0; j < (*cs).size(); j++) {
            subjectList << "," << "\"" << (*cs)[j].filename << "\\,"<< (*cs)[j].classification << "\\," << (*cs)[j].manufacturer << "\\," << (*cs)[j].sensor << "\"";
        }
        subjectList << endl;
    }
    subjectList.close();
}
    
// Use to parse a user defined database
void sampleDatabase::parseDatabaseCSV(string& sequenceIDColumnName, string& formatColumnName, string& subjectColumnName, string& textureColumnName, string& contactsColumnName, string& tagsColumnName, string& manufacturerTag, string& sensorColumnName) {
    // Set up file input and CSVIterator
    ifstream databaseFile(Directory + Name);
    CSVIterator databaseCSV(databaseFile);
    
    // Find the indices of the desired columns
    int numDesiredColumns = 7;
    int sequenceidIdx = 0;
    int formatIdx = 0;
    int subjectIdx = 0;
    int textureIdx = 0;
    int contactsIdx = 0;
    int tagsIdx = 0;
    int sensorIdx = 0;
    
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
        } else if ((*databaseCSV)[i] == sensorColumnName) {
            sensorIdx = i;
            numDesiredColumns--;
        }
    }
    
    if (numDesiredColumns == 0) {
        // Increment databaseCSV to skip header line
        databaseCSV++;
        
        // Step through database and add to list of samples
        while (databaseCSV != CSVIterator()) {
            irisSample currentSample;
            
            // Set filename
            currentSample.filename = (*databaseCSV)[sequenceidIdx] + "." + (*databaseCSV)[formatIdx];
            
            
            // Determine classification
            bool contacts = false;
            bool textured = false;
            
            if ((*databaseCSV)[contactsIdx] == "Yes") {contacts = true;}
            if ((*databaseCSV)[textureIdx] == "Yes") {textured = true;}
            
            
            if (contacts && textured) {
                currentSample.classification = "textured";
            } else if (contacts) {
                currentSample.classification = "clear";
            } else {
                currentSample.classification = "none";
            }
            
            // Determine sensor
            if ((*databaseCSV)[sensorIdx] == "nd1N00006") {
                currentSample.sensor = "LG2200";
            } else if ((*databaseCSV)[sensorIdx] == "nd1N00020") {
                currentSample.sensor = "LG4000";
            } else if ((*databaseCSV)[sensorIdx] == "nd1N00049") {
                currentSample.sensor = "IGAD100";
            } else if ((*databaseCSV)[sensorIdx] == "nd1N00074") {
                currentSample.sensor = "LG4000";
            } else if ((*databaseCSV)[sensorIdx] == "nd1N00079") {
                currentSample.sensor = "IGAD100";
            } else if ((*databaseCSV)[sensorIdx] == "nd1N00077") {
                currentSample.sensor = "IGAD100";
            } else {
                currentSample.sensor = (*databaseCSV)[sensorIdx];
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
                newSubject.size = newSubject.numNone + newSubject.numClear + newSubject.numTextured;
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
void sampleDatabase::load(void) {
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
    
vector<subject>& sampleDatabase::operator*() {return listOfSubjects;}

