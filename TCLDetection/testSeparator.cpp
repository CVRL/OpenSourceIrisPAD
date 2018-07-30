//
//  testSeparator.cpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/28/18.

#include <opencv2/core/core.hpp>
#include "testSeparator.hpp"
#include "sampleDatabase.hpp"
//#include <time.h>

class testSeparator {
public:
    testSeparator(sampleDatabase& database) : data(database) {
        setsLoaded = false;
    };
    
    // All separate functions will create a text file for training and test sets in the training folder
    // Assumes an even class division (6000 training samples -> 2000 from each of 3 classes)
    void separate(bool subjectDisjoint, int trainSize, string& Dir) {
        if (!subjectDisjoint) {
            // Set starting numbers of each class
            int numNone = trainSize/3;
            int numClear = trainSize/3;
            int numText = trainSize/3;
            
            // Loop through subjects and samples
            for (int i = 0; i < (*data).size(); i++) {
                for (int j = 0; j < (*(*data)[i]).size(); j++) {
                    if ((*(*data)[i])[j].classification == "none") {
                        if (numNone > 0) {
                            trainingSet.push_back((*(*data)[i])[j].filename);
                            numNone--;
                        } else {
                            testingSet.push_back((*(*data)[i])[j].filename);
                        }
                    } else if ((*(*data)[i])[j].classification == "clear") {
                        if (numClear > 0) {
                            trainingSet.push_back((*(*data)[i])[j].filename);
                            numClear--;
                        } else {
                            testingSet.push_back((*(*data)[i])[j].filename);
                        }
                    } else if ((*(*data)[i])[j].classification == "textured") {
                        if (numText > 0) {
                            trainingSet.push_back((*(*data)[i])[j].filename);
                            numText--;
                        } else {
                            testingSet.push_back((*(*data)[i])[j].filename);
                        }
                    }
                }
            }
            
            // Save files
            save(Dir);
            
            setsLoaded = true;
            
        } else {
            // Need to use subset sum algorithm to generate a subject disjoint set
            
            // First, sort the database (makes slightly faster)
            std::sort((*data).begin(), (*data).end(), [](const subject& lhs, const subject& rhs) {
                return lhs.size < rhs.size;
            });
            
            // Now use set finder algorithm
            //clock_t start = clock();
            findDisjointSet(trainSize);
            //clock_t end = clock();
            //double time = (double) (end - start)/(CLOCKS_PER_SEC * 1000.0);
            //cout << "Algorithm took: " << time << " s" << endl;
            
            // Check results and display
            checkSize();
            
            // Load samples into training and testing sets
            // Loop through subjects and samples
            for (int i = 0; i < (*data).size(); i++) {
                // Can use binary search since the trainingIdx will be sorted by default
                if (binary_search(trainingIdx.begin(), trainingIdx.end(), i)) {
                    for (int j = 0; j < (*(*data)[i]).size(); j++) {
                        trainingSet.push_back((*(*data)[i])[j].filename);
                    }
                } else {
                    for (int j = 0; j < (*(*data)[i]).size(); j++) {
                        testingSet.push_back((*(*data)[i])[j].filename);
                    }
                }
            }
            
            // Save files
            save(Dir);
            
            setsLoaded = true;
        }
    }
    
    // Use config to select include or exclude manufacturers or sensors
    // Names should be a list of items to include separated by commas (i.e. "CibaVision,Coopervision")
    // Not subject disjoint for now (possibly add capability later)
    void separate(string config, string names, int trainSize, string& Dir) {
        // Read names and store in accepted or disallowed vector
        vector<string> nameVector;
        
        stringstream nameStream(names);
        string singleName = "";
        
        while (getline(nameStream, singleName, ',')) {
            nameVector.push_back(singleName);
        }
        
        if (config == "include-m") {
            
            // Only have size requirements for non-textured classes
            int numNone = trainSize/3;
            int numClear = trainSize/3;
            
            // Loop through subjects and samples
            for (int i = 0; i < (*data).size(); i++) {
                for (int j = 0; j < (*(*data)[i]).size(); j++) {
                    if ((*(*data)[i])[j].classification == "none") {
                        if (numNone > 0) {
                            trainingSet.push_back((*(*data)[i])[j].filename);
                            numNone--;
                        } else {
                            testingSet.push_back((*(*data)[i])[j].filename);
                        }
                    } else if ((*(*data)[i])[j].classification == "clear") {
                        if (numClear > 0) {
                            trainingSet.push_back((*(*data)[i])[j].filename);
                            numClear--;
                        } else {
                            testingSet.push_back((*(*data)[i])[j].filename);
                        }
                    } else if ((*(*data)[i])[j].classification == "textured") {
                        // Include all samples from specified manufacturers
                        if (find(nameVector.begin(), nameVector.end(), (*(*data)[i])[j].manufacturer) != nameVector.end()) {
                            trainingSet.push_back((*(*data)[i])[j].filename);
                            cout << (*(*data)[i])[j].manufacturer << endl;
                        } else {
                            testingSet.push_back((*(*data)[i])[j].filename);
                        }
                    }
                }
            }
            
        } else if (config == "include-s") {
            
            // Loop through subjects and samples
            for (int i = 0; i < (*data).size(); i++) {
                for (int j = 0; j < (*(*data)[i]).size(); j++) {
                    // Include all samples from specified sensors
                    if (find(nameVector.begin(), nameVector.end(), (*(*data)[i])[j].sensor) != nameVector.end()) {
                        trainingSet.push_back((*(*data)[i])[j].filename);
                    } else {
                        testingSet.push_back((*(*data)[i])[j].filename);
                    }
                }
            }
            
        } else if (config == "exclude-m") {
            
            // Only have size requirements for non-textured classes
            int numNone = trainSize/3;
            int numClear = trainSize/3;
            
            // Loop through subjects and samples
            for (int i = 0; i < (*data).size(); i++) {
                for (int j = 0; j < (*(*data)[i]).size(); j++) {
                    if ((*(*data)[i])[j].classification == "none") {
                        if (numNone > 0) {
                            trainingSet.push_back((*(*data)[i])[j].filename);
                            numNone--;
                        } else {
                            testingSet.push_back((*(*data)[i])[j].filename);
                        }
                    } else if ((*(*data)[i])[j].classification == "clear") {
                        if (numClear > 0) {
                            trainingSet.push_back((*(*data)[i])[j].filename);
                            numClear--;
                        } else {
                            testingSet.push_back((*(*data)[i])[j].filename);
                        }
                    } else if ((*(*data)[i])[j].classification == "textured") {
                        // Exclude all samples from specified manufacturers
                        if (find(nameVector.begin(), nameVector.end(), (*(*data)[i])[j].manufacturer) == nameVector.end()) {
                            trainingSet.push_back((*(*data)[i])[j].filename);
                        } else {
                            testingSet.push_back((*(*data)[i])[j].filename);
                        }
                    }
                }
            }
        } else if (config == "exclude-s") {
            
            // Loop through subjects and samples
            for (int i = 0; i < (*data).size(); i++) {
                for (int j = 0; j < (*(*data)[i]).size(); j++) {
                    // Exclude all samples from specified sensors
                    if (find(nameVector.begin(), nameVector.end(), (*(*data)[i])[j].sensor) == nameVector.end()) {
                        trainingSet.push_back((*(*data)[i])[j].filename);
                    } else {
                        testingSet.push_back((*(*data)[i])[j].filename);
                    }
                }
            }
            
        }
        
        // Save files
        save(Dir);
    }
    
    // Load the training feature set
    void loadTraining(cv::Mat& output) {
        
    }
    
private:
    // Database object to be used
    sampleDatabase& data;
    
    // List of filenames for each set
    vector<string> trainingSet;
    vector<string> testingSet;
    
    // Training set index list
    vector<int> trainingIdx;
    
    // Track current state
    bool setsLoaded;
    
    // Create subject disjoint sets (algorithm can be kind of touchy, some values don't work well)
    void findDisjointSet(int trainSize) {
        int i = 0;
        // Set starting numbers of each class
        int numNone = trainSize/3;
        int numClear = trainSize/3;
        int numText = 1980;
        
        
        while ((numNone + numClear + numText) > 0) {
            if (i == (*data).size()) {
                // Reached the end without succeeding
                // Start i back at the last successful addition
                i = trainingIdx.back();
                
                // Remove last successful addition
                numNone += (*data)[i].numNone;
                numClear += (*data)[i].numClear;
                numText += (*data)[i].numTextured;
                trainingIdx.pop_back();
                
                // Skip this element and try again
                i++;
                
            } else if (((numNone - (*data)[i].numNone) > -1) && ((numClear - (*data)[i].numClear) > -1) && ((numText - (*data)[i].numTextured) > -1)) {
                // Add the element if you can
                trainingIdx.push_back(i);
                numNone -= (*data)[i].numNone;
                numClear -= (*data)[i].numClear;
                numText -= (*data)[i].numTextured;
                
                i++;
            } else {
                // If you can't add the element and you haven't reached the end, try the next one
                i++;
            }
        }
    }
    
    void checkSize(void) {
        int checkSum = 0;
        int checkNone = 0;
        int checkClear = 0;
        int checkText = 0;
        for (int i = 0; i < trainingIdx.size(); i++) {
            checkSum += (*data)[trainingIdx[i]].size;
            checkNone += (*data)[trainingIdx[i]].numNone;
            checkClear += (*data)[trainingIdx[i]].numClear;
            checkText += (*data)[trainingIdx[i]].numTextured;
        }
        cout << "Total training samples: " << checkSum << endl;
        cout << "   No contacts - " << checkNone << endl;
        cout << "   Clear contacts - " << checkClear << endl;
        cout << "   Textured - " << checkText << endl;
    }
    
    void save(string& Dir) {
        // Save result in a text file
        ofstream train;
        train.open((Dir + "trainList.txt"), ios::out | ios::trunc);
        for (int i = 0; i < trainingSet.size(); i++) {
            train << trainingSet[i] << endl;
        }
        train.close();
        
        ofstream test;
        test.open((Dir + "testList.txt"), ios::out | ios::trunc);
        for (int j = 0; j < testingSet.size(); j++) {
            test << testingSet[j] << endl;
        }
    }
};
