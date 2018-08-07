//
//  main.cpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/11/18.


#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>
#include <regex>
#include <fstream>
//#include "CSVIterator.hpp"
//#include "BSIFFilter.hpp"
#include "TCLManager.cpp"


using namespace std;


int main(int argc, char *argv[]) {
  
    
    
    // Load TCLManager
    TCLManager tcl;
    
    // Pass configuration file
    tcl.loadConfig(argv[1]);
    // Show configuration
    tcl.showConfig();
    
    tcl.run();
    
    
    
    
    return 0;
    
    
}
