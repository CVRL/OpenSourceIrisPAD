//
//  main.cpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/11/18.


#include "TCLManager.hpp"


using namespace std;


int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Error: Please enter correct arguments." << endl;
    }
    
    // Load TCLManager
    TCLManager tcl;
    
    
    // Pass configuration file
    tcl.loadConfig(argv[1]);
    // Show configuration
    tcl.showConfig();
    
    
    tcl.run();
    
    
    return 0;
    
    
}
