//
//  main.cpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/11/18.


#include "TCLManager.hpp"


using namespace std;


int main(int argc, char *argv[]) {
    
    if (argc != 2)
    {
        cout << "Usage: TCLDetection config.ini" << endl;
        cout << "Please provide configuration file" << endl;
        return 0;
    }
    
    // Load TCLManager
    TCLManager tcl;
    
    
    // Load configuration
    tcl.loadConfig(argv[1]);
    // Show configuration
    tcl.showConfig();
    
    try
    {
        tcl.run();
    }
    catch (exception& e)
    {
        cout << e.what() << endl;
    }
    
    return 0;
    
    
}
