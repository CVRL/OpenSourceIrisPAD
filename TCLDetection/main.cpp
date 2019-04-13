//
//  main.cpp
//  TCLDetection



#include "TCLManager.hpp"


using namespace std;


int main(int argc, char *argv[]) {
    
    if (argc != 2)
    {
        cout << "Usage: tclDetect config.ini" << endl;
        cout << "Please provide configuration filename" << endl;
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
    catch (runtime_error& e)
    {
        cout << e.what() << endl;
    }
    
    return 0;
    
    
}
