CC=gcc

all: main.cpp TCLManager.cpp  featureExtractor.cpp CSVIterator.cpp filtermap.cpp BSIFFilter.cpp
	g++ main.cpp TCLManager.cpp  featureExtractor.cpp CSVIterator.cpp filtermap.cpp BSIFFilter.cpp -o tclDetect `pkg-config opencv --cflags --libs`

clean : tcl
	rm *[~o]
