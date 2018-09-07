CC=g++
CFLAGS=-Wall -Wextra

all: main.cpp TCLManager.cpp  featureExtractor.cpp BSIFFilter.cpp
	$(CC) $(CFLAGS) main.cpp TCLManager.cpp  featureExtractor.cpp BSIFFilter.cpp -o tclDetect `pkg-config opencv --cflags --libs`

clean : tcl
	rm *[~o]
