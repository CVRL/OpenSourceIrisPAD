CC=g++
CFLAGS=-Wall -Wextra -std=c++11

all: main.cpp TCLManager.cpp  featureExtractor.cpp BSIFFilter.cpp
	$(CC) $(CFLAGS) main.cpp TCLManager.cpp  featureExtractor.cpp BSIFFilter.cpp -o tclDetect `pkg-config opencv --cflags --libs` -I/usr/local/opt/szip/include -L/usr/local/Cellar/hdf5/1.10.4/lib /usr/local/Cellar/hdf5/1.10.4/lib/libhdf5_hl.a /usr/local/Cellar/hdf5/1.10.4/lib/libhdf5.a -L/usr/local/opt/szip/lib -lsz -lz -ldl -lm

clean : tcl
	rm *[~o]
