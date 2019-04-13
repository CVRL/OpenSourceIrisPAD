//
//  featureExtractor.cpp
//  TCLDetection



#include "featureExtractor.hpp"


using namespace std;

featureExtractor::featureExtractor(int bits, vector<string>& inFilenames, std::string& segmentationType) : bitsize(bits), segmentation(segmentationType), filenames(inFilenames) {}

void featureExtractor::extract(std::string& outDir, std::string& outName, std::string& imageDir, int filtersize)
{
    outputLocation = outDir + outName;
    imageLocation = imageDir;
    
    try
    {
        filter(filtersize);
    }
    catch (runtime_error& e)
    {
        throw e;
    }
}






// Function produces features for filter size, bit size
void featureExtractor::filter(int filterSize)
{
    std::stringstream nameStream;
    nameStream << outputLocation << "_filter_" << filterSize << "_" << filterSize << "_" << bitsize << ".hdf5";
    string filtername = nameStream.str();
    
    // Open files
    //ofstream histOut;
    //histOut.open(filtername, ios::out | ios::trunc);
    
    // HDF5
    hid_t       file_id;   /* file identifier */
    herr_t      status;
    
    const char* new_filename = filtername.c_str();
    /* Create a new file using default properties. */
    file_id = H5Fcreate(new_filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    
    
    
    bool downsample = false;
    if ((filterSize % 2) == 0)
    {
        downsample = true;
        filterSize /= 2;
    }
    
    // Load filter
    BSIFFilter currentFilter;
    currentFilter.loadFilter(filterSize, bitsize);
    
    // Initialize histogram
    int histsize = pow(2,bitsize) + 1; // add one because 0 position will not be used (need 257 slots because use positions 1-256)
    std::vector<int> histogram(histsize, 0);
    
    // create dataspace
    hid_t       dataset_id, dataspace_id;  /* identifiers */
    hsize_t     dims[1];
    dims[0] = histsize - 1;
    
    
    
    dataspace_id = H5Screate_simple(1, dims, NULL);
    
    // Loop through images
    for (int i = 0; i < (int)filenames.size(); i++)
    {
        
        
        // Load image from file
        cv::Mat image = cv::imread((imageLocation + filenames[i]), 0);
        
        // create dataset
        dataset_id = H5Dcreate2(file_id, filenames[i].c_str(), H5T_STD_I64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        if ( image.empty() )
        {
            throw runtime_error("Error: unable to read image " + filenames[i] + " for feature extraction.");
        }
        
        // Segmentation
        cv::Mat imageToUse;
        if (segmentation == "wi")
        {
            imageToUse = image;
        }
        else if (segmentation == "bg")
        {
            imageToUse = image(cv::Rect(195, 125, 250, 250));
        }
        else
        {
            throw runtime_error("Error: invalid segmentation type " + segmentation);
        }
        
        // Save image information
        //histOut << filenames[i] << ",";
        //downHistOut << filenames[i] << ",";
        
        if (downsample)
        {
            // Downsample image by 50% in either direction
            cv::Mat downImage;
            cv::pyrDown(imageToUse, downImage, cv::Size(imageToUse.cols / 2, imageToUse.rows / 2));
            
            // Run filter on downsampled image (simulates doubling of BSIF kernel size)
            currentFilter.generateHistogram(downImage, histogram);
            
        }
        else
        {
            // Calculate histograms for full sized image
            currentFilter.generateHistogram(imageToUse, histogram);
            
        }
        // Ignore 0 position in histogram (image initialized to 1s in BSIFfilter so no 0s will be present)
        // Only go to (histsize - 1) to output endl after last
        //for (int i = 1; i < (histsize - 1); i++) histOut << histogram[i] << ", ";
        // Need to output endl after last column instead of ","
        //histOut << histogram[histsize - 1] << std::endl;
        
        int *array_from_vector = &histogram[1]; // skip zero slot
        status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, array_from_vector);
        
        std::fill(histogram.begin(), histogram.end(), 0);
        
        status = H5Dclose(dataset_id);
    }

    // Close files
    //histOut.close();
    /* Terminate access to the data space. */
    status = H5Sclose(dataspace_id);
    /* Terminate access to the file. */
    status = H5Fclose(file_id);
}
