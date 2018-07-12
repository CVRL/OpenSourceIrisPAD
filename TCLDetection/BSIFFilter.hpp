//
//  BSIFFilter.hpp
//  TCLDetection
//
//  Created by Joseph McGrath on 7/11/18.


#ifndef BSIFFilter_hpp
#define BSIFFilter_hpp

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "filtermap.hpp"

class BSIFFilter {
public:
    BSIFFilter(int dimension, int bitlength);
    BSIFFilter();
    
    void generateHistogram(cv::Mat src, std::vector<int>& histogram);
    void generateImage(cv::Mat src, cv::Mat& dst);
private:
    int size;
    int bits;
    double* myFilter;
};

int s2i(int size, int bits, int i, int j, int k);
#endif /* BSIFFilter_hpp */
