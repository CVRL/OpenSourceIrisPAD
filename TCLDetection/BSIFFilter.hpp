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
#include <map>
#include <string>
#include <cstdio>
#include <iostream>


class BSIFFilter
{
public:
    BSIFFilter();
    
    void loadFilter(int dimension, int bitlength);
    
    void generateHistogram(cv::Mat src, std::vector<int>& histogram);
    void generateImage(cv::Mat src, cv::Mat& dst);
    
    std::string filtername;
    std::string downFiltername;
private:
    int size;
    int bits;
    double* myFilter;
};

int s2i(int size, int bits, int i, int j, int k);


// types to represent a map of filters and pairs to add to the map
typedef std::map<std::string, double*> t_filtermap;
typedef std::pair<std::string, double*> t_filterpair;

t_filtermap build_filter_map();

#endif /* BSIFFilter_hpp */
