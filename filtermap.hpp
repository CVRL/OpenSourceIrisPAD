#ifndef __FILTERMAP__
#define __FILTERMAP__

#include <map>
#include <string>

// types to represent a map of filters and pairs to add to the map
typedef std::map<std::string, double*> t_filtermap;
typedef std::pair<std::string, double*> t_filterpair;

t_filtermap build_filter_map();

#endif
