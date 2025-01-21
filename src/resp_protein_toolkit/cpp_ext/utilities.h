#ifndef UTILITIES_HEADER_H
#define UTILITIES_HEADER_H

#include <string>
#include <vector>
#include <tuple>
#include <nanobind/stl/vector.h>    // Enables automatic type conversion for C++, python containers

int get_max_length(std::vector<std::string> sequenceList, bool allSameLength);


#endif
