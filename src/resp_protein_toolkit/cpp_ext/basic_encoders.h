#ifndef BASIC_TOKENIZERS_HEADER_H
#define BASIC_TOKENIZERS_HEADER_H

#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>    // Enables automatic type conversion for C++, python containers
#include <vector>
#include <string>


namespace nb = nanobind;




int onehot_flat_encode_list(std::vector<std::string> sequenceList,
        nb::ndarray<uint8_t, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> ouputArray,
        bool expandedSymbolSet, bool addGaps);

int onehot_3d_encode_list(std::vector<std::string> sequenceList,
        nb::ndarray<uint8_t, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> outputArray,
        bool expandedSymbolSet, bool addGaps);

int integer_encode_list(std::vector<std::string> sequenceList,
        nb::ndarray<uint8_t, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArray,
        bool expandedSymbolSet, bool addGaps);

int expandedSymbolSetCharReader(char &letter);
int gappedSymbolSetCharReader(char &letter);
int standardSymbolSetCharReader(char &letter);

#endif
