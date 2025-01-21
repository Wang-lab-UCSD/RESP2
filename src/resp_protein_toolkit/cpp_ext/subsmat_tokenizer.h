#ifndef IG_ALIGNERS_HEADER_H
#define IG_ALIGNERS_HEADER_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>    // Enables automatic type conversion for C++, python containers
#include <vector>
#include <string>



namespace nb = nanobind;



int subsmat_flat_encode_list(std::vector<std::string> sequenceList,
                nb::ndarray<float, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArray,
                nb::ndarray<float, nb::shape<21,21>, nb::device::cpu, nb::c_contig> aaTokens);
int subsmat_3d_encode_list(std::vector<std::string> sequenceList,
                nb::ndarray<float, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> outputArray,
                nb::ndarray<float, nb::shape<21,21>, nb::device::cpu, nb::c_contig> aaTokens);

int subsmat_encode_array(float *outputArray, float *aaTokens,
        std::vector<std::string> &sequenceList, size_t maxAAs);
int symbolSetCharReader(char &letter);

#endif
