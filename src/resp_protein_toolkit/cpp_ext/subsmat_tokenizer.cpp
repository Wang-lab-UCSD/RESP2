#include "subsmat_tokenizer.h"

namespace nb = nanobind;

// Codes for sequence validation.
#define VALID_SEQUENCE 1
#define INVALID_SEQUENCE 0

//The substitution matrix always has an alphabet size of 21.
#define STD_ALPHABET_SIZE 21


// Encodes the input sequence list using a 2d "flat" array.
int subsmat_flat_encode_list(std::vector<std::string> sequenceList,
        nb::ndarray<float, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArray,
        nb::ndarray<float, nb::shape<STD_ALPHABET_SIZE,STD_ALPHABET_SIZE>,
                nb::device::cpu, nb::c_contig> aaTokens
        ){
    if (sequenceList.size() != outputArray.shape(0))
        return INVALID_SEQUENCE;

    int errorCode = subsmat_encode_array(outputArray.data(), aaTokens.data(),
            sequenceList, outputArray.shape(1));

    return errorCode;
}



// Encodes the input sequence list using a 3d array.
int subsmat_3d_encode_list(std::vector<std::string> sequenceList,
        nb::ndarray<float, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> outputArray,
        nb::ndarray<float, nb::shape<STD_ALPHABET_SIZE,STD_ALPHABET_SIZE>,
                        nb::device::cpu, nb::c_contig> aaTokens
        ){
    if (sequenceList.size() != outputArray.shape(0))
        return INVALID_SEQUENCE;

    if (outputArray.shape(2) != STD_ALPHABET_SIZE)
        return INVALID_SEQUENCE;

    int errorCode = subsmat_encode_array(outputArray.data(), aaTokens.data(),
            sequenceList, outputArray.shape(1) * STD_ALPHABET_SIZE);

    return errorCode;
}



// Notice that 3d numpy arrays and 2d numpy arrays -- as long as c-contiguous
// -- are the same in memory, and we can encode both using the same function,
// so we do this to avoid redundancy.
int subsmat_encode_array(float *outputArray, float *aaTokens,
        std::vector<std::string> &sequenceList, size_t outputDim1){

    for (size_t i=0; i < sequenceList.size(); i++){
        if ( (sequenceList[i].length() * STD_ALPHABET_SIZE) > outputDim1)
            return INVALID_SEQUENCE;

        float *currentElement = outputArray + i * outputDim1;

        for (size_t j=0; j < sequenceList[i].length(); j++){
            int positionCode = symbolSetCharReader(sequenceList[i][j]);
            if (positionCode < 0)
                return INVALID_SEQUENCE;

            float *currentAAToken = aaTokens + positionCode * STD_ALPHABET_SIZE;
            for (size_t k=0; k < STD_ALPHABET_SIZE; k++){
                *currentElement = currentAAToken[k];
                currentElement++;
            }
        }
    }

    return VALID_SEQUENCE;
}



// Returns a position code for the gapped symbol set. A value
// < 0 indicates an invalid character.
int symbolSetCharReader(char &letter){
    int positionCode;

    switch (letter){
        case 'A':
            positionCode = 0;
            break;
        case 'C':
            positionCode = 1;
            break;
        case 'D':
            positionCode = 2;
            break;
        case 'E':
            positionCode = 3;
            break;
        case 'F':
            positionCode = 4;
            break;
        case 'G':
            positionCode = 5;
            break;
        case 'H':
            positionCode = 6;
            break;
        case 'I':
            positionCode = 7;
            break;
        case 'K':
            positionCode = 8;
            break;
        case 'L':
            positionCode = 9;
            break;
        case 'M':
            positionCode = 10;
            break;
        case 'N':
            positionCode = 11;
            break;
        case 'P':
            positionCode = 12;
            break;
        case 'Q':
            positionCode = 13;
            break;
        case 'R':
            positionCode = 14;
            break;
        case 'S':
            positionCode = 15;
            break;
        case 'T':
            positionCode = 16;
            break;
        case 'V':
            positionCode = 17;
            break;
        case 'W':
            positionCode = 18;
            break;
        case 'Y':
            positionCode = 19;
            break;
        case '-':
            positionCode = 20;
            break;
        default:
            return -1;
            break;
    }
    return positionCode;
}
