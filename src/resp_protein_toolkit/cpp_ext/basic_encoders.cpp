/* Includes the functions for basic tokenizers -- one hot, substitution
 * matrix and integer.*/
#include "basic_encoders.h"

namespace nb = nanobind;

// Codes for sequence validation.
#define VALID_SEQUENCE 1
#define INVALID_SEQUENCE 0


// One-hot encodes a list of input sequences in a 2d "flat" array. This is generally
// only useful if dealing with an MSA; for sequences of different length, a 3d array
// with zero padding at the end is likely better. The sequences
// are checked to ensure they are all less than the max length and have
// expected characters. If expandedSymbolSet, unusual AAs like O, U, X, J, B, Z and
// gaps are allowed; otherwise if addGaps, gaps are allowed. The array should
// be all zeros (this is not checked).
int onehot_flat_encode_list(std::vector<std::string> sequenceList,
        nb::ndarray<uint8_t, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArray,
        bool expandedSymbolSet, bool addGaps){

    auto arrView = outputArray.view();

    if (sequenceList.size() != outputArray.shape(0))
        return INVALID_SEQUENCE;

    // We use switches to convert characters into integers. This should
    // (compiler-dependent but on nearly any modern compiler) be slightly
    // faster than using std::map (although the performance difference will
    // be very small in general). It is however more verbose...

    if (expandedSymbolSet){
        size_t alphabetSize = 27;

        for (size_t i=0; i < sequenceList.size(); i++){
            if ((sequenceList[i].length() * alphabetSize) > arrView.shape(1))
                return INVALID_SEQUENCE;

            int sequencePosition = 0;

            for (size_t j=0; j < sequenceList[i].length(); j++){
                // We can't inline this function call because it contains a switch,
                // so there is very probably some very small amount of overhead.
                // We're accepting that for now as the price of more readability...
                int positionCode = expandedSymbolSetCharReader(sequenceList[i][j]);
                if (positionCode < 0)
                    return INVALID_SEQUENCE;
                arrView(i, sequencePosition+positionCode) = 1;
                sequencePosition += alphabetSize;
            }
        }
    }

    else if (addGaps){
        size_t alphabetSize = 21;

        for (size_t i=0; i < sequenceList.size(); i++){
            if ((sequenceList[i].length() * alphabetSize) > arrView.shape(1))
                return INVALID_SEQUENCE;

            int sequencePosition = 0;

            for (size_t j=0; j < sequenceList[i].length(); j++){
                // We can't inline this function call because it contains a switch,
                // so there is very probably some very small amount of overhead.
                // We're accepting that for now as the price of more readability...
                int positionCode = gappedSymbolSetCharReader(sequenceList[i][j]);
                if (positionCode < 0)
                    return INVALID_SEQUENCE;

                arrView(i, sequencePosition+positionCode) = 1;
                sequencePosition += alphabetSize;
            }
        }
    }

    else{
        size_t alphabetSize = 20;

        for (size_t i=0; i < sequenceList.size(); i++){
            if ((sequenceList[i].length() * alphabetSize) > arrView.shape(1))
                return INVALID_SEQUENCE;

            int sequencePosition = 0;

            for (size_t j=0; j < sequenceList[i].length(); j++){
                // We can't inline this function call because it contains a switch,
                // so there is very probably some very small amount of overhead.
                // We're accepting that for now as the price of more readability...
                int positionCode = standardSymbolSetCharReader(sequenceList[i][j]);
                if (positionCode < 0)
                    return INVALID_SEQUENCE;
                arrView(i, sequencePosition+positionCode) = 1;
                sequencePosition += alphabetSize;
            }
        }
    }
    return VALID_SEQUENCE;
}



// One-hot encodes a list of input sequences in a 3d array. The sequences
// are checked to ensure they are all less than the max length and have
// expected characters. If expandedSymbolSet, unusual AAs like O, U, X, J, B, Z and
// gaps are allowed; otherwise if addGaps, gaps are allowed. The array should
// be all zeros (this is not checked).
int onehot_3d_encode_list(std::vector<std::string> sequenceList,
        nb::ndarray<uint8_t, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> outputArray,
        bool expandedSymbolSet, bool addGaps){

    auto arrView = outputArray.view();

    if (sequenceList.size() != outputArray.shape(0))
        return INVALID_SEQUENCE;

    // We use switches to convert characters into integers. This should
    // (compiler-dependent but on nearly any modern compiler) be slightly
    // faster than using std::map (although the performance difference will
    // be very small in general). It is however more verbose...

    if (expandedSymbolSet){
        size_t alphabetSize = 27;
        if (arrView.shape(2) != alphabetSize)
            return INVALID_SEQUENCE;

        for (size_t i=0; i < sequenceList.size(); i++){
            if ((sequenceList[i].length()) > arrView.shape(1))
                return INVALID_SEQUENCE;

            for (size_t j=0; j < sequenceList[i].length(); j++){
                // We can't inline this function call because it contains a switch,
                // so there is very probably some very small amount of overhead.
                // We're accepting that for now as the price of more readability...
                int positionCode = expandedSymbolSetCharReader(sequenceList[i][j]);
                if (positionCode < 0)
                    return INVALID_SEQUENCE;
                arrView(i, j, positionCode) = 1;
            }
        }
    }

    else if (addGaps){
        size_t alphabetSize = 21;
        if (arrView.shape(2) != alphabetSize)
            return INVALID_SEQUENCE;

        for (size_t i=0; i < sequenceList.size(); i++){
            if ((sequenceList[i].length()) > arrView.shape(1))
                return INVALID_SEQUENCE;

            for (size_t j=0; j < sequenceList[i].length(); j++){
                // We can't inline this function call because it contains a switch,
                // so there is very probably some very small amount of overhead.
                // We're accepting that for now as the price of more readability...
                int positionCode = gappedSymbolSetCharReader(sequenceList[i][j]);
                if (positionCode < 0)
                    return INVALID_SEQUENCE;
                arrView(i, j, positionCode) = 1;
            }
        }
    }

    else{
        size_t alphabetSize = 20;
        if (arrView.shape(2) != alphabetSize)
            return INVALID_SEQUENCE;

        for (size_t i=0; i < sequenceList.size(); i++){
            if ((sequenceList[i].length()) > arrView.shape(1))
                return INVALID_SEQUENCE;

            for (size_t j=0; j < sequenceList[i].length(); j++){
                // We can't inline this function call because it contains a switch,
                // so there is very probably some very small amount of overhead.
                // We're accepting that for now as the price of more readability...
                int positionCode = standardSymbolSetCharReader(sequenceList[i][j]);
                if (positionCode < 0)
                    return INVALID_SEQUENCE;
                arrView(i, j, positionCode) = 1;
            }
        }
    }
    return VALID_SEQUENCE;
}





// Integer-encodes a list of input sequences in a 2d array. This is useful for
// LightGBM and some kinds of clustering procedures. The sequences
// are checked to ensure they are all less than the max length and have
// expected characters. If expandedSymbolSet, unusual AAs like O, U, X, J, B, Z and
// gaps are allowed; otherwise if addGaps, gaps are allowed. In this case,
// it does not matter if the sequence is all zeros or not.
int integer_encode_list(std::vector<std::string> sequenceList,
        nb::ndarray<uint8_t, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArray,
        bool expandedSymbolSet, bool addGaps){

    auto arrView = outputArray.view();

    if (sequenceList.size() != outputArray.shape(0))
        return INVALID_SEQUENCE;

    // We use switches to convert characters into integers. This should
    // (compiler-dependent but on nearly any modern compiler) be slightly
    // faster than using std::map (although the performance difference will
    // be very small in general). It is however more verbose...

    if (expandedSymbolSet){
        size_t alphabetSize = 27;

        for (size_t i=0; i < sequenceList.size(); i++){
            if ((sequenceList[i].length()) > arrView.shape(1))
                return INVALID_SEQUENCE;

            for (size_t j=0; j < sequenceList[i].length(); j++){
                // We can't inline this function call because it contains a switch,
                // so there is very probably some very small amount of overhead.
                // We're accepting that for now as the price of more readability...
                int positionCode = expandedSymbolSetCharReader(sequenceList[i][j]);
                if (positionCode < 0)
                    return INVALID_SEQUENCE;

                arrView(i, j) = positionCode;
            }
        }
    }

    else if (addGaps){
        size_t alphabetSize = 21;

        for (size_t i=0; i < sequenceList.size(); i++){
            if ((sequenceList[i].length()) > arrView.shape(1))
                return INVALID_SEQUENCE;

            for (size_t j=0; j < sequenceList[i].length(); j++){
                // We can't inline this function call because it contains a switch,
                // so there is very probably some very small amount of overhead.
                // We're accepting that for now as the price of more readability...
                int positionCode = gappedSymbolSetCharReader(sequenceList[i][j]);
                if (positionCode < 0)
                    return INVALID_SEQUENCE;

                arrView(i, j) = positionCode;
            }
        }
    }

    else{
        size_t alphabetSize = 20;

        for (size_t i=0; i < sequenceList.size(); i++){
            if ((sequenceList[i].length()) > arrView.shape(1))
                return INVALID_SEQUENCE;

            for (size_t j=0; j < sequenceList[i].length(); j++){
                // We can't inline this function call because it contains a switch,
                // so there is very probably some very small amount of overhead.
                // We're accepting that for now as the price of more readability...
                int positionCode = standardSymbolSetCharReader(sequenceList[i][j]);
                if (positionCode < 0)
                    return INVALID_SEQUENCE;

                arrView(i, j) = positionCode;
            }
        }
    }
    return VALID_SEQUENCE;
}





// Returns a position code for the expanded symbol set. A value
// < 0 indicates an invalid character.
int expandedSymbolSetCharReader(char &letter){
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
        case 'B':
            positionCode = 21;
            break;
        case 'J':
            positionCode = 22;
            break;
        case 'O':
            positionCode = 23;
            break;
        case 'U':
            positionCode = 24;
            break;
        case 'X':
            positionCode = 25;
            break;
        case 'Z':
            positionCode = 26;
            break;
        default:
            return -1;
            break;
    }
    return positionCode;
}



// Returns a position code for the gapped symbol set. A value
// < 0 indicates an invalid character.
int gappedSymbolSetCharReader(char &letter){
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


// Returns a position code for the standard symbol set. A value
// < 0 indicates an invalid character.
int standardSymbolSetCharReader(char &letter){
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
        default:
            return -1;
            break;
    }
    return positionCode;
}
