/* Utilities for sequence processing. */
#include "utilities.h"


//Gets the longest length for any sequence in the set. If allSameLength is passed
//and the sequences are NOT all the same length, 0 is returned.
int get_max_length(std::vector<std::string> sequenceList, bool allSameLength){
    int maxLength = 0;

    if (sequenceList.size() == 0)
        return 0;

    if (allSameLength){
        for (auto & sequence : sequenceList){
            if (sequence.length() != maxLength){
                if (maxLength > 0)
                    return 0;
                else
                    maxLength = sequence.length();
            }
        }
    }
    else{
        for (auto & sequence : sequenceList){
            if (sequence.length() > maxLength)
                maxLength = sequence.length();
        }
    }

    return maxLength;
}
