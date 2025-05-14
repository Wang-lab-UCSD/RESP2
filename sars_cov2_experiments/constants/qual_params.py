"""This file contains the magic numbers we need for sequence
processing, which in this case involve the length of the sequence,
the offset from left and right etc."""

phred_threshold = 8
left_nonoverlapping = 81
right_offset = -2
left_offset = 223
left_readlen = 303

wt_dna_seq = ("TCGGCTAGCGAGGTCCAACTTCTAGAATCTGGTGGGGGGCTTGTCCAGCCTGGTGGCACTTT"
            "GAGGCTGTCTTGTGCTGCCTCAGGGTTCATCGTAAGTAGTAATTACATGAGTTGGGTGCGTCAG"
            "GCACCCGGAAAAGGGCTTGAATGGGTTAGTTTAGTTTACCCAGGTGGTAGCACTTACTATGCCG"
            "ACTCTGTCAAAGGTAGGTTCACCGTTAGTCGTGATAATTCTAAAAACACTTTATACCTGCAAAT"
            "GAATTCACTACGTGCTGAAGACATGGCTGTTTATTATTGCGCACGTGACCTTCCATCCGGCGTT"
            "GATGCCGTTGACGCGTTCGATATATGGGGCCAAGGTACTATGGTGACGGTCAGTTCAGGGATTCT")


#Determines how much to "trim" from the end of each sequence on left and right.
LEFT_CUTOFF = 3
RIGHT_CUTOFF = -2
