"""Stores values used by simulated annealing only."""

#Standard aas in alphabetical order.
aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']



#The most important target positions.
TARGET_POSITIONS = [25, 26, 28, 31, 72, 102]

#Antigens in high bins.
HIGH_ANTIGENS = ['lambda', 'wt', 'delta', 'alpha']

#Antigens in super bins.
SUPER_ANTIGENS = ['beta', 'omicron', 'ba5', 'ba2', 'gamma', 'kappa']

#Maximum variance for high bin antigens. This is set at the 80th percentile
#for variance in the training set.
MAX_HIGH_VAR = 2.2452

#Maximum variance for super bin antigens. This is set at the 80th percentile
#for variance in the training set.
MAX_SUPER_VAR = 0.78597
