"""Stores values used by data encoding modules."""

#Only a portion of the RBD was used in the experiments.
RBD_CUTOFFS = (318, 537)

#The starting wt sequence in the experiments.
wt = "EVQLLESGGGLVQPGGTLRLSCAASGFIVSSNYMSWVRQAPGKGLEWVSLVYPGGSTYYADSVKGRFTVSRDNSKNTLYLQMNSLRAEDMAVYYCARDLPSGVDAVDAFDIWGQGTMVTVSS"

#Standard aas in alphabetical order.
aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']


#A list of the Chothia numbered positions used for training
#the heavy chain autoencoder (Parkinson et al. 2023).
chothia_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
            '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
            '22', '23', '24', '25', '26', '27', '28', '29', '30', '31',
            '31A', '31B', '32', '33', '34', '35', '36', '37', '38', '39',
            '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
            '51', '52', '52A', '52B', '52C', '53', '54', '55', '56', '57',
            '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68',
            '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79',
            '80', '81', '82', '82A', '82B', '82C', '83', '84', '85', '86',
            '87', '88', '89', '90', '91', '92', '93', '94', '95', '96',
            '97', '98', '99', '100', '100A', '100B', '100C', '100D', '100E',
            '100F', '100G', '100H', '100I', '100J', '100K', '101', '102',
            '103', '104', '105', '106', '107', '108', '109', '110', '111',
            '112', '113']

#Size of batches to save data.
BATCH_SIZE = 2500

#Minimum total frequency for a sequence to be considered "not noise".
MIN_FREQ = 2


#We trim the left and right ends to make library construction easier on the experimental
#side; these are the trim points. Start from left trim point (inclusive), go up to
#right trim point (exclusive).
LEFT_TRIM_POINT = 9
RIGHT_TRIM_POINT = -9


#Denotes split points between CDRs and between the antigen and antibody
#for cases where all data is concatenated into a single vector. Based
#on the Chothia numbering.
CHOTHIA_SPLIT_POINTS = [16,22,42,46,86,91,104]

#Antigens in high bins.
HIGH_ANTIGENS = ['lambda', 'wt', 'delta', 'alpha']

#Antigens in super bins.
SUPER_ANTIGENS = ['beta', 'omicron', 'ba5', 'ba2', 'gamma', 'kappa']


"""Physicochemical properties (from Grantham et al.)"""
COMPOSITION = {'A': -0.887315380986987, 'R': 0.041436628097621, 'N': 1.0130541145246, 'D': 1.08449657676187, 'C': 3.04202004206328, 'Q': 0.384360446836553, 'E': 0.42722592417892, 'G': 0.17003306012472, 'H': -0.0585828190345676, 'I': -0.887315380986987, 'L': -0.887315380986987, 'K': -0.415795130220955, 'M': -0.887315380986987, 'F': -0.887315380986987, 'P': -0.330064175536222, 'S': 1.14165054655169, 'T': 0.127167582782354, 'W': -0.701564979170065, 'Y': -0.601545532037877, 'V': -0.887315380986987, '-':0.0}
VOLUME = {'A': -1.23448897385975, 'R': 0.930668490901342, 'N': -0.652457397311072, 'D': -0.699019923434967, 'C': -0.67573866037302, 'Q': 0.0226992314853985, 'E': -0.0238632946384961, 'G': -1.88636433959428, 'H': 0.278793125166818, 'I': 0.628012071096028, 'L': 0.628012071096028, 'K': 0.814262175591606, 'M': 0.488324492724344, 'F': 1.11691859539692, 'P': -1.19956707926683, 'S': -1.21120771079781, 'T': -0.536051082001336, 'W': 2.00160659175092, 'Y': 1.21004364764471, 'V': -0.0005820315765488, '-':0.0}
POLARITY = {'A': -0.0836274309924444, 'R': 0.808398499593631, 'N': 1.21724371777891, 'D': 1.73759217728746, 'C': -1.04998885579403, 'Q': 0.808398499593631, 'E': 1.47741794753319, 'G': 0.250882292977334, 'H': 0.771230752485878, 'I': -1.16149209711728, 'L': -1.27299533844054, 'K': 1.10574047645566, 'M': -0.975653361578519, 'F': -1.16149209711728, 'P': -0.120795178100197, 'S': 0.32521778719284, 'T': 0.102211304546321, 'W': -1.08715660290178, 'Y': -0.789814626039753, 'V': -0.901317867363013, '-':0.0}
