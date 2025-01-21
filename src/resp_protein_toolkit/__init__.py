"""Import all common functions for convenience."""
import sys

from .encoders import OneHotProteinEncoder
from .encoders import IntegerProteinEncoder
from .encoders import SubstitutionMatrixEncoder
from .directed_evolution.directed_evolution import InSilicoDirectedEvolution

# PyTorch is an optional dependency, and the remaining modules rely on PyTorch.
# Only try to import them if PyTorch is present, otherwise PyTorch would be a
# required dependency.
try:
    import torch
except:
    pass

if "torch" in sys.modules:
    from .classic_rffs import VanillaRFFLayer
    from .protein_ml_models.bytenet_antibody_antigen import ByteNetPairedSeqs
    from .protein_ml_models.bytenet_antibody_only import ByteNetSingleSeq
