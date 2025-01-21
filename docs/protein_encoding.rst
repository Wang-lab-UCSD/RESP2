Protein encoding tools in resp_protein_toolkit
===============================================

To use them as input to an ML model, protein sequences have to
be encoded (e.g. one-hot encoding, etc.) Writing Python code to do
this is easy but frequently redundant. For convenience, this toolkit
contains tools for encoding proteins using some very common
schemes, using Python-wrapped C++ code to ensure speed. There aren't any
embeddings supported yet since there are too many protein LLMs available
for it to be practical to maintain a shared API in one package, but we
may add this at some point in the future.

Sequences are encoded as numpy arrays which are easily converted to Jax /
PyTorch (e.g. in PyTorch, use `torch.from_numpy(my_array)`. Currently
supported schemes include:

- One-hot encoding with either a 2d or 3d array as output, using either the
basic 20 amino acid alphabet, or the basic alphabet plus gaps, or an extended
alphabet including unusual symbols (B, J, O, U, X, Z).
- Integer encoding, using either the basic 20 amino acid alphabet, or the
basic alphabet plus gaps, or an extended alphabet including unusual symbols
(B, J, O, U, X, Z). Integer encoding is useful for LightGBM (gradient boosted
trees) and some clustering schemes.
- Substitution matrix encoding using a 21 letter alphabet (standard AAs plus
gaps) with various percent homologies and two encoding schemes supported.

For details on available options see below.

.. autoclass:: resp_protein_toolkit.OneHotProteinEncoder
   :special-members: __init__
   :members: encode

.. autoclass:: resp_protein_toolkit.SubstitutionMatrixEncoder
   :special-members: __init__
   :members: encode

.. autoclass:: resp_protein_toolkit.IntegerProteinEncoder
   :special-members: __init__
   :members: encode


If you are encoding only a single sequence, make sure to pass it as a list, e.g.::

  encoder1.encode([my_sequence])
