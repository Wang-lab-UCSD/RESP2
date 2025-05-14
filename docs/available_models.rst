Built-in models for protein sequence data / fitness landscapes
===============================================================

The `resp_protein_toolkit` contains a couple of built-in deep learning
models that are easy to use for modeling protein fitness landscapes.
Currently the available built-in models are based on Microsoft's ByteNet,
but are adapted so that they can be made uncertainty-aware using the
VanillaRFFs layer also available in this package. You do not have to
use these models and can substitute another uncertainty-aware model of
your choosing when using the RESP *in silico* directed evolution
also available in this package if desired.

Here are the details:

.. autoclass:: resp_protein_toolkit.ByteNetSingleSeq
   :special-members: __init__
   :members: forward, predict

.. autoclass:: resp_protein_toolkit.ByteNetPairedSeqs
   :special-members: __init__
   :members: forward, predict


To train these models, it's typical to pass one of them together with
training settings (learning rate, learning rate scheduler, selected
optimizer etc.) to a function that will train the model for some set
number of epochs (say 1 or 2), then calculate some performance metric
on the training and test set.

The details of learning rate, learning rate scheduler,
optimizer etc. may need to be changed depending on your problem; it's
usually a good idea to check performance on a validation set and adjust
as needed. For an example of how to train this kind of model and use it
with RESP to generate new sequences, see the example notebook on the main
page of the docs.

Notice that ordinal regression (``objective='ordinal'``) is special in some
ways. When this objective is selected, the model calculates a single latent
"score" for each input datapoint, and the model output is a vector of shape
``K-1``, where K is the number of possible categories. Each element ``i`` of the output
is the probability that the input datapoint belongs to category ``i+1`` *or* to a
higher category. This arrangement only makes sense if the categories are ranked,
of course, and this is the only situation where ordinal regression is useful.
When training this type of model you should use binary cross entropy loss on the
output and as ground truth labels use an array of shape ``(N,K-1)`` where each
element ``[i,j]`` is either 1 (indicating that datapoint ``i`` belongs to category
``j+1`` or above) or 0. We'll add an example of what this looks like in practice
soon.
