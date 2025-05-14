What is in silico directed evolution
======================================

The computational portion of the RESP pipeline may be divided into
three steps:

#. Step 1. Prepare the experimental data for analysis (quality filter
   raw reads, translate to amino acid sequences, number or align if
   necessary).
#. Step 2. Fit an uncertainty-aware model and ensure that performance
   is sufficient (e.g. using cross-validation).
#. Step 3. Run *in silico* directed evolution, using the uncertainty-aware
   model to identify candidate sequences predicted to be tight binders or
   exhibit high activity.
#. Step 4. Filter the accepted sequences from the *in silico* search for
   humanness, developability and other desired properties.

Step 1 tends to be fairly problem-specific -- it may depend, for example,
on how the experiment was set up and the sequencing technology -- and so
tools for this step are not provided here.

There are a broad range of options available for Step 2. Of these, we've found
CNNs with a last-layer GP (an SNGP model), gradient boosted trees and approximate
Gaussian processes
`to work well <https://www.biorxiv.org/content/10.1101/2024.07.30.605700v1>`_,
although we've used variational Bayesian NNs
`before as well. <https://www.nature.com/articles/s41467-023-36028-8)>`_ To learn
how to fit an approximate GP to sequence data, see the
`xGPR library documentation <https://xgpr.readthedocs.io/en/latest/>`_. We'll
eventually provide support for training a variety of models here.

Step 4 requires models with high accuracy for predicting immunogenicity / humanness
and other key developability properties (e.g. stability). For humanness, we've
released a model in `the AntPack library
<https://antpack.readthedocs.io/en/latest/index.html>`_ that
`achieves excellent performance <https://academic.oup.com/bioinformatics/article/40/5/btae278/7656770>`_.

This leaves step 3, where we search sequence space for new candidates with predicted high
binding. It's not necessary to use generative models for this step -- indeed, we've found
this merely substitutes a black-box alternative for methods like genetic
algorithms or simulated annealing. Our preferred alternative here is *in silico* directed
evolution: a modified variant on simulated annealing in which we introduce and score mutations
into a starting wild-type sequence. For details, `see this paper <https://www.nature.com/articles/s41467-023-36028-8>`_.

The `resp_protein_toolkit` provides support for conducting an in silico directed evolution
search using a trained uncertainty-aware model. Uncertainty is key here: if the model is
not uncertainty-aware, we can veer into poorly-mapped regions of sequence space that are
not represented in the training data. By restricting our search to high-confidence candidates,
we can minimize the number of experimental evaluations needed for success.

**How to run in silico directed evolution**
First, create a Python class that exposes a function called `predict`. `predict` must take as an
argument a sequence (as a string), and must return:

* Two 1d numpy arrays of equal size, the first of which is predicted scores against your antigens
  of interest and the second of which is the uncertainty on each score. If there is only one
  antigen each array should be shape[0]=1.

You will likely need to set up this class so it wraps / calls a suitable function for encoding
the protein sequence supplied to `predict` and then feeding the encoded sequence into a
trained uncertainty-aware ML model.

Next, create an instance of the `InSilicoDirectedEvolution` class, as below::

  from resp_protein_toolkit import InSilicoDirectedEvolution
  isde_tool = InSilicoDirectedEvolution(model_object, uncertainty_threshold,
                prob_distro = None, seed = 123)

The `model_object` should be the object you've created that has a `predict` method.
For more details on `uncertainty_threshold`, `prob_distro` and `seed` see
the `Details` section below.

Once this object has been created, you can run *in silico* directed evolution by
calling `run_chain` as shown below::

  isde_tool.run_chain(wild_type_sequence)

  accepted_seqs = isde_tool.get_accepted_seqs()
  accepted_scores = isde_tool.get_scores()
  accepted_uncertainty = isde_tool.get_uncertainty()

To see how much progress the in silico directed evolution process made, it can
be useful to plot the scores. If there is only one target, this is easily done
via::

  import matplotlib.pyplot as plt

  plt.plot(accepted_scores)

There are a variety of options you can set when calling `run_chain` to change
how the process is conducted; for more details on these see `Details` below.
Note that early on in the process, mutations that are slightly deleterious or
neutral have some probability of being accepted, whereas later on only
mutations that improve the score have a non-negligible probability of being
accepted. This procedure ensures the algorithm explores sequence space widely
initially before focusing on the best mutations found so far later on.

For this reason, however, the earlier portion of the `accepted_seqs` are often
associated with low scores, so after plotting the score trajectory during
*in silico* directed evolution it's typical to discard all of the accepted
sequences that score below some threshold. For example, if the best score
achieved by the chain is 4 and the wild-type scores 0, it might be reasonable
to discard anything below 3 -- depending on the dataset, your uncertainty-
aware model, and the number of sequences you can afford to test experimentally.
The threshold ultimately depends on what a "good" score is, which is dependent
on your model.

Finally, it is sometimes desirable to see how many mutations in an accepted
sequence can be converted *back* to wild-type with minimal loss in score. To
do this, you can use the `polish` method of the isde_tool. For example::

  polished_sequence = isde_tool.polish(accepted_mutant_sequence, wild_type_sequence,
                thresh=0.99)

This call will create a polished_sequence by reverting as many positions to wild-type
as possible *as long as* the resulting sequence has a score > thresh * the score of the
accepted mutant sequence. With a thresh of 0.99, for example, the polished sequence
is guaranteed to have a score > 0.99 * the score of the accepted mutant sequence. If
there are multiple targets, the score for each target is guaranteed to be > thresh *
the score of the accepted mutant sequence for that target. `polish` is slow but can
sometimes be very useful if you are trying to minimize the number of mutations
in an accepted sequence.

Once you've found an appropriate set of accepted sequences, you can filter them
based on predicted humanness and other properties. It's usual to run more than
one *in silico* directed evolution chain; since an `InSilicoDirectedEvolution`
tool only has one random seed and stores the sequences associated with its
completed run, you'll need to create a second object with a new random seed to
do this.

**Details**

Here are the details on how to use the `InSilicoDirectedEvolution` class.

.. autoclass:: resp_protein_toolkit.InSilicoDirectedEvolution
   :special-members: __init__
   :members: run_chain, polish, get_scores, get_accepted_seqs, get_uncertainty, get_acceptance_rate
