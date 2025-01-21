Using VanillaRFFLayer
===============================================
VanillaRFFLayer is a random-features approximated last layer for a neural net. You can
add it to a PyTorch model you are constructing as shown below:::

  import torch
  from uncertaintyAwareDeepLearn import VanillaRFFLayer


  class MyNiftyNeuralNetwork(torch.nn.Module):

      def __init__(self):
          super().__init__()
        
          self.my_last_layer = VanillaRFFLayer(in_features = 212, RFFs = 1024,
                          out_targets = 1, gp_cov_momentum = 0.999,
                          gp_ridge_penalty = 1e-3, amplitude = 1.,
                          likelihood = "gaussian", random_seed = 123)
          # Insert other layers here

      def forward(self, x, update_precision = False, get_var = False):
          # Convert x to latent representation here. Note that update_precision
          # should be set to True when training and False when evaluating.
          if not get_var:
              preds = self.my_last_layer(latent_rep, update_precision)
              return preds

          # Note that if get_var is True, VanillaRFFLayers will also return
          # estimated variance.
          preds, var = self.my_last_layer(latent_rep, update_precision,
                              get_var)
          return preds, var

To understand the parameters accepted by the class constructor, see
below:

.. autoclass:: resp_protein_toolkit.VanillaRFFLayer
   :special-members: __init__
   :members: forward

Notice that there are two ways to generate the precision matrix. It can
either be generated during the course of training, by setting a momentum
value between 0 and 1; or, by setting momentum to a value less than 0
(e.g. -1), it will be generated over the course of a single epoch. If
you are going to use the first strategy, you should pass
``update_precision=True`` to the ``forward`` function of VanillaRFFLayer
on every epoch. Otherwise, you should leave ``update_precision=False``
(the default) during every training epoch right up until the last
epoch, then set ``update_precision=True`` during that last epoch.
The first strategy gives a slightly less accurate estimate of
uncertainty but is easier to implement; the latter is slightly more
accurate and is cheaper during training (except during the last
epoch).

As soon as you call ``model.eval()`` on your model, the model will use
the precision matrix (however generated) to build a covariance matrix.
The covariance matrix is then used to estimate uncertainty any time
you call ``forward`` with ``get_var`` set to True. If you try to
call ``forward`` with ``get_var`` set to True without ever calling
``model.eval()``, a RuntimeError will be generated. If you call
``model.eval()`` but you never set ``update_precision=True`` at any
time during training, a covariance matrix will still be generated,
but the uncertainty estimates it supplies will not be accurate,
so make sure you set ``update_precision=True`` at some point during
training as described above.
