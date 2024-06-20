'''Semi-supervised autoencoder trained both to reconstruct the input
and to make a prediction about it (antibody or not). This is distinct
from the "unadapted" autoencoder trained only to reconstruct its input.'''

import numpy as np
import torch
import torch.nn.functional as F

LOSS_BALANCING_FACTOR = 3

#######################################################################

#This class is used to fit a TaskAdaptedAutoencoder (our shorthand for
#semi-supervised autoencoder that both reconstructs the input and
#predicts whether the input is a "true" antibody or a mutant). This
#model is then used to encode the raw sequence data for the atezolizumab
#dataset; the encoded data is then used as the input to a Bayes-by-backprop
#ordinal regression model. The code required to train is not included here
#since only the already-trained model is used in these experiments.


class TaskAdaptedAutoencoder(torch.nn.Module):


    def __init__(self, random_seed=123):
        super().__init__()
        torch.manual_seed(random_seed)
        #Encoder. Feed input through two convolutional layers and normalize
        #(a very shallow network but turns out to work surprisingly well)
        self.expander = torch.nn.Conv1d(in_channels=21, out_channels=40,
                            kernel_size=21, padding=10)
        self.compressor = torch.nn.Conv1d(in_channels=20, out_channels=6,
                                    kernel_size=11, padding=5)
        self.normalizer = torch.nn.BatchNorm1d(num_features=132,affine=False)

        #The decoder. This is a very small decoder module by most standards --
        #ensures the burden lies on the encoder to generate a meaningful representation
        self.final_adjust = torch.nn.Linear(3,21)

        #The "predictor" generates a binary prediction for antibody vs mutant
        #using simple logistic regression.
        self.predictor = torch.nn.Linear(396,1)


    #Forward pass. Uses "gated" activation (see Dauphin et al, 2016)
    def forward(self, x, decode = True, training=False):
        """Forward pass. Uses 'gated' activation (see Dauphin et al, 2016)."""
        #encode
        x2 = x.transpose(-1,-2)
        x2 = self.expander(x2)
        x2 = x2[:,0:20,:]*torch.sigmoid(x2[:,20:,:])
        embed = self.compressor(x2).transpose(-1,-2)
        embed = embed[:,:,0:3]*torch.sigmoid(embed[:,:,3:])
        embed = self.normalizer(embed)
        if not decode:
            return embed
        #decode
        aas = self.final_adjust(embed)
        aas = torch.softmax(aas, dim=-1)

        pred_cat = torch.sigmoid(self.predictor(embed.reshape(embed.shape[0],
                        embed.shape[1]*embed.shape[2]))).squeeze(-1)

        return aas, pred_cat


    def nll(self, aas_pred, cat_pred, x_mini, y_mini):
        """Custom loss function. Incorporates (1) the cross entropy loss
        for the reconstruction PLUS (2) the binary cross entropy loss for
        the antibody vs mutant prediction. (2) is weighted relative to
        (1) since reconstruction applies across the whole sequence and
        would otherwise therefore predominate.

        Args:
            aas_pred (tensor): The predicted aas for the reconstruction.
            cat_pred (tensor): The binary category predictions.
            x_mini (tensor): The input that is reconstructed.
            y_mini (tensor): The actual categories.

        Returns:
            loss (tensor): The loss.
        """
        lossterm1 = -torch.log(aas_pred)*x_mini
        loss = torch.mean(torch.sum(torch.sum(lossterm1, dim=2), dim=1))
        lossterm2 = torch.mean(LOSS_BALANCING_FACTOR *
                        F.binary_cross_entropy(cat_pred, y_mini))
        return lossterm2 + loss



    def extract_hidden_rep(self, x, use_cpu=False):
        """Generate the encoding for the input sequences.

        Args:
            x (tensor): A PyTorch tensor with one-hot encoded input.
            use_cpu (bool): If True, use CPU rather than GPU.

        Returns:
            replist (tensor): The encoded input.
        """
        with torch.no_grad():
            self.eval()
            if use_cpu:
                self.cpu()
            else:
                self.cuda()
                x = x.cuda()
            return self.forward(x, decode=False).cpu()


    def predict(self, x, use_cpu = False):
        """Reconstructs the input and predicts the category to which
        each sequence belongs. This is used to assess performance
        of the autoencoder.

        Args:
            x (tensor): A PyTorch tensor with one-hot encoded input.
            use_cpu (bool): If True, use CPU rather than GPU.

        Returns:
            replist (tensor): The reconstructed input.
            cat_pred (tensor): The binary category predictions for each
                input sequence.
        """
        with torch.no_grad():
            self.eval()
            if use_cpu:
                self.cpu()
            else:
                x = x.cuda()
                self.cuda()
            aas, cat_pred = self.forward(x)
            return aas.cpu().numpy(), cat_pred.cpu().numpy()

    def reconstruct_accuracy(self, x, use_cpu = False):
        """A helper function to evaluate the reconstruction accuracy of the
        autoencoder for supplied input."""
        reps, _ = self.predict(x, use_cpu)
        if use_cpu:
            x.cpu()
        else:
            x = x.cuda()
            self.cuda()
        pred_aas = np.argmax(reps, axis=-1)
        gt_aas = np.argmax(x.numpy(), axis=-1)
        mismatches, num_preds = 0, 0
        for i in range(x.shape[0]):
            mismatches += np.argwhere(gt_aas[i,:] != pred_aas[i,:]).shape[0]
            num_preds += gt_aas.shape[1]
        return 1 - mismatches / num_preds

    def cat_accuracy(self, x, y, use_cpu = False):
        """A helper function to evaluate antibody vs mutant predictive accuracy."""
        _, cat_preds = self.predict(x, use_cpu)
        return 1 - np.sum(np.abs(y.numpy() - np.rint(cat_preds))) / y.shape[0]
