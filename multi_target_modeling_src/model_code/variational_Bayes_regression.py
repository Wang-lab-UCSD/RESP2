'''Bayes-by-backprop NN adapted from Blundell et al 2015, designed
to perform regression using paired input sequences (antigen and antibody).
The model can be run in MAP mode for reproducibility, which generates a MAP
prediction, or using sampling, which provides a (VERY crude, but nonetheless
useful) estimate of uncertainty.
'''
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import r2_score

#######################################################################


class FC_Layer(torch.nn.Module):
    """The FC_Layer is a single fully connected layer in the network.

    Attributes:
        mean_prior1: The mean of the prior distribution on the weights
        sigma_prior1: The width of the prior distribution on the weights
        input_dim: Expected input dimensionality
        output_dim: Expected output dimensionality
        pi2log (float): A useful constant
        pilog (float): A useful constant
        weight_means: The means of the distributions for the weights in the layer
        weight_rhos: The rho parameters used to generate the standard deviation of
            the distribution for each weight.
        bias_means: The means of the distributions for the biases in the layer
        bias_rhos: The rho parameters used to generate the standard deviation of
            the distribution for each bias term.
    """
    def __init__(self, n_in, n_out, sigma_prior1 = 1.0):
        super(FC_Layer, self).__init__()
        torch.manual_seed(123)
        self.register_buffer("sigma_prior1", torch.tensor([sigma_prior1]))
        self.register_buffer("mean_prior1", torch.tensor([0.0]))
        
        self.input_dim = n_in
        self.output_dim = n_out
        
        self.register_buffer("pi2log", torch.log(torch.tensor([2.0*3.1415927410125732])) )
        self.register_buffer("pilog", torch.log(torch.tensor([3.1415927410125732])) )

        self.weight_means = torch.nn.Parameter(torch.zeros((n_in, n_out)).uniform_(-0.1,0.1).float())
        self.weight_rhos = torch.nn.Parameter(torch.zeros((n_in, n_out)).uniform_(-3,-2).float())

        self.bias_means = torch.nn.Parameter(torch.zeros((n_out)).uniform_(-0.1,0.1).float())
        self.bias_rhos = torch.nn.Parameter(torch.zeros((n_out)).uniform_(-3,-2).float())


    def forward(self, x, sample=True, random_seed = None):
        """The forward pass. Notice this layer does not apply any activation.

        If we are sampling, we use the means and rhos to generate a sample
        from the normal distribution they describe for each weight, and this gives us
        our weight matrix -- we use the reparameterization trick so that the gradient can
        be evaluated analytically. Additionally, we evaluate the KL divergence -- the other term
        in the ELBO -- for the variational distribution of the weights from p(w), aka the 
        complexity term.

        If not sampling (MAP mode), the KL divergence cannot be evaluated; rather than drawing
        samples we just use the mean of the weight & bias distributions for the forward pass.
        This is used only for predictions; obviously if we used this for training we would
        be reverting to a simple FCNN.
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
        if sample:
            weight_epsilons = Variable(self.weight_means.data.new(self.weight_means.size()).normal_())
            bias_epsilons = Variable(self.bias_means.data.new(self.bias_means.size()).normal_())
            weight_stds = self.softplus(self.weight_rhos)
            bias_stds = self.softplus(self.bias_rhos)

            weight_sample = self.weight_means + weight_epsilons*weight_stds
            bias_sample = self.bias_means + bias_epsilons*bias_stds

            output = torch.mm(x, weight_sample) + bias_sample
            kl_loss = -self.log_gaussian(self.mean_prior1, self.sigma_prior1, 
                                        weight_sample).sum()
            kl_loss = kl_loss - self.log_gaussian(self.mean_prior1, self.sigma_prior1,
                                            bias_sample).sum()

            kl_loss = kl_loss + self.log_gaussian(self.weight_means, weight_stds,
                                    weight_sample).sum()
            kl_loss = kl_loss + self.log_gaussian(self.bias_means, bias_stds,
                                    bias_sample).sum()
        else:
            kl_loss = 0
            output = torch.mm(x, self.weight_means) + self.bias_means
        return output, kl_loss

    def softplus(self, x):
        """Helper function for generating standard deviations from rho values."""
        return torch.log(1 + torch.exp(x))

    def log_gaussian(self, mean, sigma, x):
        """The log of a gaussian function -- all weights follow
        normal distributions."""
        return -0.5*self.pi2log - torch.log(sigma) - 0.5*(x-mean)**2 / (sigma**2)
    


class bayes_regression_nn(torch.nn.Module):
    """The full model which is trained on an enrichment dataset.
    Note that last layer must output shape[1] == 1 in order
    to perform regression."""
    def __init__(self, sigma1_prior = 1.0, xdim=264,
                 n1_size=60, n2_size=30):
        super().__init__()
        torch.manual_seed(123)

        self.bnorm_x = torch.nn.BatchNorm1d(xdim, affine = False)
        self.n1_x = FC_Layer(xdim, n1_size, sigma1_prior)

        self.bnorm_2 = torch.nn.BatchNorm1d(n1_size, affine = False)
        self.n2 = FC_Layer(n1_size, n2_size, sigma1_prior)
        self.bnorm_3 = torch.nn.BatchNorm1d(n2_size, affine = False)
        self.n3 = FC_Layer(n2_size,1, sigma1_prior)
        self.register_buffer("train_mean", torch.zeros((1,xdim))  )
        self.register_buffer("train_std", torch.zeros((1,xdim))  )

        self.activate_scaling = False


    def set_scaling_factors(self, xfiles):
        """Calculates scaling values. Used on the training set only.
        If you want to use this, you MUST call it before beginning
        any training."""
        self.activate_scaling = True
        x1 = np.load(xfiles[0])
        self.train_mean = torch.zeros((1,x1.shape[1]))
        self.train_std = torch.zeros((1,x1.shape[1]))
        self.scaling_factor_calcs(xfiles, self.train_mean, self.train_std)


    def scaling_factor_calcs(self, file_list, mean_array, std_array):
        """Performs scaling factor calculations for an individual file
        list."""
        ndpoints = 0
        for xfile in file_list:
            x = torch.from_numpy(np.load(xfile))
            batch_mean = torch.mean(x, dim=0)
            batch_std = torch.std(x, dim=0)

            updated_std = mean_array[0,:]**2 + std_array[0,:]**2
            updated_std *= ndpoints
            updated_std += x.shape[0] * (batch_mean**2 + batch_std**2)
            updated_std /= (ndpoints + x.shape[0])
            updated_std -= (ndpoints * mean_array[0,:] + x.shape[0] *
                    batch_std)**2 / (ndpoints + x.shape[0])

            std_array[0,:] = torch.sqrt(updated_std)
            mean_array[0,:] = mean_array[0,:] * ndpoints + \
                    torch.sum(x, dim=0)
            ndpoints += x.shape[0]
            mean_array /= ndpoints


    def forward(self, x, sample=True, random_seed = None):
        """The forward pass. Note that activation is applied here
        (rather than inside the FC layer).

        Args:
            x (tensor): The input data.
            get_score (bool): If True, get the score and do not bother
                with class label predictions.
            sample (bool): If True, generate multiple weight samples.
            random_seed (int): The seed for the random number generator
                to ensure reproducibility.

        Returns:
            output (tensor): The predicted y-values.
        """
        x, kl_loss_x = self.n1_x(x, sample, random_seed)

        x = F.elu(self.bnorm_2(x))

        x, kl_loss2 = self.n2(x, sample, random_seed)
        x = F.elu(self.bnorm_3(x))
        x, kl_loss3 = self.n3(x, sample, random_seed)
        net_kl_loss = kl_loss_x + kl_loss2 + kl_loss3
        return x.flatten(), net_kl_loss


    def negloglik(self, ypred, ytrue, kl_loss):
        """Custom loss function. We calculate the mean-squared error for
        predictions and add in the complexity term.

        Args:
            ypred (tensor): The predicted classes.
            ytrue (tensor): The actual classes.
            kl_loss (tensor): The KL_divergence loss.

        Returns:
            loss: The resulting loss values.
        """
        loss = (ypred - ytrue)**2
        return (torch.sum(loss) + kl_loss) / ypred.shape[0]


    def scale_data(self, x):
        if self.activate_scaling:
            x = (x - self.train_mean) / self.train_std
        return x


    def train_model(self, xfiles, yfiles, valid_xfiles = None, valid_yfiles = None,
                    epochs=40, minibatch=500, track_loss = True,
                    lr=0.002, num_samples = 5, random_seed = 123):
        """This function trains the model represented by the class instance.
        If track_loss, all loss values are returned. Adam optimization with
        a low learning rate is used.

        Args:
            xfiles (list): A list of x data files (all should be .npy files).
            yfiles (list): The data files containing target values.
            valid_xfiles (list): A list of validation set xfiles. If None,
                validation set scores are not checked.
            valid_yfiles (list): A list of validation set yfiles. If None,
                validation set scores are not checked.
            epochs (int): The number of training epochs.
            minibatch (int): The minibatch size.
            track_loss (bool): If True, track loss values, training set scores
                and (if not None for valid_xfiles) validation set scores.
            lr (float): The learning rate for Adam.
            num_samples (int): The number of weight samples to draw on each
                minibatch pass. A larger number will speed up convergence for
                training but make it more expensive. Defaults to 5.
            random_seed: If not None, should be an integer seed for the random
                number generator.

        Returns:
            losses (list): A list of loss values.
        """
        num_batches = len(yfiles)

        self.train()
        self.cuda()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses, train_scores, valid_scores = [], [], []

        for epoch in range(0, epochs):
            seed = None
            if random_seed is not None:
                torch.manual_seed(epoch%50)
            permutation = torch.randperm(len(xfiles)).tolist()

            for j in permutation:
                x, y = np.load(xfiles[j]), np.load(yfiles[j])
                x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()

                for k in range(0, x.shape[0], minibatch):
                    x_mini = x[k:(k+minibatch),...].cuda()
                    y_mini = y[k:(k+minibatch)].cuda()
                    x_mini = self.scale_data(x_mini)

                    loss = 0
                    for i in range(num_samples):
                        if random_seed is not None:
                            seed = random_seed + i + j + epoch
                        y_pred, kl_loss = self.forward(x_mini, random_seed = seed)
                        loss += self.negloglik(y_pred, y_mini, kl_loss/num_batches)
                    loss = loss / num_samples
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if track_loss:
                        losses.append(loss.item())
            print("Epoch complete", flush=True)

            if track_loss:
                preds, gt = [], []
                for xfile, yfile in zip(xfiles, yfiles):
                    xdata = np.load(xfile)
                    xdata = torch.from_numpy(xdata).float().cuda()
                    preds.append(self.map_predict(xdata).cpu().numpy())
                    gt.append(np.load(yfile))

                preds, gt = np.concatenate(preds), np.concatenate(gt)
                train_scores.append(r2_score(gt, preds))

                if valid_xfiles is not None and valid_yfiles is not None:
                    preds, gt = [], []
                    for xfile, yfile in zip(valid_xfiles, valid_yfiles):
                        xdata = np.load(xfile)
                        xdata = torch.from_numpy(xdata).float().cuda()
                        preds.append(self.map_predict(xdata).cpu().numpy())
                        gt.append(np.load(yfile))

                    preds, gt = np.concatenate(preds), np.concatenate(gt)
                    valid_scores.append(r2_score(gt, preds))

                self.train()

        if track_loss:
            return losses, train_scores, valid_scores


    def map_predict(self, x):
        """This function makes a MAP prediction for y-value without performing any sampling. Unlike
        a sampling based prediction, this point value prediction is fully reproducible
        and should therefore be used for generating predictions in applications where
        reproducibility is desired.

        If (approximate) quantitation of uncertainty is desired,
        by contrast, predict should be used instead.
        """
        with torch.no_grad():
            self.eval()
            x = self.scale_data(x)
            return self.forward(x, sample=False)[0].flatten()


    def predict(self, x, num_samples=5, random_seed=None):
        """This function returns the predicted y-value for each
        datapoint using sampling (not MAP) and the standard deviation of the
        predicted y-value.

        Args:
            x (tensor): The input data.
            num_samples (int): The number of weight samples to draw. Larger numbers
                improve accuracy and decrease speed.
            random_seed (int): A seed for reproducibility.

        Returns:
            predy (tensor): predicted y-values
            std_scores (tensor): The standard deviation of the score _for each datapoint._
        """
        with torch.no_grad():
            self.eval()
            scores = []
            x = self.scale_data(x)
            seed = random_seed
            for i in range(num_samples):
                if seed is not None:
                    seed = random_seed + i
                scores.append(self.forward(x, random_seed = seed)[0])
            scores = torch.stack(scores, dim=1)
            mean_scores, std_scores = torch.mean(scores, dim=-1), torch.std(scores, dim=-1)
            return mean_scores, std_scores
