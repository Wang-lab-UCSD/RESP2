'''Functionality for training deep learning models.'''
import torch
import numpy as np
from sklearn.metrics import r2_score


def model_train(model, xfiles, yfiles, epochs=2,
                    minibatch = 500, track_loss = True,
                    random_seed = 123, optimizer = None,
                    scheduler = None, update_covariance = True):
    """This function trains the model passed to it by caller..
    If track_loss, all loss values are returned.

    Args:
        model: A valid nn.Module with a forward method.
        xfiles (list): A list of x data files (all should be .npy files).
        yfiles (list): The data files containing target values.
        epochs (int): The number of training epochs.
        minibatch (int): The minibatch size.
        track_loss (bool): If True, track loss values during training and
            return them as a list.
        random_seed: If not None, should be an integer seed for the random
            number generator.
        optimizer: An optimizer object that will be used for sgd.
        scheduler: A learning rate scheduler. If None, no learning rate
            scheduling is performed.
        update_covariance (bool): If True, the covariance matrix is updated
            (LLGP models only).

    Returns:
        losses (list): A list of loss values.
    """
    model = model.train()
    model = model.cuda()

    loss_fn = torch.nn.MSELoss()
    losses = []

    for epoch in range(0, epochs):
        if random_seed is not None:
            torch.manual_seed(epoch%50)
        permutation = torch.randperm(len(xfiles)).tolist()

        for j in permutation:
            x, y = np.load(xfiles[j]), np.load(yfiles[j])
            x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()

            for k in range(0, x.shape[0], minibatch):
                x_mini = x[k:(k+minibatch),...].cuda()
                y_mini = y[k:(k+minibatch)].cuda()

                if update_covariance:
                    y_pred = model(x_mini, update_precision=True)
                else:
                    y_pred = model(x_mini)
                loss = loss_fn(y_pred, y_mini)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if track_loss:
                losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()
            print(scheduler.get_last_lr())
        print("Epoch complete")
    if track_loss:
        return losses
