"""Runs experiments comparing the performance of different
model architectures on the test set using different encoding
schemes, using pre-specified hyperparameters for the DL models
(for which hyperparameter tuning is expensive) and tuning
hyperparameters on the fly for other models."""
import os
import pickle
import time

import yaml
import numpy as np
import torch

from sklearn.metrics import r2_score
from scipy.stats import norm
from xGPR import build_regression_dataset
from xGPR import xGPRegression as xGPReg

from .bytenet_cnn import CNNRegModel
from .model_trainer import model_train
from .variational_Bayes_regression import bayes_regression_nn
from ..utilities.data_loader_utilities import get_traintest_flists
from ..config import gp_config



def estimate_score(preds, gt):
    """Calculates the R^2 score and estimates a
    confidence interval using a nonparametric bootstrap."""
    score = r2_score(gt, preds)
    all_scores = []
    rng = np.random.default_rng(123)

    for _ in range(1000):
        idx = rng.choice(gt.shape[0], gt.shape[0], replace=True)
        all_scores.append(r2_score(gt[idx], preds[idx]))

    all_scores = np.sort(all_scores)
    lcb, ucb = all_scores[25], all_scores[975]
    return score, lcb, ucb





def calc_miscalibration(preds, ground_truth, var):
    """Calculates the miscalibration area of each model."""
    epe = np.linspace(0, 1, 100).tolist()
    residuals = np.abs(preds - ground_truth)
    calibration_err = []
    for epe_val in epe:
        cutoff = norm.ppf((1+epe_val)/2)
        in_interval = (residuals <= (cutoff * np.sqrt(var)))
        fraction_in_interval = float(in_interval.sum()) / \
                float(residuals.shape[0])
        miscalibration = np.abs(fraction_in_interval - epe_val)
        calibration_err.append(miscalibration)
    return np.trapz(y=np.array(calibration_err),
            x = np.array(epe))





def eval_model(model, xfiles, ant_files, yfiles, get_auce = False):
    """Evaluates a deep learning model on a list of input files."""
    ground_truth = np.concatenate([np.load(y) for y in yfiles])

    if not get_auce:
        auce = 1
        preds = []
        model.eval()
        for xfile, afile in zip(xfiles, ant_files):
            xdata = torch.from_numpy(np.load(xfile))
            adata = torch.from_numpy(np.load(afile))
            preds.append(model(xdata.cuda(), adata.cuda()).detach().cpu().numpy())
        preds = np.concatenate(preds)

    else:
        preds, variance = [], []
        for xfile, afile in zip(xfiles, ant_files):
            xdata = torch.from_numpy(np.load(xfile))
            adata = torch.from_numpy(np.load(afile))
            pred, var = model(xdata.cuda(), adata.cuda(), get_var = True)
            preds.append(pred.detach().cpu().numpy())
            variance.append(var.detach().cpu().numpy())

        preds, variance = np.concatenate(preds), np.concatenate(variance)
        auce = calc_miscalibration(preds, ground_truth, variance)

    return preds, ground_truth, auce




def eval_varbayes_model(model, xfiles, yfiles, get_auce = False):
    """Evaluates a variational Bayesian NN with a map_predict
    method on a list of input files."""
    ground_truth = np.concatenate([np.load(y) for y in yfiles])

    if not get_auce:
        auce = 1
        preds = [model.map_predict(torch.from_numpy(np.load(xfile)).cuda() ).cpu().numpy()
                 for xfile in xfiles]
        preds = np.concatenate(preds)

    else:
        preds, variance = [], []
        for xfile in xfiles:
            xdata = torch.from_numpy(np.load(xfile)).cuda()
            pred = model.map_predict(xdata).cpu().numpy()
            _, var = model.predict(xdata, num_samples=10, random_seed=123)
            preds.append(pred)
            variance.append(var.cpu().numpy())

        preds, variance = np.concatenate(preds), np.concatenate(variance)**2
        auce = calc_miscalibration(preds, ground_truth, variance)

    return preds, ground_truth, auce




def eval_xgpr_model(model, xfiles, yfiles, get_auce = False):
    """Evaluates an xGPR model on a list of
    input files."""
    ground_truth = np.concatenate([np.load(f) for f in yfiles])

    if not get_auce:
        auce = 1
        preds = np.concatenate([model.predict(np.load(xfile), get_var=False) for
                            xfile in xfiles])
    else:
        preds, variance = [], []
        for xfile in xfiles:
            pred, var = model.predict(np.load(xfile), get_var = True)
            preds.append(pred)
            variance.append(var)
        preds, variance = np.concatenate(preds), np.concatenate(variance)
        auce = calc_miscalibration(preds, ground_truth, variance)

    return preds, ground_truth, auce




def write_res_to_file(start_dir, model_description, test_preds, test_gt,
                      train_preds, train_gt,
                      data_prefix, data_suffix, model_class,
                      llgp = "N/A", hyperparams = "N/A",
                      time_elapsed = "0", auce=1):
    """Writes the results of an evaluation to file, together with some
    metadata."""
    os.chdir(os.path.join(start_dir, "results_and_resources"))
    if "traintest_log.txt" not in os.listdir():
        with open("traintest_log.txt", "w+", encoding="utf-8") as fhandle:
            fhandle.write("Model_description,Model_class,Data_prefix,Data_suffix,"
                          "LLGP,Train_score,Train_LCB,Train_UCB,Test_score,"
                          "Test_LCB,Test_UCB,"
                          "Hyperparams,Time_elapsed,AUCE\n")

    train_score, train_lcb, train_ucb = estimate_score(train_preds, train_gt)
    test_score, test_lcb, test_ucb = estimate_score(test_preds, test_gt)

    with open("traintest_log.txt", "a+", encoding="utf-8") as fhandle:
        fhandle.write(f"{model_description},{model_class},{data_prefix},"
                      f"{data_suffix},{llgp},{np.round(train_score,3)},{np.round(train_lcb,3)},"
                      f"{np.round(train_ucb,3)},{np.round(test_score,3)},{np.round(test_lcb,3)},{np.round(test_ucb,3)},"
                      f"{hyperparams},{time_elapsed},{np.round(auce,3)}\n")
    os.chdir(start_dir)



def traintest_cnn(start_dir, config_filepath, data_class = "high",
                  model_type = "cnn", prefix = "onehot",
                  suffix = "x", output_fname = None):
    """Runs a train-test evaluation on the bytenet cnn. Does not store validation
    set results during training."""
    train_xfiles, train_yfiles, train_antfiles, test_xfiles, \
            test_yfiles, test_antfiles = get_traintest_flists(start_dir,
                    model_class = data_class, prefix = prefix, suffix = suffix)

    with open(config_filepath, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    time_elapsed = time.time()

    example_x = np.load(train_xfiles[0])
    example_ant = np.load(train_antfiles[0])

    if model_type == "cnn":
        model = CNNRegModel(input_dim = example_x.shape[2],
            hidden_dim = config['model']['hidden_dim'],
            n_layers = config['model']['n_layers'],
            kernel_size = config['model']['kernel_size'],
            dil_factor = config['model']['dil_factor'],
            dropout = config['model']['dropout'],
            num_antibody_tokens = example_x.shape[1],
            num_antigen_tokens = example_ant.shape[1],
            llgp = config['model']['llgp'],
            antigen_dim = example_ant.shape[2],
            use_spectral_norm = config['model']['use_spectral_norm'],
            contextual_regression = config['model']['conreg'])
    else:
        raise RuntimeError("Non-implemented model supplied.")

    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"Num parameters: {n_parameters}")

    optimizer = torch.optim.Adam(model.parameters(), lr = config['training']['learn_rate'],
                                 weight_decay = config['training']['weight_decay'])
    scheduler = None
    if config['training']['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                config['training']['epochs'], eta_min=config['training']['eta_min'],
                                                               verbose=True)

    all_losses = []

    for i in range(int(config['training']['epochs'] / 2)):
        all_losses += model_train(model, train_xfiles, train_antfiles, train_yfiles,
                    epochs = 2, random_seed = i, optimizer = optimizer,
                    scheduler = scheduler)
        print(f"{i * 2 + 2} epochs", flush=True)

    model.eval()

    get_auce = bool(config['model']['llgp'])

    train_preds, train_gt, _ = eval_model(model, train_xfiles,
                            train_antfiles, train_yfiles)
    test_preds, test_gt, test_auce = eval_model(model,
                                test_xfiles, test_antfiles,
                                test_yfiles, get_auce)
    if output_fname is not None:
        os.chdir(os.path.join(start_dir, "results_and_resources", "trained_models"))
        with open(f"{output_fname}.pk", "wb") as fhandle:
            pickle.dump({"model":model, "predictions":test_preds, "gt":test_gt}, fhandle)
        os.chdir(start_dir)

    time_elapsed = str(time.time() - time_elapsed)

    write_res_to_file(start_dir, "Bytenet_CNN", test_preds, test_gt,
                      train_preds, train_gt, prefix, suffix,
                      data_class, config['model']['llgp'],
                      time_elapsed = time_elapsed,
                      auce = test_auce)




def traintest_xgp(start_dir, model_class, kernel_type = "RBF",
                  prefix = "fastconv_pfa", suffix = "concat",
                  output_fname = None, final_model = False):
    """Runs a train-test evaluation on xGPR RBF or linear kernels.

    Args:
        start_dir (str): Path to the project directory.
        model_class (str): Whether to train on high antigens or super antigens.
        prefix (str): The data file prefix.
        suffix (str): The data file suffix.
        output_fname (str): Either None, in which case the model is not saved,
            or a filename, in which case the model and its predictions are
            saved to this filename.
        final_model (bool): If True, the final model is trained on the full
            combined training and test set, and no predictions are generated /
            no evaluation is performed.
    """
    train_xfiles, train_yfiles, _, test_xfiles, test_yfiles, _ = \
                    get_traintest_flists(start_dir, model_class = model_class,
                            prefix = prefix, suffix = suffix)

    time_elapsed = time.time()
    train_dset = build_regression_dataset(train_xfiles, train_yfiles,
            chunk_size = 2500)

    if "conv" in prefix:
        split_pt = [3000]
    elif "ablang" in prefix:
        split_pt = [104*768]
    elif "autoencoder" in prefix:
        split_pt = [114*3]
    else:
        split_pt = [104*21]

    # Only tune hyperparameters if not building the final model.
    # For the final model, just use the preset hyperparameters from
    # the config file.
    token = f"{prefix}_{model_class}"
    if final_model and token in gp_config.hparams:
        train_xfiles += test_xfiles
        train_yfiles += test_yfiles
        print(f"Hyperparameters for {token} obtained from config file",
                flush=True)
        xgp = xGPReg(num_rffs = 3000, variance_rffs = 1024,
                  kernel_choice = kernel_type, verbose = True, device = "gpu",
                  kernel_settings = {"intercept":True, "split_points":split_pt})
        xgp.set_hyperparams(np.asarray(gp_config.hparams[token]), train_dset)

    else:
        if kernel_type == "Linear":
            xgp = xGPReg(num_rffs = 1024, variance_rffs = 12,
                    kernel_choice = "Linear", verbose = True, device = "gpu")
            if "ablang" not in prefix:
                xgp.tune_hyperparams_crude(train_dset)
            else:
                xgp.set_hyperparams(np.array([1.]), train_dset)
        # ARD kernels have more hyperparameters and are thus harder to tune, requiring
        # multiple restarts of L-BFGS.
        # An easier way is to find the optimal lengthscale for a non-ARD kernel
        # then use this as a starting point for L-BFGS.
        elif kernel_type == "MiniARD":

            xgp = xGPReg(num_rffs = 1024, variance_rffs = 512,
                    kernel_choice = "MiniARD", verbose = True, device = "gpu",
                    kernel_settings = {"split_points":split_pt})
            if "ablang" not in prefix or "conv" in prefix:
                xgp.tune_hyperparams(train_dset, max_iter = 250, tuning_method = "L-BFGS-B",
                         n_restarts = 10, nmll_method = "exact")
                xgp.num_rffs = 16384
                xgp.tune_hyperparams(train_dset, max_iter = 250, tuning_method = "Powell",
                         n_restarts = 1, nmll_method = "approximate")
            else:
                xgp.num_rffs = 8192
                xgp.tune_hyperparams(train_dset, max_iter = 250, tuning_method = "Powell",
                         n_restarts = 1, nmll_method = "exact")

    xgp.num_rffs = 32768
    if kernel_type != "Linear":
        if final_model:
            xgp.variance_rffs = 4096
        else:
            xgp.variance_rffs = 2048
    else:
        xgp.variance_rffs = min(1024, np.load(train_xfiles[0]).shape[1] - 10)

    preconditioner, _ = xgp.build_preconditioner(train_dset, max_rank =
                                min(xgp.num_rffs - 10, 1024), method = "srht")
    xgp.fit(train_dset, mode="cg", preconditioner=preconditioner,
              tol=1e-6)

    if not final_model:
        test_preds, test_gt, test_auce = eval_xgpr_model(xgp, test_xfiles,
                                                                test_yfiles, get_auce=True)
        train_preds, train_gt, _ = eval_xgpr_model(xgp, train_xfiles, train_yfiles)

        hparams = "_".join([str(z) for z in xgp.get_hyperparams().tolist()])
        time_elapsed = str(time.time() - time_elapsed)
        write_res_to_file(start_dir, f"xGPR_{kernel_type}", test_preds, test_gt,
                    train_preds, train_gt, prefix, suffix, model_class,
                    hyperparams = hparams, time_elapsed = time_elapsed,
                    auce = test_auce)

    if output_fname is not None:
        os.chdir(os.path.join(start_dir, "results_and_resources", "trained_models"))
        if final_model:
            with open(f"{output_fname}.pk", "wb") as fhandle:
                pickle.dump({"model":xgp}, fhandle)

        os.chdir(start_dir)





def traintest_varbayes(start_dir, model_class, prefix = "onehot", suffix = "concat"):
    """Runs a train-test evaluation on the varbayes NN."""
    train_xfiles, train_yfiles, _, test_xfiles, test_yfiles, _ = \
                    get_traintest_flists(start_dir, model_class = model_class,
                            prefix = prefix, suffix = suffix)

    config_filepath = os.path.join(start_dir, "multi_target_modeling_src",
                                   "yaml_config_files", "varbayes_config.yaml")
    with open(config_filepath, 'r', encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    model = bayes_regression_nn(xdim=np.load(train_xfiles[0]).shape[1],
                                n1_size=config['model']['n1_size'],
                                n2_size=config['model']['n2_size'])

    model.train_model(train_xfiles, train_yfiles,
                    epochs=config['training']['epochs'],
                    minibatch=config['training']['batch_size'],
                    lr=config['training']['learning_rate'],
                    num_samples=config['training']['num_samples'],
                    random_seed=123, track_loss = False)

    test_preds, test_gt, test_auce = eval_varbayes_model(model, test_xfiles,
                                           test_yfiles, get_auce=True)
    train_preds, train_gt, _ = eval_varbayes_model(model, train_xfiles, train_yfiles)
    write_res_to_file(start_dir, "varbayes_reg", test_preds, test_gt,
                      train_preds, train_gt, prefix, suffix,
                      model_class, auce = test_auce)
