"""Compares / benchmarks alternate models using train-test split."""
import os
import time
import pickle
import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import r2_score
from xGPR import xGPRegression, build_regression_dataset
from resp_protein_toolkit import ByteNetSingleSeq
from ..constants import constants
from ..utilities.utilities import get_data_lists
from .variational_bayes_regression import bayes_regression_nn
from .model_trainer import model_train



def get_reg_statistics_plus_auce(gt, preds, var = None):
    """Calculates R^2, MAE, RMSE for the input predictions
    and ground truth. Does not calculate AUCE."""
    r2 = r2_score(gt, preds)
    mae = np.mean(np.abs(gt - preds))
    rmse = np.sqrt( np.mean( (gt - preds)**2 ) )

    if var is not None:
        auce = calc_miscalibration(preds, gt, var)
    else:
        auce = 1

    return r2, mae, rmse, auce



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

def get_preds_gt(model, file_lists, test_type):
    """Gets predicted and ground-truth values with no variance.
    This is useful for non-uncertainty-aware models."""
    x, y = file_lists[test_type]["x"], file_lists[test_type]["y"]
    predictions = []
    ground_truth = np.concatenate([np.load(f) for f in y])

    for xfile in x:
        pred = model.predict(np.load(xfile))
        predictions.append(pred)

    predictions = np.concatenate(predictions)
    return ground_truth, predictions



def get_preds_gt_var(model, file_lists, test_type):
    """Gets predicted and ground-truth values in addition to
    calculated variance. This is useful for evaluation on the
    test set."""
    x, y = file_lists[test_type]["x"], file_lists[test_type]["y"]
    variance, predictions = [], []
    ground_truth = np.concatenate([np.load(f) for f in y])

    for xfile in x:
        pred, var = model.predict(np.load(xfile), get_var=True)
        predictions.append(pred)
        variance.append(var)

    predictions = np.concatenate(predictions)
    variance = np.concatenate(variance)
    return ground_truth, predictions, variance


def get_xgpr_preds(model, file_lists, test_type):
    """Gets predicted and ground-truth values in addition to
    calculated variance for an xGPR model which requires
    sequence length as input."""
    x, y = file_lists[test_type]["x"], file_lists[test_type]["y"]
    s = file_lists[test_type]["s"]
    variance, predictions = [], []
    ground_truth = np.concatenate([np.load(f) for f in y])

    for xfile, sfile in zip(x, s):
        pred, var = model.predict(np.load(xfile),
                                  np.load(sfile), get_var=True)
        predictions.append(pred)
        variance.append(var)

    predictions = np.concatenate(predictions)
    variance = np.concatenate(variance)
    return ground_truth, predictions, variance





def vbnn_traintest(project_dir, data_group_name = "neuraminidase"):
    """Does a train-test split with a variational Bayes NN,
    and saves the resulting model to file."""
    file_lists = get_data_lists(project_dir, data_group_name,
            "x_vbnn.npy")
    xlist = file_lists["train"]["x"] + file_lists["val"]["x"]
    ylist = file_lists["train"]["y"] + file_lists["val"]["y"]

    time_taken = time.time()

    vbnn_model = bayes_regression_nn(xdim=43*21, n1_size=360, n2_size=60, sigma1_prior=1.)
    _, _, _ = vbnn_model.train_model(xlist, ylist, lr=0.01, num_samples=10,
                        minibatch=1000, epochs=1)
    _, _, _ = vbnn_model.train_model(xlist, ylist, lr=0.002, num_samples=20,
                        minibatch=1000, epochs=40)

    gt, preds, var = get_preds_gt_var(vbnn_model, file_lists, "test")
    r2, mae, rmse, auce = get_reg_statistics_plus_auce(gt, preds, var)
    time_taken = time.time() - time_taken

    write_results_to_file(r2, mae, rmse, auce, data_group_name,
            "test", "vBNN", [0], time_taken,
            project_dir)

    os.chdir(os.path.join(project_dir, "absolut_results"))
    with open(f"{data_group_name}_vbnn_final_model.pk", "wb") as fhandle:
        pickle.dump(vbnn_model, fhandle)



def cnn_traintest(project_dir, data_group_name):
    """Does a train-test split with a CNN,
    and saves the resulting model to file."""
    file_lists = get_data_lists(project_dir, data_group_name,
            "x_cnn.npy")
    xlist = file_lists["train"]["x"] + file_lists["val"]["x"]
    ylist = file_lists["train"]["y"] + file_lists["val"]["y"]

    time_taken = time.time()

    param_dict = constants.CNN_MODEL_PARAMS

    cnn_model = ByteNetSingleSeq(input_dim=param_dict["input_dim"],
                        hidden_dim=param_dict["hidden_dim"],
                        n_layers=param_dict["n_layers"],
                        kernel_size=param_dict["kernel_size"],
                        dil_factor=1,
                        rep_dim=param_dict["rep_dim"],
                        pool_type=param_dict["pool_type"],
                        dropout=0.0,
                        slim=False, llgp=False,
                        objective="regression")
    optimizer = torch.optim.Adam(cnn_model.parameters(),
                        lr = param_dict["lr"],
                        weight_decay = param_dict["weight_decay"])
    for _ in range(int(param_dict["n_epochs"] / 2)):
        _ = model_train(cnn_model, xlist, ylist, epochs = 2,
                minibatch = 200, random_seed = 123, optimizer = optimizer)
        cnn_model = cnn_model.eval()
        # can add check for r2 score during training here if desired...

    gt, preds = get_preds_gt(cnn_model, file_lists, "test")
    r2, mae, rmse, auce = get_reg_statistics_plus_auce(gt, preds, None)
    time_taken = time.time() - time_taken

    write_results_to_file(r2, mae, rmse, auce, data_group_name,
            "test", "CNN", [0], time_taken,
            project_dir)

    os.chdir(os.path.join(project_dir, "absolut_results"))
    with open(f"{data_group_name}_CNN_final_model.pk", "wb") as fhandle:
        pickle.dump(cnn_model, fhandle)



def xgpr_nmll(project_dir, data_group_name = "neuraminidase"):
    """Calculates the negative marginal log likelihood on the
    training data for xGPR only."""
    file_lists = get_data_lists(project_dir, data_group_name)
    xlist = file_lists["train"]["x"] + file_lists["val"]["x"]
    ylist = file_lists["train"]["y"] + file_lists["val"]["y"]
    slist = file_lists["train"]["s"] + file_lists["val"]["s"]

    regdata = build_regression_dataset(xlist, ylist, slist,
                                       chunk_size=5000)
    model = xGPRegression(num_rffs=3000, variance_rffs=1024,
                kernel_choice="Conv1dTwoLayer",
                kernel_settings={"intercept":True,
                    "conv_width":11, "averaging":"sqrt",
                    "init_rffs":2048},
                device="cuda")

    _ = model.tune_hyperparams_crude(regdata, max_bayes_iter=50)
    model.num_rffs = 5000
    _ = model.tune_hyperparams(regdata, max_iter=50, tuning_method="Powell",
                          nmll_method="exact")
    print(np.round(model.get_hyperparams(), 4))



def xgpr_traintest(project_dir, data_group_name):
    """Does a train-test split with xGPR,
    and saves the resulting model to file."""
    file_lists = get_data_lists(project_dir, data_group_name)
    xlist = file_lists["train"]["x"] + file_lists["val"]["x"]
    ylist = file_lists["train"]["y"] + file_lists["val"]["y"]
    slist = file_lists["train"]["s"] + file_lists["val"]["s"]

    regdata = build_regression_dataset(xlist, ylist, slist,
                                       chunk_size=5000)
    time_taken = time.time()
    model = xGPRegression(num_rffs=32768, variance_rffs=1024,
                kernel_choice="Conv1dTwoLayer",
                kernel_settings={"intercept":True,
                    "conv_width":11, "averaging":"sqrt",
                    "init_rffs":2048},
                device="cuda")

    model.set_hyperparams(np.array(constants.XGPR_MODEL_PARAMS[data_group_name]),
                          regdata)
    model.fit(regdata)

    gt, preds, var = get_xgpr_preds(model, file_lists, "test")
    r2, mae, rmse, auce = get_reg_statistics_plus_auce(gt, preds, var)
    time_taken = time.time() - time_taken

    write_results_to_file(r2, mae, rmse, auce, data_group_name,
            "test", "xGPR", model.get_hyperparams().tolist(), time_taken,
            project_dir)
    os.chdir(os.path.join(project_dir, "absolut_results"))
    with open(f"{data_group_name}_xgpr_final_model.pk", "wb") as fhandle:
        pickle.dump(model, fhandle)




def write_results_to_file(r2, mae, rmse, auce, data_group_name,
        test_set_type, model_type, hyperparams,
        time_taken, project_dir):
    """Writes the regression statistics from a model evaluation
    to a log file."""
    os.chdir(project_dir)
    os.makedirs("absolut_results", exist_ok = True)
    os.chdir("absolut_results")
    if "traintest_log.rtxt" not in os.listdir():
        with open("traintest_log.rtxt", "w+", encoding="utf-8") as fhandle:
            fhandle.write("Model_type,Hyperparameters,Data_group,"
                    "Test_set_type,R^2,MAE,RMSE,AUCE,Time_taken\n")

    with open("traintest_log.rtxt", "a", encoding="utf-8") as fhandle:
        fhandle.write(f"{model_type},{'_'.join([str(np.round(h,4)) for h in hyperparams])},"
                f"{data_group_name},{test_set_type},{np.round(r2,3)},{np.round(mae,3)},"
                f"{np.round(rmse,3)},{np.round(auce,3)},{np.round(time_taken,2)}\n")

    os.chdir(project_dir)
