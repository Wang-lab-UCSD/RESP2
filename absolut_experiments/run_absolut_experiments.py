"""A CLI for key experiments."""
import argparse
import zipfile
import os
import shutil
import sys
import wget

from src.constants import constants
from src.data_processing.encode_input_data import data_encoding
from src.modeling.model_comparison import xgpr_traintest, vbnn_traintest, cnn_traintest, xgpr_nmll
from src.modeling.run_resp import run_resp_search
from src.data_processing.gen_ai_preprocessing import extract_high_scoring_seqs
from src.modeling.process_absolut_scores import check_absolut_scores


def get_argparser():
    """Constructs a basic CLI with a menu of available experiments."""
    arg_parser = argparse.ArgumentParser(description="Use this command line app "
            "to run key experiments.")
    arg_parser.add_argument("--retrieve_data", action="store_true",
            help="Retrieve the raw data from the Absolut database and "
                "move it to a suitable location.")
    arg_parser.add_argument("--encode_data", action="store_true",
            help="Encode the retrieved absolut data.")
    arg_parser.add_argument("--nmll", action="store_true",
            help="Calculates the negative marginal log-likelihood "
            "on the training data for xGPR.")
    arg_parser.add_argument("--run_test_split", action="store_true",
            help="Test the model on the held-out test set and save final "
            "models to the results folder.")
    arg_parser.add_argument("--run_resp_search", action="store_true",
            help="Generate candidate sequences using the RESP search "
            "with the final trained models. The resulting candidate sequences "
            "will be saved to the 'absolut_results' folder under files ending "
            "in .rtxt; these files can be used as input to Absolut! for scoring.")
    arg_parser.add_argument("--evaluate_resp_candidates", action="store_true",
            help="Once the candidate sequences have been scored using Absolut!, "
            "move the Absolut! output to the 'absolut_results/absolut_scores' "
            "folder, then use this argument to evaluate these scores and "
            "calculate / print success rates for each model and target.")
    return arg_parser


def retrieve_raw_data(project_dir):
    """Retrieves the raw data from the Absolut database."""
    os.chdir(project_dir)

    if "absolut_data" not in os.listdir():
        os.mkdir("absolut_data")

    template_link = "https://ns9999k.webs.sigma2.no/10.11582_2021.00063/projects/NS9603K/pprobert/AbsolutOnline/RawBindingsPerClassMurine"

    os.chdir("absolut_data")
    for target in [f"{template_link}/1NCA_NAnalyses/1NCA_N_500kNonMascotte.txt.zip",
            f"{template_link}/1NCA_NAnalyses/1NCA_N_Mascotte.txt.zip",
            f"{template_link}/4QNP_AAnalyses/4QNP_A_500kNonMascotte.txt.zip",
            f"{template_link}/4QNP_AAnalyses/4QNP_A_Mascotte.txt.zip",
            f"{template_link}/4YUE_CAnalyses/4YUE_C_500kNonMascotte.txt.zip",
            f"{template_link}/4YUE_CAnalyses/4YUE_C_Mascotte.txt.zip",
            f"{template_link}/5LQB_AAnalyses/5LQB_A_500kNonMascotte.txt.zip",
            f"{template_link}/5LQB_AAnalyses/5LQB_A_Mascotte.txt.zip",
            f"{template_link}/1TQB_AAnalyses/1TQB_A_500kNonMascotte.txt.zip",
            f"{template_link}/1TQB_AAnalyses/1TQB_A_Mascotte.txt.zip",
            f"{template_link}/2W9E_AAnalyses/2W9E_A_500kNonMascotte.txt.zip",
            f"{template_link}/2W9E_AAnalyses/2W9E_A_Mascotte.txt.zip",
            f"{template_link}/4H88_AAnalyses/4H88_A_500kNonMascotte.txt.zip",
            f"{template_link}/4H88_AAnalyses/4H88_A_Mascotte.txt.zip",
            f"{template_link}/3L95_XAnalyses/3L95_X_500kNonMascotte.txt.zip",
            f"{template_link}/3L95_XAnalyses/3L95_X_Mascotte.txt.zip",
            f"{template_link}/5CZV_AAnalyses/5CZV_A_500kNonMascotte.txt.zip",
            f"{template_link}/5CZV_AAnalyses/5CZV_A_Mascotte.txt.zip",
            f"{template_link}/3RVV_AAnalyses/3RVV_A_500kNonMascotte.txt.zip",
            f"{template_link}/3RVV_AAnalyses/3RVV_A_Mascotte.txt.zip",
            f"{template_link}/4PP1_AAnalyses/4PP1_A_500kNonMascotte.txt.zip",
            f"{template_link}/4PP1_AAnalyses/4PP1_A_Mascotte.txt.zip",
            f"{template_link}/1JPS_TAnalyses/1JPS_T_500kNonMascotte.txt.zip",
            f"{template_link}/1JPS_TAnalyses/1JPS_T_Mascotte.txt.zip",
            f"{template_link}/4M7L_TAnalyses/4M7L_T_500kNonMascotte.txt.zip",
            f"{template_link}/4M7L_TAnalyses/4M7L_T_Mascotte.txt.zip",
            ]:
        fname = wget.download(target)
        with zipfile.ZipFile(fname, 'r') as zip_ref:
            zip_ref.extractall("temp")

        os.chdir("temp")
        os.chdir(os.listdir()[0])
        shutil.move(os.listdir()[0], os.path.join("..", ".."))
        os.chdir(os.path.join("..", ".."))
        shutil.rmtree("temp")
        os.remove(fname)

    os.chdir(project_dir)




def get_full_target_list():
    """Gets a full list of all protein targets for
    model training."""
    target_list = []
    for _, target_set in constants.TARGET_PROTEIN_GROUPS.items():
        target_list += target_set
    return target_list




if __name__ == "__main__":
    parser = get_argparser()
    home_dir = os.path.abspath(os.path.dirname(__file__))
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)


    if args.retrieve_data:
        retrieve_raw_data(home_dir)
    if args.encode_data:
        data_encoding(home_dir)
        extract_high_scoring_seqs(home_dir)


    if args.nmll:
        for target in get_full_target_list():
            print(f"\n\n**********************\n{target}")
            xgpr_nmll(home_dir, target)


    if args.run_test_split:
        for target in get_full_target_list():
            print(f"\n\n**********************\n{target}")
            xgpr_traintest(home_dir, target)
            cnn_traintest(home_dir, target)
            vbnn_traintest(home_dir, target)

    if args.run_resp_search:
        for target_group in ["neuraminidase", "il2", "prion",
                                 "dust_mite", "notch", "tissue_factor"]:
            for target_model in ["CNN", "xgpr"]:
                run_resp_search(home_dir, target_group,
                                target_model = target_model)


    if args.evaluate_resp_candidates:
        check_absolut_scores(home_dir)
