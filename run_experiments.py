"""A CLI for key experiments."""
import argparse
import time
import os
import sys
import xGPR
from sars_cov2_experiments.data_encoding.normalized_encode_sequences import generate_basic_encodings
from sars_cov2_experiments.data_encoding.simplified_sequence_extraction import calc_enrichment_values
from sars_cov2_experiments.model_code.model_comparison_experiments import traintest_xgp, traintest_cnn, traintest_varbayes
from sars_cov2_experiments.simulated_annealing.run_markov_chains import run_annealing_chains
from sars_cov2_experiments.simulated_annealing.run_markov_chains import analyze_annealing_results
from sars_cov2_experiments.simulated_annealing.run_markov_chains import id_most_important_positions



def get_argparser():
    """Constructs a basic CLI with a menu of available experiments."""
    arg_parser = argparse.ArgumentParser(description="Use this command line app "
            "to run key experiments.")
    arg_parser.add_argument("--encodeall", action="store_true", help=
            "Encodes the amino acid sequence data.")
    arg_parser.add_argument("--traintest", action="store_true", help=
            "Run train-test evaluations on the available models.")
    arg_parser.add_argument("--traintest_llgp", action="store_true", help=
            "Run train-test evaluations on the SNGP / LLGP model.")
    arg_parser.add_argument("--id_key_positions", action="store_true", help=
            "Find the most important positions to search in silico.") 
    arg_parser.add_argument("--evolution", action="store_true", help=
            "Run the simulated annealing process.")
    arg_parser.add_argument("--evanal", action="store_true", help=
            "Analyze the simulated annealing results.")
    arg_parser.add_argument("--simplified_extract", action="store_true", help=
            "Calculates enrichment scores and writes the raw sequences back to "
            "file with this information. Primarily useful for benchmarking "
            "against other methods.")
    return arg_parser




def check_version(expected_version = "0.2.0.5"):
    """Checks that the xGPR version installed is appropriate for
    the requested experiment."""
    if xGPR.__version__ != expected_version:
        raise RuntimeError("The xGPR version installed is not appropriate. "
                "Unfortunately the experiments in this repo are run with an "
                "older version of xGPR. Newer versions are faster and easier to "
                "install (although should give results basically indistinguishable "
                "from the old otherwise). Nonetheless, to ensure complete "
                "reproducibility, please use version 0.2.0.5 for these "
                "experiments.")





if __name__ == "__main__":
    parser = get_argparser()
    home_dir = os.path.abspath(os.path.dirname(__file__))
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.encodeall:
        check_version("0.2.0.5")
        generate_basic_encodings(home_dir)

    if args.traintest:
        check_version("0.2.0.5")
        for (prefix, suffix) in [
                    ("autoencoderconv", "concat"),
                    ("onehotconv", "concat"),
                    ("onehotconvESM", "concat"),
                    ("pfaconv", "concat"),
                    ("autoencoder", "concat"),
                    ("onehotESM", "concat"),
                    ("onehot", "concat"),
                    ("pfa", "concat"),
                    ]:

            if "conv" in prefix:
                kernel_list = ["MiniARD"]
            else:
                kernel_list = ["Linear", "MiniARD"]

            for kernel in kernel_list:
                fname = f"{prefix}_{suffix}_{kernel}_xgp"
                traintest_xgp(home_dir, "high", kernel, prefix = prefix,
                            suffix = suffix, output_fname = fname)

        for (prefix, suffix) in [
                                 ("autoencoder", "concat"),
                                 ("onehot", "concat"),
                                 ("onehotESM", "concat"),
                                 ("pfa", "concat"),
                                 ]:
            traintest_varbayes(home_dir, "high", prefix, "concat")


        for (prefix, suffix) in [("autoencoder", "x"), ("onehot", "x"), ("pfa", "x")]:
            config_fpath = os.path.join(home_dir, "sars_cov2_experiments",
                                        "yaml_config_files", "cnn_config.yaml")
            traintest_cnn(home_dir, config_fpath, "high",
                          "cnn", prefix=prefix, suffix=suffix)

    if args.traintest_llgp:
        for (prefix, suffix) in [("autoencoder", "x"), ("onehot", "x"), ("pfa", "x")]:
            config_fpath = os.path.join(home_dir, "sars_cov2_experiments",
                                        "yaml_config_files",
                                        "cnn_llgp_config.yaml")
            traintest_cnn(home_dir, config_fpath, "high",
                          "cnn", prefix=prefix, suffix=suffix)


    if args.id_key_positions:
        check_version("0.2.0.5")
        id_most_important_positions(home_dir)


    if args.evolution:
        check_version("0.2.0.5")
        run_annealing_chains(home_dir)

    if args.evanal:
        check_version("0.2.0.5")
        analyze_annealing_results(home_dir)

    if args.simplified_extract:
        calc_enrichment_values(home_dir)
