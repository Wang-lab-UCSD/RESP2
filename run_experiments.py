"""A CLI for key experiments."""
import argparse
import os
import sys
from multi_target_modeling_src.data_encoding.normalized_encode_sequences import generate_basic_encodings
from multi_target_modeling_src.model_code.model_comparison_experiments import traintest_xgp, traintest_cnn, traintest_varbayes
from multi_target_modeling_src.simulated_annealing.run_markov_chains import run_annealing_chains
from multi_target_modeling_src.simulated_annealing.run_markov_chains import analyze_annealing_results



def get_argparser():
    """Constructs a basic CLI with a menu of available experiments."""
    arg_parser = argparse.ArgumentParser(description="Use this command line app "
            "to run key experiments.")
    arg_parser.add_argument("--encodeall", action="store_true", help=
            "Encodes the amino acid sequence data.")
    arg_parser.add_argument("--traintest", action="store_true", help=
            "Run train-test evaluations on the available models.")
    arg_parser.add_argument("--evolution", action="store_true", help=
            "Run the simulated annealing process.")
    arg_parser.add_argument("--evanal", action="store_true", help=
            "Analyze the simulated annealing results.")
    return arg_parser





if __name__ == "__main__":
    parser = get_argparser()
    home_dir = os.path.abspath(os.path.dirname(__file__))
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.encodeall:
        generate_basic_encodings(home_dir)

    if args.traintest:
        for (prefix, suffix) in [
                    #("autoencoderconv", "concat"),
                    #("onehotconv", "concat"),
                    #("onehotconvESM", "concat"),
                    #("pfaconv", "concat"),
                    #("autoencoder", "concat"),
                    #("onehotESM", "concat"),
                    #("onehot", "concat"),
                    #("pfa", "concat"),
                    ]:

            if "conv" in prefix:
                kernel_list = ["MiniARD"]
            else:
                kernel_list = ["MiniARD", "Linear"]

            for kernel in kernel_list:
                fname = f"{prefix}_{suffix}_{kernel}_xgp"
                traintest_xgp(home_dir, "high", kernel, prefix = prefix,
                            suffix = suffix, output_fname = fname)

        for (prefix, suffix) in [("autoencoder", "concat"),
                                 ("onehot", "concat"),
                                 ("onehotESM", "concat"),
                                 ("pfa", "concat"),
                                 ]:
            traintest_varbayes(home_dir, "high", prefix, "concat")


        for (prefix, suffix) in [("autoencoder", "x"), ("onehot", "x"), ("pfa", "x")]:
            config_fpath = os.path.join(home_dir, "multi_target_modeling_src",
                                        "yaml_config_files", "cnn_config.yaml")
            traintest_cnn(home_dir, config_fpath, "high",
                          "cnn", prefix=prefix, suffix=suffix)
            config_fpath = os.path.join(home_dir, "multi_target_modeling_src",
                                        "yaml_config_files",
                                        "cnn_llgp_config.yaml")
            traintest_cnn(home_dir, config_fpath, "high",
                          "cnn", prefix=prefix, suffix=suffix)

    if args.evolution:
        run_annealing_chains(home_dir)

    if args.evanal:
        analyze_annealing_results(home_dir)
