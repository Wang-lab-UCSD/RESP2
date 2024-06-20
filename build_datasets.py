"""Encodes the amino acid sequences in a variety of different formats. These
encodings are saved to disk (faster for training than generating on the fly,
although for some a large amount of diskspace is required)."""
import os
import sys
import argparse
from multi_target_modeling_src.data_encoding.normalized_encode_sequences import generate_basic_encodings


def get_argparser():
    parser = argparse.ArgumentParser(description="Use this command line app "
            "to run / reproduce all of the key steps in the pipeline:")
    parser.add_argument("--encodeall", action="store_true", help=
            "Encodes all of the assembled sequences. Must be run after "
            "processraw, reorg.")
    return parser





if __name__ == "__main__":
    parser = get_argparser()
    start_dir = os.path.abspath(os.path.dirname(__file__))
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.encodeall:
        generate_basic_encodings(start_dir)
