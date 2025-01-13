"""Extracts the highest-scored sequences in the training
set to use for fine-tuning a generative AI model."""
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from ..utilities.utilities import read_raw_data_files
from ..constants import constants


def extract_high_scoring_seqs(project_dir):
    """Extracts the highest-scoring sequences and writes them to txt
    files so that we can use them for fine-tuning generative models
    (e.g. LLMs)."""
    target_groups = []
    for _, target_list in constants.TARGET_PROTEIN_GROUPS.items():
        target_groups += target_list

    os.chdir(project_dir)
    if "absolut_encoded_data" not in os.listdir() or \
            "absolut_data" not in os.listdir():
        raise RuntimeError("Data must be downloaded and encoded first.")

    for target_name, target_list in constants.TARGET_PROTEIN_GROUPS.items():
        percentile_groups = {80:[], 90:[]}

        for target in target_list:
            seqs, scores = read_raw_data_files([ os.path.join(project_dir, "absolut_data",
                                                         f"{target}_500kNonMascotte.txt") ])

            # Very important to "flip" the scores, since a lower (more negative) value
            # indicates tighter binding, but we will maximize scores in RESP.
            scores = -np.array(scores)

            for percentile in [80,90]:
                threshold = np.percentile(scores, percentile)
                idx = np.where(scores > threshold)[0].tolist()
                print(f"For target group {target}, the {percentile}th percentile "
                    f"is {threshold}, with {len(idx)} above it.")
                percentile_groups[percentile] += [seqs[i] for i in idx]

        random.seed(123)

        for percentile in [80, 90]:
            trainx, testx = train_test_split(percentile_groups[percentile],
                                             test_size=0.2, shuffle=True,
                                             random_state=123)

            with open(os.path.join(project_dir, "absolut_encoded_data",
                                   f"{target_name}_{percentile}_genAI_train.txt"), "w+",
                                   encoding = "utf-8") as fhandle:
                fhandle.write("Sequence\n")
                for seq in trainx:
                    fhandle.write(f"{seq}\n")

            with open(os.path.join(project_dir, "absolut_encoded_data",
                                   f"{target_name}_{percentile}_genAI_test.txt"), "w+",
                                   encoding = "utf-8") as fhandle:
                fhandle.write("Sequence\n")
                for seq in testx:
                    fhandle.write(f"{seq}\n")
