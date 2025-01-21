"""Generates a PSSM using the SAM mixture model from
AntPack v0.1.0."""
import sys
import numpy as np
from antpack import SequenceScoringTool, SingleChainAnnotator
from scipy.special import logsumexp
from generate_pssm_loadfiles import get_factored_pfasum_pssm, factor_pssm
sys.path.append("..")
from constants import seq_encoding_constants


# We mask the CDRs when calculating distance between
# distributions in the mixture model. These are
# the start-end of the IMGT-defined CDRs. First
# position is included, second excluded.
CDR_MASK_POINTS = [('27', '39'), ('56', '66'),
        ('105','118')]



def generate_antibody_pssm():
    """Builds the 'stacked PSSM' which contains a
    PSSM for every position in the wild-type sequence."""
    wt = seq_encoding_constants.wt
    alignment_tool = SingleChainAnnotator()
    numbering = alignment_tool.analyze_seq(wt)[0]

    # Copy the position probabilities, get marginal
    # probabilities for each amino acid at each position
    # and for each amino acid overall.
    score_tool = SequenceScoringTool()
    model_probs = score_tool.models["human"]["H"].log_mu_mix + \
            score_tool.models["human"]["H"].log_mix_weights[:,None,None]
    marginal_probs = np.exp(logsumexp(model_probs, axis=0))
    marginal_aa_probs = marginal_probs.sum(axis=0)
    marginal_aa_probs /= marginal_aa_probs.sum()

    # Get the CDR excluded positions.
    cdr_excluded = [np.arange(score_tool.position_dict["H"][a],
        score_tool.position_dict["H"][b]) for (a,b) in
        CDR_MASK_POINTS]
    cdr_excluded = set(np.concatenate(cdr_excluded).tolist())

    stacked_pssms = np.zeros((len(wt), 21, 21))
    default_pssm = get_factored_pfasum_pssm(62)[0]
    default_marginal = (1/21)**2

    for k, imgt in enumerate(numbering):
        mapped_pos = score_tool.position_dict["H"][imgt]
        if mapped_pos in cdr_excluded:
            stacked_pssms[k,:,:] = default_pssm
            continue

        pmat = np.zeros((21,21))
        for i in range(21):
            pssm_ratio = (marginal_probs[k,i].clip(min=1/21))**2 / default_marginal
            pmat[i,i] = np.log2(pssm_ratio)

            for j in range(i+1,21):
                max_self_sim = max(max(marginal_probs[k,i]**2,
                        marginal_probs[k,j]**2), default_marginal)
                pssm_ratio = marginal_probs[k,i] * marginal_probs[k,j]
                pssm_ratio = min(max(pssm_ratio, default_marginal), max_self_sim)
                pssm_ratio /= default_marginal
                pmat[i,j] = np.log2(pssm_ratio)
                pmat[j,i] = np.log2(pssm_ratio)

        if np.abs(pmat).max() > 15:
            print("ow")
            import pdb
            pdb.set_trace()

        try:
            pmat_factored = factor_pssm(pmat)
        except:
            import pdb
            pdb.set_trace()
        stacked_pssms[k,:,:] = pmat_factored

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    generate_antibody_pssm()
