'''This class performs modified simulated annealing, where
sequences are generated through random modifications to the wild-type,
with the probability of a given mutation determined by the abundance
of that aa at that position in the training set. The "score" from
regression is then contrasted with the score of the premodification
sequence and the modification is accepted with a probability determined
by the temperature. Sequences can be scored against one antigen or
multiple. The size of the search space can be limited by selecting
specific positions for mutation.'''

import random
import copy
import numpy as np




class InSilicoDirectedEvolution():

    def __init__(self, model_object, uncertainty_threshold,
            prob_distro = None, seed:int = 123,
            approach = "liberal"):
        """Class constructor.

        Args:
            model_object: An object that has a "predict" method. "predict"
                must take as input a protein sequence and must return
                two 1d numpy arrays, the first of which is scores against
                the antigens and the second of which is the uncertainty on
                those scores. For example, if there is one antigen, it should
                return two numpy arrays both with shape[0] = 1.
                The model_object will usually contain a trained model and
                a sequence encoder as attributes.
            uncertainty_threshold: The maximum allowed uncertainty
                on a prediction before a sequence is rejected. This should be an
                array of the same length as the number of antigens if there are
                multiple antigens OR a float if there is only one. This value is
                dataset- and model-dependent. For regression problems, you may
                often be able to set this threshold based on prior knowledge (for
                example, if you are training on Kd values and predicting log Kd,
                you likely know how wide of a confidence interval you can tolerate
                before the prediction is not useful). You can also set this value
                using your test set (e.g. set it to the 95th percentile of the
                uncertainty scores for the test set or some similar heuristic).
            prob_distro: Either None or a 2d numpy array. If None, all possible
                mutations are assumed to have equal probability. If a numpy
                array, its shape[0] must be of the same length as the sequence
                and should give the probability of each possible mutation at
                each position.
            seed (int): The seed to the random number generator.
            approach (str): One of "liberal", "conservative". If 'liberal',
                the probability that a sequence is accepted is determined by
                the best score improvement against any of the targets. If
                'conservative', the acceptance probability is determined by
                the worst score improvement against any target. Ignored if
                there is only one target.
        """
        self.model_object = model_object
        self.prob_distro = prob_distro
        self.seed = seed
        self.uncertainty_threshold = uncertainty_threshold
        self.approach = approach

        # Amino acids in alphabetical order.
        self.aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
                'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']

        # The accepted sequences, uncertainties and scores generated
        # by this chain. Initialized to empty.
        self.accepted_seqs = []
        self.scores = []
        self.uncertainty = []
        self.acceptance_rate = 0


    def get_scores(self):
        """Convenience function for retrieving the scores from
        a run."""
        return self.scores

    def get_accepted_seqs(self):
        """Convenience function for retrieving the accepted
        sequences from a run."""
        return self.accepted_seqs

    def get_uncertainty(self):
        """Convenience function for retrieving the uncertainty
        values from a run."""
        return self.uncertainty


    def get_acceptance_rate(self):
        """Convenience function for retrieving the acceptance
        rate from a completed run."""
        return self.acceptance_rate



    def _calc_transition_prob(self, proposed_seq:list, temperature:float,
            current_scores):
        """Scores a proposed sequence and calculates the transition
        probability according to an update rule.

        Args:
            proposed_seq (list): A list of letters that forms the proposed
                sequence. Will be converted to a string before passing to
                the model object.
            temperature (float): The current temperature; used to determine
                acceptance probability.
            current_scores: A 1d numpy array with the same shape[0] as there
                are targets (e.g. if 1 target, shape[0] = 1).
        """
        new_scores, new_uncertainties = self.model_object.predict("".join(proposed_seq))
        if new_scores.shape[0] == 1 and new_uncertainties.shape[0] == 1:
            if new_uncertainties[0] > self.uncertainty_threshold:
                return -1, new_scores, new_uncertainties
            acceptance_prob = np.exp( min(0, ((new_scores - current_scores) /
                    temperature) ) )
            return acceptance_prob, new_scores, new_uncertainties

        if new_scores.shape[0] > 1:
            if self.uncertainty_threshold.shape[0] != new_uncertainties.shape[0]:
                raise RuntimeError("The number of uncertainty thresholds must "
                        "match the number of uncertainties for each datapoint.")
            for new_uncertainty, threshold in zip(new_uncertainties.tolist(),
                    self.uncertainty_threshold.tolist()):
                if new_uncertainty > threshold:
                    return -1, new_scores, new_uncertainties

            if self.approach == "liberal":
                acceptance_prob = np.exp( ((new_scores - current_scores) /
                    temperature).max().clip(max=0) )
            else:
                acceptance_prob = np.exp( ((new_scores - current_scores) /
                    temperature).min().clip(max=0) )
            return acceptance_prob, new_scores, new_uncertainties

        raise RuntimeError("The model object must have a method called 'predict' "
                    "that when called returns two 1d numpy arrays (scores and "
                    "uncertainties for a set of targets).")



    def run_chain(self, wild_type, max_iterations:int = 3000,
            cooldown:float = 0.995, sparse_mutations:int = -1,
            max_num_failures:int = 100, starting_temp:float = 25.):
        """This function runs a simulated annealing experiment as described above.

        Args:
            wild_type (str): The starting wild-type sequence that
                we would like to modify.
            max_iterations (int): The maximum number of iterations for
                the chain.
            cooldown (float): The rate at which to cool down. Slower means more
                exploration of (possibly useless) sequences.
            sparse_mutations (int): If > 0, try to revert to WT any time
                there are more than this number of mutations. Encourages the
                search to stay as close to WT as possible.
            max_num_failures (int): If this number of sequences are rejected
                in a row, terminate the evolution process.
            starting_temp (float): The starting temperature for the algorithm.
        """
        # The accepted sequences, uncertainties and scores generated
        # by this chain. Initialized to empty.
        self.accepted_seqs = []
        self.scores = []
        self.uncertainty = []
        self.acceptance_rate = 0

        num_accepted = 0
        random.seed(self.seed)
        rng = np.random.default_rng(self.seed)
        np.random.seed(self.seed)

        if self.prob_distro is not None:
            aa_probs = self.prob_distro.sum(axis=1)
            aa_probs /= aa_probs.sum()
            if len(wild_type) != self.prob_distro.shape[0]:
                raise RuntimeError("If prob distro is not None, "
                        "it must be the same length as the wild-type "
                        "sequence.")

        original_seq, current_seq = list(wild_type), list(wild_type)
        current_scores, _ = self.model_object.predict(wild_type)

        i, temp, num_failures = 0, starting_temp, 0

        while temp >= 0.01:
            if self.prob_distro is None:
                chosen_one = random.randint(0, len(wild_type) - 1)
                new_aa = self.aas[random.randint(0, len(self.aas) - 1)]
            else:
                chosen_one = rng.choice(len(wild_type),
                        p = aa_probs, size=1)[0]
                new_aa = self.aas[np.random.choice(21, size=1,
                    p=self.prob_distro[chosen_one,:])[0]]

            num_mutated_positions = len([mp for mp, wp in zip(current_seq, original_seq)
                    if mp != wp])
            if num_mutated_positions >= sparse_mutations > 0:
                new_aa = original_seq[chosen_one]

            #If we accidentally selected the existing aa, try again, without changing
            #temp or any other settings.
            if new_aa == current_seq[chosen_one]:
                continue

            proposed_seq = current_seq.copy()
            proposed_seq[chosen_one] = new_aa

            acceptance_prob, new_scores, new_uncertainties = \
                    self._calc_transition_prob(proposed_seq, temp,
                    current_scores)
            runif = np.random.uniform()

            if acceptance_prob > runif:
                current_seq = copy.copy(proposed_seq)
                current_scores = new_scores.copy()

                num_accepted += 1
                num_failures = 0
            else:
                num_failures += 1

            self.scores.append(current_scores)
            self.uncertainty.append(new_uncertainties)
            self.accepted_seqs.append(''.join(current_seq))

            temp *= cooldown
            i += 1
            if i % 250 == 0:
                print(f"Temperature: {temp}")
            if i > max_iterations or num_failures > max_num_failures:
                print(f"Num consecutive failures: {num_failures}")
                print(f"Num iterations: {i}")
                break

        self.scores = np.stack(self.scores)
        self.uncertainty = np.stack(self.uncertainty)
        self.acceptance_rate = num_accepted / max_iterations


    def polish(self, starting_sequence, wild_type, thresh = 0.95):
        """This function is available as an option to do
        additional processing on sequences
        selected by the simulated annealing process. Ideally we would like
        to use a sequence with a small number of mutations from the original.
        The polish function takes each mutant and tries to return the amino acid
        at as many mutated positions as possible to the aa present in the wild
        type at that position, while keeping the score within a threshold of the
        score for the original selected mutant.

        Args:
            starting_sequence (str): The starting sequence / mutant that we
                would like to polish.
            wild_type (str): The wild type sequence we would like to revert to
                wherever possible.
            thresh (float): The threshold, should be in the range 0 - 1.
                This function will try to maintain the score against each
                target at >= the starting score times thresh.
        """
        revised_mutant = list(starting_sequence)
        score_thresh = thresh * self.model_object.predict(starting_sequence)[0]
        keep_cycle = True

        if self.prob_distro is None:
            positions_for_mutation = np.arange(len(wild_type)).tolist()
        else:
            positions_for_mutation = [i for i in range(self.prob_distro.shape[0])
                    if self.prob_distro[i,:].sum() > 0]

        while keep_cycle:
            idx = []
            score_tracker = []
            for position in positions_for_mutation:
                proposed_mut = revised_mutant.copy()

                if proposed_mut[position] != wild_type[position]:
                    proposed_mut[position] = wild_type[position]
                    proposed_mut = "".join(proposed_mut)
                    score_tracker.append(self.model_object.predict(proposed_mut)[0] -
                            score_thresh)
                    idx.append(position)

            score_tracker = np.stack(score_tracker)
            if len(score_tracker.shape) == 1:
                score_tracker = score_tracker[:,None]
            if np.max(np.min(score_tracker, axis=1)) < 0:
                keep_cycle = False
                break

            best_mut = idx[np.argmax(np.min(score_tracker, axis=1))]
            revised_mutant[best_mut] = wild_type[best_mut]

        revised_mutant = "".join(revised_mutant)
        scores, uncertainties = self.model_object.predict(revised_mutant)
        return revised_mutant, scores, uncertainties
