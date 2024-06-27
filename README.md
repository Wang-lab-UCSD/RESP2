# covid_experiments

This repo contains code needed to reproduce key experiments for Parkinson / Hard, Ko and Wang 2024 for the ML-assisted discovery of broad-spectrum anti-COVID antibodies.

### Installation

These experiments were originally run on a Nvidia A6000 GPU with cuda12.3.

To run this package, first clone it:
```
git clone https://github.com/Wang-lab-UCSD/covid_experiments/
cd covid_experiments
```

then activate a suitable virtual environment and install the requirements file, e.g.:
```
pip install requirements.txt
```
IMPORTANT!: You also need to install one additional package not included in the requirements,
xGPR, which is not distributed on PyPi. For instructions on installation for xGPR, see the docs at
https://xgpr.readthedocs.io/en/latest/index.html .

Unfortunately some of the experiments in this repo involve two different versions of the xGPR
package. Initially all experiments were run using xGPR v0.2.0.5. Later, however, the traintest split
experiments were re-run using xGPR v0.4.0.1. While the performance difference in terms of held-out
test set score is negligible between these two variants, xGPR v0.4.0.1 is *much* faster and therefore
it was preferable to use it for any experiment where the speed of xGPR is being compared with the
speed of some other approach. xGPR also has an improved API and is easier to install, so it is
preferred in general.

What this means is that you want to reproduce all experiments, you will need to run the first two
in the pipeline using xGPR v0.4.0.1, and all others using v0.2.0.5. There are several ways you can
do this. You can for example set up two virtual environments, or alternative install one variant and
run the corresponding experiments, then install the other variant and run the remaining experiments.
This is definitely cumbersome and we're not thrilled about it either, but to reproduce this experiments in
the way they were run, this is what you'll need to do.

Note also that the requirements file has ```cupy-cuda12x```, which may be inappropriate if you
have cuda 11, and that a specific version of PyTorch is listed which may be inappropriate for
your device. You may need to change these lines in the requirements file if this is indeed the
case. Note that the weights for the final model will be downloaded if you run any of the
last three experiments and the weights have not been downloaded already.

### Usage

To run experiments in the pipeline, from the command line, run:
```
python run_experiments.py
```

and you'll see a list of options:
```
usage: run_experiments.py [-h] [--encodeall] [--traintest] [--evolution] [--evanal]              
                                                                                                                
Use this command line app to run key experiments.

options:
  -h, --help          show this help message and exit
  --encodeall         Encodes the amino acid sequence data.
  --traintest         Run train-test evaluations on the available models.
  --id_key_positions  Find the most important positions to search in silico.
  --evolution         Run the simulated annealing process.
  --evanal            Analyze the simulated annealing results.
```

```--encodeall`` needs to be run first, after that the remaining experiments can be run in any order.
