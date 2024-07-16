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
pip install -r requirements.txt
```
IMPORTANT!: You also need to install one additional package not included in the requirements,
xGPR, which is not distributed on PyPi. For instructions on installation for xGPR, see the docs at
https://xgpr.readthedocs.io/en/latest/index.html .

Unfortunately the experiments in this repo were run with an older version of xGPR, v0.2.0.5. The older
version is slower and somewhat harder to install than later versions, so we recommend using later versions
(>0.4.0.1) for new projects. The difference in outcome should be negligible; nontheless to reproduce
the experiments here exactly as they were run initially you should use v0.2.0.5.

Note also that the requirements file has ```cupy-cuda12x```, which may be inappropriate if you
have cuda 11, and that a specific version of PyTorch is listed which may be inappropriate for
your device. You may need to change these lines in the requirements file if this is indeed the
case.

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
