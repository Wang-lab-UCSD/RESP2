# RESP2

This repo contains code needed to reproduce key experiments for Parkinson / Hard, Ko and Wang 2024 for the ML-assisted discovery of broad-spectrum anti-COVID antibodies.

### Installation

These experiments were originally run on a Nvidia A6000 GPU with cuda12.3.

To run this package, first clone it:
```
git clone https://github.com/Wang-lab-UCSD/covid_experiments/
cd covid_experiments
```

then create a suitable virtual environment, activate it and install the requirements:
```
pip install -r requirements.txt
```
then install two additional packages:

```
pip install git+https://github.com/jlparkI/uncertaintyAwareDeepLearn@0.0.5
pip install git+https://github.com/jlparkI/xGPR@0.2.0.5
```

Note: the experiments in this repo were run with an older version of xGPR, v0.2.0.5. The older
version is slower and somewhat harder to install than later versions -- the most recent version is
distributed on pypi as a wheel whereas older versions are not -- so we recommend using later versions
(>0.4.0.1) for new projects. The difference in outcome should be negligible; nontheless to reproduce
the experiments here exactly as they were run initially you should use v0.2.0.5 as illustrated above.

These experiments were run with cuda runtime 12.1 locally installed and with cuSOLVER version 11.3.2.

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
  --traintest_llgp    Run train-test evaluations on the SNGP / LLGP model.
  --id_key_positions  Find the most important positions to search in silico.
  --evolution         Run the simulated annealing process.
  --evanal            Analyze the simulated annealing results.
```

`--encodeall` needs to be run first, after that the remaining experiments can be run in any order.
Note that to ensure maximum reproducibility the SNGP / LLGP model is run using torch.use_deterministic_algorithms
set to True. This may cause an error unless certain environment variables are set. When running the traintest_llgp
experiment, then, you should first run the following line:
```
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```
