# Renard Joint

Renard is an NLP software suite developed internally at Crédit Agricole.

This open-source project, dubbed `renard_joint`, is a component of this suite which deals with joint entity and relation
extraction. The repository provides a **pipeline** and an implementation of **SpERT**[[1]](#1) for joint entity and 
relation extraction. The pipeline consists of a simple entity recognition and a multiple relation extraction[[2]](#2) 
model. The main contribution, described in upcoming paper, is that we provide a model trained on Environmental,
Social and Governance reports, as well as Corporate Social Responsability (CSR) reports annotated by analysts at Crédit
Agricole, such that these can be analyzed automatically.

For storage reasons on Github, RE models as well as SpERT models for CoNLL04 and SciERC are not hosted here. Hence, lots
of tests are deactivated for Github, but run on our internal Gitlab platform. If you wish to get them as well, please 
email Dr. Adrien Ehrhardt at <adrien.ehrhardt _ at _ credit-agricole-sa.fr>.

Current test coverage on that platform: 89 %.

## Installation

- Clone this repository: `git clone https://github.com/adimajo/renard_joint.git` or `git clone git@github.com:adimajo/renard_joint.git`;
- The data can be found in the `data` subfolder (stored as LFS);
- The pretrained models can be found in the `model` subfolder (stored as LFS);
- If you wish to clone **only** the code, not the datasets nor the models (i.e. to cherry-pick or because they're heavy):
    - Use `GIT_LFS_SKIP_SMUDGE=1 git clone [...]` (Unix) or `set GIT_LFS_SKIP_SMUDGE=1 && git clone [...]` (Windows);
    - Optionnally download the dataset(s) and model(s) you'd like to have through Github's GUI;
    - Set the environment variables `DATA` and `MODEL` to the location of your choice (`data` and `model` by default resp.).
- Install `pipenv` with `pip install pipenv`;
- Install the dependencies using: `pipenv install`;
- Install the package using: `pip install .`

## Directory

- `renard_joint/` contains the main package;
- `notebooks/` contains the notebooks to explore the datasets and fine-tune the model;
- `tests/` contains the `pytest` tests for `renard_joint` and `scripts`;
- `docs/` contains the utilities to build the Sphinx documentation;
- `.gitlab-ci.yml` defines the CI/CD gitlab pipeline;
- `.github/workflows/python-package.yml` defines the CI/CD github pipeline;
- `Pipefile(.lock)` are used to manage the dependencies of the project.
- `data/` contains the public datasets CoNLL04 and SciERC, the internal dataset is kept private as of now (as LFS).
- `model/` contains the Named Entity Recognition, Relation Extraction and SpERT models (as LFS).

## Usage

### Pipeline

From the command line, once the package is installed:

- For CoNLL04:
    - To evaluate: `pipeline [dataset] evaluate`
    - To predict: `pipeline [dataset] predict "sentence 1" "sentence 2" ...`

- For the internal dataset:
    - To evaluate: `pipeline [dataset] evaluate`
    - To predict: `pipeline [dataset] predict "sentence 1" "sentence 2" ...`

where `dataset` is either `conll04`, `scierc`, or `internal`. Example:

```
$ pipeline internal predict "Dirty company does bad coal activity" "Nice company treats people equally"

 Sentence: Dirty company does bad coal activity
 Entities: ( 2 )
 Organisation | company
 CoalActivity | coal
 Relations: ( 0 )
 Sentence: Nice company treats people equally
 Entities: ( 0 )
 Relations: ( 0 )
```

### SpERT

From the command line, once the package is installed:

- To retrain model: `spert [dataset] train`
- To evaluate model: `spert [dataset] evaluate [checkpoint]`
- To predict: `spert [dataset] predict [checkpoint] "sentence 1" "sentence 2" ...`

where `dataset` is either `conll04`, `scierc`, or `internal` and `checkpoint` is the model checkpoint number used 
for evaluation (for pretrained models, choose 19). Example:

```
$ pipeline spert predict 26 "Dirty company does bad coal activity" "Nice company treats people equally"

 Sentence: Dirty company does bad coal activity
 Entities: ( 1 )
 CoalActivity | coal
 Relations: ( 0 )
 Sentence: Nice company treats people equally
 Entities: ( 1 )
 Organisation | company
 Relations: ( 0 )
```

Note: The hyperparameters for retraining can be modified in the `[dataset]_constants.py` files.

## Reference

<a id="1">[1]</a> Eberts, M., & Ulges, A. (2019). Span-based joint entity and relation extraction with transformer pre-training. arXiv preprint arXiv:1909.07755.

<a id="2">[2]</a> Wang, H., Tan, M., Yu, M., Chang, S., Wang, D., Xu, K., ... & Potdar, S. (2019). Extracting multiple-relations in one-pass with pre-trained transformers. arXiv preprint arXiv:1902.01030.

## Disclaimer

The documents presented here reflect the methodologies, calculations, analyses and opinions of theirs authors and 
are transmitted in a strictly informative aim. Under no circumstances will the abovementioned authors nor the 
Crédit Agricole be liable for any lost profit, lost opportunity or any indirect, consequential, incidental or
 exemplary damages arising out of any use or misinterpretation of the website's content or any portion thereof, 
 regardless of whether the Crédit Agricole has been apprised of the likelihood of such damages.
