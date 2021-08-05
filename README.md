# Renard Joint

Renard is an NLP software suite developed internally at Crédit Agricole.

This open-source project, dubbed `renard_joint`, is a component of this suite which deals with joint entity and relation
extraction. The repository provides a **pipeline** and an implementation of **SpERT**[[1]](#1) for joint entity and 
relation extraction. The pipeline consists of a simple entity recognition and a multiple relation extraction[[2]](#2) 
model. The main contribution, described in a paper accepted for publication (proceedings to be published later) at 
[SoGood2021](https://sites.google.com/view/ecmlpkddsogood2021/home), is that we provide a model trained on Environmental,
Social and Governance reports, as well as Corporate Social Responsability (CSR) reports annotated by analysts at Crédit
Agricole, such that these can be analyzed automatically.

Current test coverage on internal Gitlab platform: 89 %.

## Installation

### As a Python package

- Clone this repository: `git clone https://github.com/adimajo/renard_joint.git` or `git clone git@github.com:adimajo/renard_joint.git`;
- Set the environment variables `DATA` and `MODEL` to the location of your choice (`data` and `model` by default resp., see below);
- Have a working python development environment, including the `pip` package manager;
- Install `pipenv` with `pip install pipenv`;
- Install the python dependencies of this package using: `pipenv install`;
- Install the package using: `pip install .` (append `pipenv run` if the virtual environment created by `pipenv` hasn't been activated, e.g. in a script).

### As a Flask API


## Documentation

The Sphinx documentation is available [as a Github Page](https://adimajo.github.io/renard_joint).

It can be built by running:
```
$ cd docs
$ make html
```

The API's documentation is available as a Swagger, at the `/spec` endpoint.

Thus, once deployed, *e.g.* on localhost, the `/spec` endpoint will return:
![Swagger input](images/spec_endpoint.png)

Then, a Swagger reader is necessary to turn this json file into a webpage documentation.
Copy-pasting it on [https://editor.swagger.io/](https://editor.swagger.io/), we get:
![Swagger output](images/swagger_output.png)

## Directory

- `renard_joint/` contains the main package;
- `notebooks/` contains the notebooks to explore the datasets and fine-tune the model;
- `tests/` contains the `pytest` tests for `renard_joint` and `scripts`;
- `docs/` contains the utilities to build the Sphinx documentation;
- `.gitlab-ci.yml` defines the CI/CD gitlab pipeline;
- `.github/workflows/python-package.yml` defines the CI/CD github pipeline;
- `Pipefile(.lock)` are used to manage the dependencies of the project.

## Models

Models can be downloaded from ufile.io:

### Spert models

- [The ClimLL model](https://ufile.io/dse7uk5v);
- [The CoNLL04 model](https://ufile.io/0gug8or2);
- [The SciERC model](https://ufile.io/pf9ks53h).

### NER & RE models

#### NER models

- [The ClimLL model](https://ufile.io/bxsmnvtw);
- [The CoNLL04 model](https://ufile.io/j0ff5qrz);

#### RE models

- [The ClimLL model]();
- [The CoNLL04 model]();

### Installation

The models are searched by the package, either in the subfolder `model/`, or in the folder pointed to by the
environment variable `MODEL`. The organisation of this folder must be the following:

- `ner/`: containing the NER model(s);
- `re/`: containing the RE model(s);
- `spert/`: containing the Spert model(s).

## Data

Data, except the ClimLL dataset, can be downloaded from ufile.io:

- [The CoNLL04 dataset](https://ufile.io/vwcg7m9j);
- [The SciERC dataset](https://ufile.io/4828j92x).

### Installation

The data are searched by the package, either in the subfolder `data/`, or in the folder pointed to by the
environment variable `data`. The organisation of this folder must be the following:

- `CoNLL04/`: containing the CoNLL04 data;
- `SciERC/`: containing the SciERC data.

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
 exemplary damages arising out of any use or misinterpretation of the software's content or any portion thereof, 
 regardless of whether the Crédit Agricole has been apprised of the likelihood of such damages.
