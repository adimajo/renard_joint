# DataHavesting

The repository provides a **pipeline** and an implementation of **SpERT**[[1]](#1) for joint entity and relation extraction. The pipeline consisted of a simple entity recognition and a multiple relation extraction[[2]](#2) models.

## Installation

- Clone this repository
- The data can be found in the directory `N:\Projets02\GRO_STAGES\GRO_STG_2021_01 - Joint Entity and Relation Extraction\Data`
- The pretrained models can be found in the directory `N:\Projets02\GRO_STAGES\GRO_STG_2021_01 - Joint Entity and Relation Extraction\model`
- Set the environment variables `DATA` and `MODEL` to their respective location.
- Install the dependencies using: `pipenv install`
- Install the package using: `pip install .`

## Directory
- `renard_joint` contains the main package;
- `notebooks` contains the notebooks to explore the datasets and fine-tune the model;
- `scripts` contains the command-line tools of the package;
- `tests` contains the `pytest` tests for `renard_joint` and `scripts`;
- `docs` contains the utilities to build the Sphinx documentation;
- `.gitlab-ci.yml` defines the CI/CD pipeline;
- `Pipefile(.lock)` are used to manage the dependencies of the project.

## Usage

### Pipeline
From the command line, once the package is installed:

- For CoNLL04:
    - To evaluate: `python conll04_pipeline.py evaluate`
    - To predict: `python conll04_pipeline.py predict "sentence 1" "sentence 2" ...`

- For the internal dataset:
    - To evaluate: `python internal_pipeline.py evaluate`
    - To predict: `python internal_pipeline.py predict "sentence 1" "sentence 2" ...`

### SpERT
From the command line, once the package is installed:

- To retrain model: `python executer.py [dataset] train`
- To evaluate model: `python executer.py [dataset] evaluate [checkpoint]`
- To predict: `python executer.py [dataset] predict [checkpoint] "sentence 1" "sentence 2" ...`

where `dataset` is either `conll04`, `scierc`, or `internal` and `checkpoint` is the model checkpoint number used for evaluation (for pretrained models, choose 19). Example:

```
python executer.py internal train

python executer.py conll04 evaluate 19

python executer.py conLL04 predict 19 "However, the Rev. Jesse Jackson, a native of South Carolina, joined critics of FEMA's effort." "International Paper spokeswoman Ann Silvernail said that under French law the company was barred from releasing details pending government approval."
```

Note: The hyperparameters for retraining can be modified in the `[dataset]_constants.py` files.

## Reference
<a id="1">[1]</a> Eberts, M., & Ulges, A. (2019). Span-based joint entity and relation extraction with transformer pre-training. arXiv preprint arXiv:1909.07755.

<a id="2">[2]</a> Wang, H., Tan, M., Yu, M., Chang, S., Wang, D., Xu, K., ... & Potdar, S. (2019). Extracting multiple-relations in one-pass with pre-trained transformers. arXiv preprint arXiv:1902.01030.

## TODO

Rename scripts

Coverage

API

Package
