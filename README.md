# DataHavesting

The repository provides a **pipeline** and an implementation of **SpERT**[[1]](#1) for joint entity and relation extraction. The pipeline consisted of a simple entity recognition and a multiple relation extraction[[2]](#2) models.

## Installation
- Clone this repository
- The pretrained models can be found in the directory `N:\Projets02\GRO_STAGES\GRO_STG_2021_01 - Joint Entity and Relation Extraction\model`
- Copy the pretrained models folder `model` to the root folder of the cloned repository.

## Directory
- `data` contains the three datasets:
     - `conll04` and `scierc` are public datasets
     - `internal` is the annotated data from RSE reports.
- `renard_joint` contains code and notebooks

## Usage

### Pipeline
From `renard_joint/relation_extraction`,

- For CoNLL04:
    - To evaluate: `python conll04_pipeline.py evaluate`
    - To predict: `python conll04_pipeline.py predict "sentence 1" "sentence 2" ...`
    
- For the internal dataset:
    - To evaluate: `python internal_pipeline.py evaluate`
    - To predict: `python internal_pipeline.py predict "sentence 1" "sentence 2" ...`
    
### SpERT
From `renard_joint/spert`,

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
