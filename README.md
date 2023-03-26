# RWCNet

This is the official repository for RWCNet

## Setup

```
pip install -r requirements.txt
```

## Training
* Currently, the model caches the hidden tensors as well as the flows at each level. This ends up taking **a lot** of space.
* The model training can be configured using the train_config.json. The options for the configuration can be inferred from `config.py`
* When running the full model against L2R data, please use the data format used for the L2R 2022 challenge.

To train the model on L2R data:

```
python l2r_train_eval.py l2r_dataset_json.json train_config.json
```

## Publication

If you find this repository useful, please cite:

    Recurrence With Correlation Network for Medical Image Registration
    Vignesh Sivan, Teodora Vujovic, Raj Ranabhat, Alexander Wong, Stewart Mclachlin, Michael Hardisty
    eprint arXiv:2302.02283
