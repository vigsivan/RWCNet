# Optimization Based Registration

## Setup

```
pip install -r requirements.txt
```

To evaluate on L2R data, set up L2R eval repo

```
git clone git@github.com:MDL-UzL/L2R.git
cd L2R/evaluation/surface-distance/
python setup.py install
```

## Data Format

The scripts expect the data to be in a JSON format. Please refer to the example for Oasis in the example directory.
The util script provides a few script for converting L2R data to the required format.
There is also support for training flownet with unpaired images.

## Running

### ConvexAdam

```
# for a full list of options
python convexAdam.py --help

python convexAdam.py ~/optimization-based-registration/oasis.json oasis_ca --split val
# functionality for running without labels
python convexAdam.py ~/optimization-based-registration/oasis.json oasis_ca --split val --no-use-labels 

```

### Flownet

WIP
