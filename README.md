# Optimization Based Registration

# Setup

The following packages are needed.

```bash
tqdm                                                                                                 
numpy==1.20.0                                                                                        
scipy==1.8.0                                                                                         
torchio==0.18.71

typer==0.4.0                                                                                         
einops==0.4.0                                                                   

torch==1.10.1
```

# Usage

```bash
# Train without labels, --warp-images will save the moved images to the output directory
python convexAdam.py train-without-labels <data.json> ./output2 --warp-images

# Train with labels, dice score is saved as output/dice_measurements.json
python convexAdam.py train-with-labels <data.json> ./output
```

For an example of the json format, refer to example jsons provided.
