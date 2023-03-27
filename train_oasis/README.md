# Oasis Artifacts

This directory contains:
- model weights (\*.pth)
- evaluation config (eval\_config.json)
- dataset split used for training and validation (data.json) (You'll likely have to change the paths for your system)
- metrics (metrics.json)

You can verify the results by running: 
```bash
python eval_config.json train_oasis/data.json train_oasis/eval_config.json
```

NB: the avg dice is 0.804 vs 0.807 in the paper, which I think is ok (within error). 
