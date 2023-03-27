# Oasis Reproducibility

This directory contains:
- model weights (\*.pth)
- evaluation config (eval\_config.json)
- dataset split used for training and validation (data.json)
- metrics (metrics.json)

You can verify the results by running: 
```bash
python eval_config.json train_oasis/data.json train_oasis/eval_config.json
```

NB: the avg dice is 0.84 vs 0.87 in the paper, which I think is ok (within error). 
