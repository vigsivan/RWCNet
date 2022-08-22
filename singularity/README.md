# Running with Singularity

## Run

Download the singularity image file:

```
wget https://utoronto-my.sharepoint.com/:u:/g/personal/m_hardisty_utoronto_ca/EbEFUPAVxn1KnSwk8wzvfi8Bb1fvGKE4nxDmufSJYQ2hmg?download=1 -O Some-RNN3.sif
```



```
singularity shell --nv --cleanenv Some-RNN3.sif
```

This will spawn a singularity instance and will present the singularity shell type the follow commands to start training

```
bash
conda init bash
bash
conda activate Some-RNN
cd /optimization-based-registration
python util.py convert-nlst-json /path/to/dataset/json /path/to/dataset /input/json/path
python run_rnn.py train /input/json/path /output/directory/path
```

Here is a complete example, note that here the singularity write results to a host directory.  This is the most convienient method for insepcting results files:
```
bash
conda init bash
bash
conda activate Some-RNN
cd /optimization-based-registration
python util.py convert-nlst-json ~/datasets/NLST-resampled/NLST_dataset.json  ~/datasets/NLST-resampled   ~/src/optimization-based-registration/singularityTraining/NLST-resampled-input.json
python run_rnn.py train ~/src/optimization-based-registration/singularityTraining/NLST-resampled-input.json ~/src/optimization-based-registration/singularityTraining
```
