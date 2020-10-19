# Learning to Orient Surfaces by Self-supervised Spherical CNNs
Repository containing the code of "Learning to Orient Surfaces by Self-supervised Spherical CNNs", accepted to [NeurIPS 2020](https://neurips.cc/).

[[Paper]]() - [[Video]]() - [[Poster]]()

### Authors
Riccardo Spezialetti - Federico Stella - Marlon Marcon - Luciano Silva - [Samuele Salti](https://vision.deis.unibo.it/ssalti/) - [Luigi Di Stefano](https://www.unibo.it/sitoweb/luigi.distefano/)

<p align="center">
  <img width="600" height="400" src="assets/teaser.png">
</p>

## Installation
Before start, make sure that all dependencies are installed. The simplest way to do so, is to use [anaconda](https://www.anaconda.com/).
You can create an anaconda environment called `compass` using
```
conda env create -f requirements.yml
conda activate compass
```

### Instal external dependencies
Install lie-learn:
```bash
git clone https://github.com/AMLab-Amsterdam/lie_learn
cd lie_learn
python setup.py install
```
Install spherical-cnns:
```bash
git clone https://github.com/jonas-koehler/s2cnn.git
cd s2cnn
python setup.py install
```
## How To

### Training
To train a new network from scratch, run:
```
python train.py CONFIG.yaml
```
For available training options, please take a look at `configs/default.yaml`.

## Test
To test the network, 
```
python test.py 
```

## Citation
If you find this code useful in your research, please cite:
```
@inproceedings{xxxx,
  title={yyyyyyy},
  author={zzzz},
  booktitle={zzzz},
  year={2020}
}
```

## License
Code is licensed under Apache 2.0 License. More information in the `LICENSE` file.

## Acknowledgements
Parts of our code are from other repositories:
