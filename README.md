# The Neural Hawkes Particle Smoothing
Source code for [Imputing Missing Events in Continuous-Time Event Streams (ICML 2019)](http://cs.jhu.edu/~jason/papers/#mei-et-al-2019) runnable on GPU and CPU.

## Reference
If you use this code as part of any published research, please acknowledge the following paper (it encourages researchers who publish their code!):

```
@inproceedings{mei-2019-smoothing,
  author =      {Hongyuan Mei and Guanghui Qin and Jason Eisner},
  title =       {Imputing Missing Events in Continuous-Time Event Streams},
  booktitle =   {Proceedings of the International Conference on Machine Learning},
  year =        {2019}
}
```

## Instructions
Here are the instructions to use the code base.

### Dependencies
This code is written in Python 3, and I recommend you to install:
* [Anaconda](https://www.continuum.io/) that provides all the Python-related dependencies;
* [PyTorch 1.0](https://pytorch.org/) that handles auto-differentiation.

### Prepare Data
Download datasets from this [Google Drive link](https://drive.google.com/drive/folders/1tZ6xODd3tO3qgSp2gp-jqYOdfh3aj827?usp=sharing) to the 'data' folder. See more details in this [README](data/README.md).

### Install Modules
Run the command line below to install the modules (add `-e` option if you need an editable installation):
```
pip install .
```

### Train Models
Go to the nhps/functions directory.

To train the neural Hawkes process with complete data, try the command line below for detailed guide (see section 2 in paper for more technical details):
```
python train_nhpf.py --help
```

To train the neural Hawkes particle smoother with incomplete data, try the command line below for detailed guide (see section 3 in paper for more technical details):
```
python train_nhps.py --help
```

### Test Models
Go to the nhps/functions directory.

To evaluate (dev or test), use the command line below for detailed guide (see section 4 in paper for more technical details):
```
python test.py --help
```

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
