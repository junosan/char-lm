
# Character-level language models using recurrent neural networks

This project explores the training and evaluation of various
recurrent neural network (RNN) architectures for the character-level
language modeling task.
Please refer to this [report](readme/report.pdf) for a detailed description.


## Requirements
- Python 2.7 or 3.x (tested on 2.7)
- [Theano](http://deeplearning.net/software/theano/install.html) 0.9+
  - For training on GPU:
[CUDA](https://developer.nvidia.com/cuda-toolkit),
[cuDNN](https://developer.nvidia.com/cudnn),
[pygpu](http://deeplearning.net/software/theano/install.html)


## Quickstart

### Requirements

If using Conda or Miniconda, run `conda install theano pygpu`.
Otherwise, follow the instructions in the above links.

### Dataset and trained models

The dataset and trained RNN models can be found in the
[Releases page](https://github.com/junosan/char-lm/releases) of this
repository.
These must be put in the `data` and `models` directories for below examples.

### Text generation

Using CPU:

```bash
python gen_text.py 'some initial text to initialize the states of RNNs'
```

Using GPU:

```bash
THEANO_FLAGS="device=cuda0" python gen_text.py 'some initial text to initialize the states of RNNs'
```

Model to be used can be set in `gen_text.cfg`.

### Training

Create a shell script with the following content and run it
(requires an Nvidia GPU and CUDA/cuDNN/pygpu installation):

```bash
#!/bin/bash

DEV="device=cuda0"
NAME="test"
MODEL_DIR=models
DATA_DIR=data

FLAGS=$DEV",floatX=float32,gpuarray.preallocate=1,base_compiledir=theano"
THEANO_FLAGS=$FLAGS python -u train.py --data_dir=$DATA_DIR --save_to=$MODEL_DIR/workspace_$NAME | tee -a $MODEL_DIR/$NAME".log"
```

Various training parameters can be configured in `train.py`.
Depending on the amount of VRAM available, `options['net_width']` or
`options['batch_size']` may need to be lowered.

### Plots

With the trained models and logs put in the `models` directory,
run `convert_logs.sh`, then run `.py` files in the `plots` directory
to reproduce the plots in the report.

## Note
This repository was forked from commit `3ab4359` of
[Sophia](https://github.com/junosan/Sophia) (written by me).
