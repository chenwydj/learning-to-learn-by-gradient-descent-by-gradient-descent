# Learning to learn by gradient descent by gradient descent [[PDF](https://arxiv.org/pdf/1606.04474v2.pdf)]

This is a Pytorch version of the LSTM-based meta optimizer.

- [x] For Quadratic functions
- [x] For Mnist

## Prerequisites
- Ubuntu
- Python 3
- NVIDIA GPU

This repository has been tested on GTX 1080Ti.

## Installation
* Clone this repo:
```bash
git clone https://github.com/chenwydj/learning-to-learn-by-gradient-descent-by-gradient-descent.git
cd learning-to-learn-by-gradient-descent-by-gradient-descent
```
* Install dependencies:
```bash
pip install requirements.txt
```

## Usage
* To reproduce the paper: simply go through the notebook `Grad_buffer.ipynb`
* To implement your own Learning-to-Optimize works: please feel free to use `meta_module.py`

## Method
The core part to reproduce the LSTM meta optimzer is to **update the `nn.Parameters` of the optimizee in place while retaining the `grad_fn`**. In Pytorch, `nn.Parameters` are designed to be leaf nodes. The only way to modify the value of an patameter is something like `p.data.add_` (take the last line in `sgd.py` in Pytorch for an example). However, modifying `.data` of a tensor does not produce a `grad_fn`, which is vital for our meta optimizer to be upadted from. More discussions can be found in [here](https://discuss.pytorch.org/t/nn-parameter-doesnt-retain-grad-fn/29214) and [here](https://discuss.pytorch.org/t/gradient-with-respect-to-parameters-that-update-model-parameters/39141).

**One way to bypass this problem is to leverage the `Buffer` in Pytorch**. `Buffer` is also a "parameter" in our model and can be saved in `state_dict`, but will not be returned by `model.parameters()`. Once typical example of `Buffer` is the `running_mean` and `running_var` in `BatchNorm` layers. The `Buffer` tensors can be treated as weights, while also have the flexibility to retain `grad_fn` when being updated in-place. We thus add parameters via `nn.Module.register_buffer()`.

This comes the reason why the `meta_module.py` is provided. The core class is `MetaModule`, which inherits `nn.Module` but we manually return the `Buffers` as parameters. Further on, we build `MetaLinear`, `MetaConv2d`, `MetaConvTranspose2d`, `MetaBatchNorm2d`, and `MetaSequential` on top of `MetaModule` with registered buffers.

## Acknowledgement
* Original L2O code from [AdrienLE/learning_by_grad_by_grad_repro](https://github.com/AdrienLE/learning_by_grad_by_grad_repro).
* Meta modules from [danieltan07/learning-to-reweight-examples](https://github.com/danieltan07/learning-to-reweight-examples).