# Neural Style Transfer in 2D and 3D

This repository is for Mengdi's master thesis with a topic in neural style transfer in 3D. The goal of this thesis project is to transfer style from 2D image onto 3D mesh, both reshaping the mesh and texturing the mesh, and to explore feature disentangling. 


## Requirements:

All notebooks in this repository can be directly run on Google Colab without additional effort. For acceleration it is helpful to apply a GPU runtime. In case of running these notebooks locally, Python3 and following Python packages are required (common packages such as numpy and PIL are not included):

- [PyTorch](https://pytorch.org/)
- [PyTorch3D](https://pytorch3d.org/)
- [KeOps 1.4.1](https://www.kernel-operations.io/keops/index.html)
- [torchinterp1d](https://github.com/aliutkus/torchinterp1d)


## Usage:

Just open the notebook on Google Colab with GPU runtime, and run the notebook step by step. All needed codes and libraries are installed automatically.


## Files:

There are 7 notebooks included in this repository. Their functions are as follow:

- 2D NST: 2D neural style transfer, i.e. transfering style from style image onto content image
- 2D NST compare statistics: allowing FFT filtering on different feature levels, comparing statistics and style reconstructions of different configs
- 2D NST per-layer operation on BNST: allowing affine transformation and 1D FFT filtering on batch normalization statistics in each layer
- 3D NST: 3D neural style transfer, i.e. transfering style from 2D style image onto 3D mesh, resuling in reshaping or texturing or both on 3D mesh
- 3D NST simultaneous reshaping and texturing: simultaneously reshaping and texturing
- 3D NST sequential reshaping and texturing: sequentially reshaping and texturing, with intermedia result from reshaping taken as color init value for texturing
- 3D NST per-layer operation on BNST: allowing affine transformation and 1D FFT filtering on batch normalization statistics in each leayer
