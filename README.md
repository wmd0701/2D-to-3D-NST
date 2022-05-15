# Neural Style Transfer in 2D and from 2D to 3D

This repository is for Mengdi's master thesis with a topic in 2D-to-3D neural style transfer. Title of the thesis is "Exploration of Deep Features for Neural Style Transfer". There are two goals of this thesis project: 1. transfer style from 2D image onto 3D mesh, resulting in stylization of both mesh shape (per-vertex displacement) and mesh texture (per-vertex color); 2. style feature decomposition (disentanglement).

This repository contains Python notebooks that are used for experiments. All notebooks should be able to be run directly on Google Colab without additional effort. It is recommended to run notebooks on Colab with a GPU runtime. To seamlessly open a notebook from GitHub on Colab, just open the notebook in GitHub and add "tocolab" after "github" in url. Example: "https://github.com/xxx.ipynb" to "https://githubtocolab.com/xxx.ipynb".

## Teaser
<img src="https://user-images.githubusercontent.com/34072813/168476387-cf9f2a10-623f-4b23-9bd3-d2ae3dc74476.jpg" width=70% height=70%>
<img src="https://user-images.githubusercontent.com/34072813/168476395-f8ff43ad-5b9e-4bfb-b4e7-8416152ac6e0.png" width=70% height=70%>
<img src="https://user-images.githubusercontent.com/34072813/168476398-18fbf20d-3ec9-4083-9f65-108ecdc3c401.png" width=70% height=70%>

## Requirements:

All required libraries are either already provided on Colab or automatically installed during runtime. Specifically, following machine learning-related frameworks (common packages such as PIL and matplotlib are not included) are already given on Colab (version by thesis submission): 

- Python 3.7.13
- PyTorch 1.10.0+cu111
- Numpy 1.21.6

And following frameworks are installed during notebook run:

- [PyTorch3D 0.6.1](https://pytorch3d.org/)
- [KeOps 1.4.1](https://www.kernel-operations.io/keops/index.html)
- [torchinterp1d](https://github.com/aliutkus/torchinterp1d)

It should be noticed that Google keeps updating Python and libraries on Colab. The installation of PyTorch3D framework is highly sentitive to Python version and PyTorch version. When running a 2D-to-3D NST notebook, please check whether PyTorch3D with version >= 0.6.1 is successfully installed with command `!pip show pytorch3d`. In case the installation fails, one may need to manually install PyTorch3D from [GitHub repo](https://github.com/facebookresearch/pytorch3d) or re-install a different version of PyTorch.


## Files:

There are totally 9 notebooks included in this repository with their name indicating their functions:

- 2D NST: 2D neural style transfer, i.e. transfering style from style image onto content image
- 2D NST statistics comparison: allowing FFT filtering on different feature levels
- 2D NST BN statistics style feature decomposition: allowing affine transformation and 1D FFT filtering on BN statistics
- 2D NST weighting BN std components: amplifying BN std components of specific frequency by weighting
- 2D-to-3D NST: 2D-to-3D neural style transfer, i.e. transfering style from 2D style image onto 3D mesh
- 2D-to-3D NST simultaneous reshaping and texturing: simultaneously reshaping and texturing
- 2D-to-3D NST sequential reshaping and texturing: sequentially reshaping and texturing with a novel color initialization strategy
- 2D-to-3D NST BN statistics style feature decomposition: transplanting style feature decomposition from 2D to 2D-to-3D
- 2D-to-3D NST amazing: some amazing results
