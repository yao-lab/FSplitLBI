# FSplitLBI
## Fudan-university implementation of Split Linearized Bregman Iteration for Parsimonious Deep Learning

We present the  Fudan Split Linearized Bregman Iteration toolbox (FSplitLBI Toolbox), which offers a strong and versatile functionality for training of Deep Neural Networks (DNNs). 
    The FSplitLBI extends the Split Linearized Bregman Iteration [1] (SLBI) algorithm in linear model [2] to learn the parameters of deep networks. The key novelty of our FSplitLBI lies 
    in a parsimonious learning of the structural sparsity of networks with provable statistical consistency, with the comparable computational cost to  Stochastic Gradient Descendent (SGD) and SGD variants.  In our recent 
    technical report [3], we found that an iterative regularization path with structural sparsity derived from  SLBI, can help prune or grow the  network structures.
    The implementation is based on Optimizer Class of Pytorch; and it can be used with Pytorch code for training DNNs seamlessly.


Environment needed:
1. Python 3.7.1
2. Pytorch 1.0.0
3. Numpy

In the submission process,  we only provide .pyc file; thus  the version of Python should be restricted.

The source codes will be released upon the acceptance; and then we do not need to restrict the python version.

To start with our toolbox, 

Just try:

python3 train_lenet.py
