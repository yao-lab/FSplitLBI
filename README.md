# FSplitLBI
## Fudan-university implementation of Split Linearized Bregman Iteration for Parsimonious Deep Learning

We present the  Fudan Split Linearized Bregman Iteration toolbox (FSplitLBI Toolbox), which offers a strong and versatile functionality for training of Deep Neural Networks (DNNs). 
    The FSplitLBI extends the Split Linearized Bregman Iteration [1,2] (SLBI) algorithm in linear model to learn the parameters of deep networks. The key novelty of our FSplitLBI lies 
    in a parsimonious learning of the structural sparsity of networks with provably improved statistical model selection consistency [3], with the comparable computational cost to Stochastic Gradient Descendent (SGD) and SGD variants. SLBI has been successfully applied in computer vision and medical image analysis [4-5]. In our recent 
    technical report [6-7], we found that an iterative regularization path with structural sparsity derived from  SLBI, can help prune or grow the network structures.
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

A tutorial is [ here ](https://github.com/yao-lab/FSplitLBI/blob/master/FuSplit_LBI_tutorial.pdf).

## Reference:

[1] Stanley Osher, Feng Ruan, Jiechao Xiong, Yuan Yao and Wotao Yin. Sparse Recovery via Differential Inclusions. 
Applied and Computational Harmonic Analysis, 41(2):436-469, 2016. [ (paper) ](https://doi.org/10.1016/j.acha.2016.01.002 "ACHA online")

[2] Chendi Huang, Xinwei Sun, Jiechao Xiong, Yuan Yao. Split LBI: An Iterative Regularization Path with Structural Sparsity. NIPS 2016. [ (paper) ](https://papers.nips.cc/paper/6288-split-lbi-an-iterative-regularization-path-with-structural-sparsity "NIPS online")


[3] Chendi Huang, Xinwei Sun, Jiechao Xiong, Yuan Yao. Boosting with Structural Sparsity: A Differential Inclusion Approach. Applied and Computational Harmonic Analysis, 2018. [ (paper) ](https://doi.org/10.1016/j.acha.2017.12.004 "ACHA online")
    
[4] Xinwei Sun, Lingjing Hu, Yuan Yao, and Yizhou Wang. GSplit LBI: Taming the Procedural Bias in Neuroimaging for Disease Prediction. Medical Image Computing and Computer Assisted Interventions Conference (MICCAI), Quebec City, Canada, Sept 10-14, 2017. [ (paper) ](https://arxiv.org/abs/1705.09249 "arXiv:1705.09249")
    
[5] Bo Zhao, Xinwei Sun, Yanwei Fu, Yuan Yao, Yizhou Wang. MSplit LBI: Realizing Feature Selection and Dense Estimation Simultaneously in Few-shot and Zero-shot Learning. ICML 2018. [ (paper) ](https://arxiv.org/pdf/1806.04360 "arXiv:1806.04360")

[6] Yanwei Fu, Donghao Li, Xinwei Sun, Shun Zhang, Yizhou Wang, and Yuan Yao. S2-lbi: Stochastic split linearized bregman iterations for parsimonious deep learning. [ (paper) ](https://arxiv.org/abs/1904.10873 "arXiv:1904.10873")
    
[7] Yanwei Fu, Chen Liu, Donghao Li, Xinwei Sun, Jinshan Zeng, Yuan Yao. Parsimonious Deep Learning: A Differential Inclusion Approach with Global Convergence. [ (paper) ](https://arxiv.org/abs/1905.09449 "arXiv:1905.09449")
