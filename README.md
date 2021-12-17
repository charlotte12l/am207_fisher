# AM207 Group Project- Fisher Information Loss


## Intro
This is the repository of AM 207 Group Project - Fisher Information Loss. Basically, the three python files are for constructing the packages
of calculating Fisher Information Loss. $fil.py$ is our own implementation based on the paper's proposed theories. And $fil_torch.py$ is similar 
with the paper's implementation using Pytorch and we used that to support our experiment for Out-of-Distribution data and overfitting in 
$ood_overfit_experiments.ipynb$ because it is more stable. And $fil_paper_experiments.ipynb$ is our similar experiment with that in the paper.

## Group Member
Max Guo, Wenqi Chen, Xingyu Liu, Yang Xiang

## Installation

The code requires Python 3.7+, PyTorch 1.7.1+, and torchvision 0.8.2+.

Create an Anaconda environment and install the dependencies:

    conda create --name fil
    conda activate fil
    conda install -c pytorch pytorch torchvision
    pip install gitpython numpy
    pip install diffprivlib==0.5.0


## How to use
- fil.py:
- fil_torch.py:
- utils.py: 
- fil_paper_experiments.ipynb
- ood_overfit_experiments.ipynb: Extension I, corresponding to section 6.5.1 & 6.5.2
    Experiment for detecting Out-of-Distribution data and Overfitting.  
- diff_privacy.ipynb: Extension II
    Implement differential privacy with Fisher Information Loss 


## Reference
