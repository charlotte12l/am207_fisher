# AM207 Group Project- Fisher Information Loss


## Intro
This is the repository of AM 207 Group Project - Fisher Information Loss. Basically, the three python files are for constructing the packages
of calculating Fisher Information Loss. `fil.py` is our own implementation based on the paper's proposed theories. And `fil_torch.py` is similar 
with the paper's implementation using Pytorch and we used that to support our experiment for Out-of-Distribution data and overfitting in 
`ood_overfit_experiments.ipynb` because it is more stable. And `fil_paper_experiments.ipynb` is our similar experiment with that in the paper.

## Group Member
Max Guo, Wenqi Chen, Xingyu Liu, Yang Xiang

## Installation

The code requires Python 3.7+, PyTorch 1.7.1+, and torchvision 0.8.2+.

Create an Anaconda environment and install the dependencies:

    conda create --name fil_207
    conda activate fil_207
    conda install -c pytorch pytorch torchvision
    pip install gitpython numpy scikit-learn


## How to use
- fil.py: Our own implementation of FIL.
- fil_torch.py: Our initial exploration to learn the paper, similar to paper's implementation using Pytorch.
- utils.py: Some util functions.
- fil_paper_experiments.ipynb: Fully reproduced all the experiments in the paper, including the attacking part.
- experiments.ipynb: Our own experiments.
- ood_overfit_experiments.ipynb: Extension I, corresponding to section 6.5.1 & 6.5.2
    Experiment for detecting Out-of-Distribution data and Overfitting.  



## AM207 Requirements
- At least one clear working pedagogical example demonstrating the problem the paper is claiming to solve.
    - We implement FIL for two toy datasetsfor both linear and logistic regression as pedagogical examples demonstrating successfully the ability of FIL to characterize data points that are more prone to leakage in `experiments.ipynb`.

- At lease a bare bones implementation of the model/algorithm/solution.
    - We've successfully implemented FIL and IRFIL in `fil.py` 

- Demonstration on at least one instance that your implementation solves the problem.
    - On toy dataset, FIL and successfully characterize data points that are more prone to leakage.

- Demonstration on at least one instance the failure mode of the model/algorithm/solution, with an explanation for why failure occurred (is the dataset too large? Did you choose a bad hyper parameter?). The point of this is to point out edge cases to the user.
    - We give instances of successes and failures of applying FIL for the MNIST dataset.

- Extensions
    - FIL for new applications in OOD and overfitting detection.

## Reference

Measuring Data Leakage in Machine-Learning Models with Fisher Information. Hannun, Awni and Guo, Chuan and van der Maaten, Laurens.
https://github.com/merlionctc/cs107-FinalProject