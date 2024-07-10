## DeSKO: Deep Stochstic Koopman Operator

This is the code for ICLR 2022 paper: 

[DeSKO: Stability-Assured Robust Control with a Deep Stochastic Koopman Operator](https://openreview.net/pdf?id=hniLRD_XCA)

by Minghao Han, Jacob Euler-Rolle, Robert K Katzschmann

## Conda environment
From the general python package sanity perspective, it is a good idea to use conda environments to make sure packages from different projects do not interfere with each other.

To create a conda env with python3, one runs 
```bash
conda create -n test python=3.7
```
To activate the env: 
```
source activate test
```

# Installation Environment

```bash
pip install tensorflow==1.13.1
pip install tensorflow-probability==0.6.0
pip install opencv-python
pip install cloudpickle
pip install gym
pip install numpy
pip install matplotlib
pip install Mosek
pip install progressbar
pip install pandas
pip install pybullet
pip install xacro
pip install cvxpy
```

Now, you are ready to run the experiments.


