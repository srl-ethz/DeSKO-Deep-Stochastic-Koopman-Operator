## Conda environment
From the general python package sanity perspective, it is a good idea to use conda environments to make sure packages from different projects do not interfere with each other.


To create a conda env with python3, one runs 
```bash
conda create -n test python=3.6
```
To activate the env: 
```
source activate test
```

# Installation Environment

```bash
pip install numpy==1.16.3
pip install tensorflow==1.13.1
pip install tensorflow-probability==0.6.0
pip install opencv-python
pip install cloudpickle
pip install gym
pip install matplotlib
pip install Mosek
pip install progressbar
pip install pandas
pip install pybullet
pip install xacro
```


Then you will have to install two packages realted to the optimization of MPC, the cvxpy and scs. These packages can't
be found by using pip install, and need to be downloaded from the following website and installed manually:

```
https://www.lfd.uci.edu/~gohlke/pythonlibs/
```

For python 3.6, download the `cvxpy‑1.1.7‑cp36‑cp36m‑win_amd64.whl` and `scs‑2.1.2‑cp36‑cp36m‑win_amd64.whl`.
Then bash in the folder that contains the downloaded folder (remember to 'source activate test' first) run the following:
```
pip install cvxpy‑1.1.7‑cp36‑cp36m‑win_amd64.whl
pip install scs‑2.1.2‑cp36‑cp36m‑win_amd64.whl
```

Now, you are ready to run the experiments that don't require the MuJoCo simulator.

# Setting up MuJoCo

Apply academic personal license on https://www.roboti.us/license.html. Download the mjpro150 and unzip under 
the directory: ~/.mujoco. Also put the applied license in this folder. 

Run the following line to install necessary packages
```
sudo apt-get install libgl1-mesa-dev libosmesa6-dev patchelf
```
Add the following lines to the end of ~/.bahsrc:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro150/bin
export MUJOCO_KEY_PATH=~/.mujoco${MUJOCO_KEY_PATH}
```
Finally, run the following command in the console:
```
source activate test
pip3 install -U 'mujoco-py<1.50.2,>=1.50.1'
```
