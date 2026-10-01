# WatGNN
Water position prediction method with SE(3)-Graph Neural Network

<img src="./images/image1.png" height="400"/>

<img src="./images/image2.png" height="500"/>

WatGNN predicts water positions around proteins and protein–compound complexes. This method places four probe points near each eligible noncarbon atom, then uses an SE(3)-equivariant graph neural network to score possible water sites and predict three-dimensional shifts from those probe points. The shifted positions are filtered by score, and predictions might have steric clash with input atoms or another duplicate predictions are removed.


## Installation (Uses Anaconda)

### For Linux/NVIDIA (Recommended)
Due to DGL's supported python and pytorch version issue, this method will install Python 3.12, PyTorch 2.4.0, and cuda 12.1.
This method was tested on Ubuntu 22.09 with Intel i9-12900k CPU and NVIDIA RTX 4090.
```bash
conda env create -f environment.yml -n watgnn
conda activate watgnn
conda list mkl
#for module import test
python -c "import torch, dgl; print(torch.__version__, torch.version.cuda, dgl.__version__)"
```

### For Windows/NVIDIA
Please install Windows Subsystem for Linux[https://learn.microsoft.com/en-us/windows/wsl/install] (WSL) on windows and follow Linux/NVIDIA Installation method.

### For MacOS (Will be installed, but not recommended)
WARNING: Following installation will use CPU for the prediction.
This method will install Python 3.11 and PyTorch 2.1.1.
Tested on MacBook Air 2020 (Apple M1 + 16GB DRAM)
```bash
#use MacOS version of pyproject.toml instead of default toml file.
mv pyproject_macos.toml pyproject.toml

conda env create -f environment_macos.yml
conda activate watgnn-macos

python -m pip install "numpy==1.26.4" "torch==2.1.1" "torchdata==0.7.1" "dgl==2.2.0"
python -m pip install -e .
#for module import test
python -c "import torch, torchdata.datapipes.iter, dgl; print(torch.__version__, dgl.__version__)"
```

## Usages
```bash
usage: watgnn.py [dataset file path] 
```

#### dataset file structure
for each line:
(Input PDB/CIF file path) (Input mol2 file path(optional))

example:
```bash
./1adl.pdb ./1adl.mol2
./1ubq.pdb
./2fwh.pdb ./2fwh.mol2
```

## Dataset used in the preprint
Now the dataset and precalculated data is located at the differet Repository, [https://github.com/shadow1229/WatGNN_SI/](https://github.com/shadow1229/WatGNN_SI/)
Dataset: [check here](https://github.com/shadow1229/WatGNN_SI/tree/main/Dataset)
Precalculated data: [check here](https://github.com/shadow1229/WatGNN_SI/tree/main/Precalculated_data)
## Reference
Sangwoo Park, "Water position prediction with SE(3)-Graph Neural Network", _bioRxiv_ (**2024**). [Link](https://www.biorxiv.org/content/10.1101/2024.03.25.586555v1)


