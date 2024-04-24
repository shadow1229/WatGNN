# WatGNN
Water position prediction method with SE(3)-Graph Neural Network


## Required Library
[NumPy](https://numpy.org/)  
[PyTorch](https://pytorch.org)  
[SciPy](https://scipy.org/)  
[DGL](https://www.dgl.ai/pages/start.html)  
[a modified SE3Transformer](https://github.com/shadow1229/SE3Transformer)  
[psutil](https://pypi.org/project/psutil/)  

#### Installation
```bash
pip install git+http://github.com/shadow1229/WatGNN
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
[check here](https://github.com/shadow1229/WatGNN/tree/main/watgnn/Dataset)

## Reference
Sangwoo Park, "Water position prediction with SE(3)-Graph Neural Network", _bioRxiv_ (**2024**). [Link](https://www.biorxiv.org/content/10.1101/2024.03.25.586555v1)


