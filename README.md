# cg2all
Convert coarse-grained protein structure to all-atom model

## Installation
These steps will install Python libraries including [cg2all (this repository)](https://github.com/huhlim/cg2all), [a modified MDTraj](https://github.com/huhlim/mdtraj), [a modified SE3Transformer](https://github.com/huhlim/SE3Transformer), and other dependent libraries. The installation steps also place executables `convert_cg2all` and `convert_all2cg` in your python binary directory.

This package is tested on Linux (CentOS) and MacOS (Apple Silicon, M1).

#### for CPU only
```bash
pip install git+http://github.com/huhlim/cg2all
```
#### for CUDA (GPU) usage
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Create an environment with [DGL](https://www.dgl.ai/pages/start.html) library with CUDA support
```bash
# This is an example with cudatoolkit=11.3.
# Set a proper cudatoolkit version that is compatible with your CUDA drivier and DGL library.
# dgl>=1.1 occassionally raises some errors, so please use dgl<=1.0.
conda create --name cg2all pip cudatoolkit=11.3 dgl=1.0 -c dglteam/label/cu113
```
3. Activate the environment
```bash
conda activate cg2all
```
4. Install this package
```bash
pip install git+http://github.com/huhlim/cg2all
```

## Usages
### convert_cg2all
convert a coarse-grained protein structure to all-atom model
```bash
usage: convert_cg2all [-h] -p IN_PDB_FN [-d IN_DCD_FN] -o OUT_FN [-opdb OUTPDB_FN]
                      [--cg {supported_cg_models}] [--chain-break-cutoff CHAIN_BREAK_CUTOFF] [-a]
                      [--fix] [--ckpt CKPT_FN] [--time TIME_JSON] [--device DEVICE] [--batch BATCH_SIZE] [--proc N_PROC]

options:
  -h, --help            show this help message and exit
  -p IN_PDB_FN, --pdb IN_PDB_FN
  -d IN_DCD_FN, --dcd IN_DCD_FN
  -o OUT_FN, --out OUT_FN, --output OUT_FN
  -opdb OUTPDB_FN
  --cg {supported_cg_models}
  --chain-break-cutoff CHAIN_BREAK_CUTOFF
  -a, --all, --is_all
  --fix, --fix_atom
  --standard-name
  --ckpt CKPT_FN
  --time TIME_JSON
  --device DEVICE
  --batch BATCH_SIZE
  --proc N_PROC
```
#### arguments
* -p/--pdb: Input PDB file (**mandatory**).
* -d/--dcd: Input DCD file (optional). If a DCD file is given, the input PDB file will be used to define its topology.
* -o/--out/--output: Output PDB or DCD file (**mandatory**). If a DCD file is given, it will be a DCD file. Otherwise, a PDB file will be created.
* -opdb: If a DCD file is given, it will write the last snapshot as a PDB file. (optional)
* --cg: Coarse-grained representation to use (optional, default=CalphaBasedModel).
  - CalphaBasedModel: CA-trace (atom names should be "CA")
  - ResidueBasedModel: Residue center-of-mass (atom names should be "CA")
  - SidechainModel: Sidechain center-of-mass (atom names should be "SC")
  - CalphaCMModel: CA-trace + Residue center-of-mass (atom names should be "CA" and "CM")
  - CalphaSCModel: CA-trace + Sidechain center-of-mass (atom names should be "CA" and "SC")
  - BackboneModel: Model only with backbone atoms (N, CA, C)
  - MainchainModel: Model only with mainchain atoms (N, CA, C, O)
  - Martini: [Martini](http://cgmartini.nl/) model
  - Martini3: [Martini3](http://www.cgmartini.nl/index.php/martini-3-0) model
  - PRIMO: [PRIMO](http://dx.doi.org/10.1002/prot.22645) model
* --chain-break-cutoff: The CA-CA distance cutoff that determines chain breaks. (default=10 Angstroms)
* --fix/--fix_atom: preserve coordinates in the input CG model. For example, CA coordinates in a CA-trace model will be kept in its cg2all output model.
* --standard-name: output atom names follow the IUPAC nomenclature. (default=False; output atom names will use CHARMM atom names)
* --ckpt: Input PyTorch ckpt file (optional). If a ckpt file is given, it will override "--cg" option.
* --time: Output JSON file for recording timing. (optional)
* --device: Specify a device to run the model. (optional) You can choose "cpu" or "cuda", or the script will detect one automatically. </br>
  "**cpu**" is usually faster than "cuda" unless the input/output system is really big or you provided a DCD file with many frames because it takes a lot for loading a model ckpt file on a GPU.
* --batch: the number of frames to be dealt at a time. (optional, default=1)
* --proc: Specify the number of threads for loading input data. It is only used for dealing with a DCD file. (optional, default=OMP_NUM_THREADS or 1)

#### examples
Conversion of a PDB file
```bash
convert_cg2all -p tests/1ab1_A.calpha.pdb -o tests/1ab1_A.calpha.all.pdb --cg CalphaBasedModel
```
Conversion of a DCD trajectory file
```bash
convert_cg2all -p tests/1jni.calpha.pdb -d tests/1jni.calpha.dcd -o tests/1jni.calpha.all.dcd --cg CalphaBasedModel
```
Conversion of a PDB file using a ckpt file
```bash
convert_cg2all -p tests/1ab1_A.calpha.pdb -o tests/1ab1_A.calpha.all.pdb --ckpt CalphaBasedModel-104.ckpt
```
<hr/>


## Datasets
The training/validation/test sets are available at [zenodo](https://zenodo.org/record/8273739).


## Reference
Lim Heo & Michael Feig, "One particle per residue is sufficient to describe all-atom protein structures", _bioRxiv_ (**2023**). [Link](https://www.biorxiv.org/content/10.1101/2023.05.22.541652v1)


[![DOI](https://zenodo.org/badge/585390653.svg)](https://zenodo.org/doi/10.5281/zenodo.10009208)
