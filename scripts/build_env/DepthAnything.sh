# run this script in the PointAnything directory.
# bash scripts/build_env/DepthAnything.sh

source activate base

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda init bash

cd Depth-Anything
conda create -n depth-anything python=3.9
conda activate depth-anything
pip install -r requirements.txt

conda deactivate