# run $ bash skeleton.sh <input_image_in_input_folder>
# arguments of this script: file name of the input image in ../data/input/

source activate base

# activate the pyskl environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda init bash
conda activate pyskl

# parse the arguments
if [ $# -ne 1 ]; then
  echo "Usage: $0 <input_image>"
  exit 1
fi

# if the input image file does not exist
if [ ! -f ../../data/input/$1 ]; then
  echo "The input image file does not exist."
  exit 1
fi

# set the input image file name
input_image=$1

# copy skeleton.py to the ../pyskl/demo/ directory
# cp skeleton.py ../../pyskl/demo/

# run the skeleton.py script
cd ../../pyskl/
CUDA_VISIBLE_DEVICES=1 python demo/skeleton.py $input_image

# deactivate the pyskl environment
conda deactivate