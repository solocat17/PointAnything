# run $ bash DepthMap.sh <input_image_in_input_folder>
# arguments of this script: file name of the input image in ../data/input/

source activate base

# activate the pyskl environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda init bash
conda activate depth-anything

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

input_image="../data/input/"$1
output_dir="../data/intermediate/depth_map"
cd ../../Depth-Anything/

CUDA_VISIBLE_DEVICES=1 python run.py --img-path $input_image --outdir $output_dir --pred-only --grayscale

conda deactivate