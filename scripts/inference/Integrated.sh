# run $ bash Integrated.sh <input_image_in_input_folder> <number_of_usable_GPUs>

source activate base

# activate the pyskl environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda init bash

# parse the arguments
if [ $# -ne 1 ] && [ $# -ne 2 ]; then
  echo "Usage: $0 <input_image> [<number_of_usable_GPUs>]"
  exit 1
fi

# $2 is the number of usable GPUs
if [ $# -eq 2 ]; then
  export CUDA_VISIBLE_DEVICES=$2
fi

# if the input image file does not exist
if [ ! -f ../../data/input/$1 ]; then
  echo "The input image file does not exist."
  exit 1
fi

conda activate pyskl
cd ../../pyskl/
pyskl_input_image=$1
python demo/skeleton.py $pyskl_input_image
conda deactivate
cd ../scripts/inference/

conda activate depth-anything
da_input_image="../data/input/"$1
da_output_dir="../data/intermediate/depth_map"
cd ../../Depth-Anything/
python run.py --img-path $da_input_image --outdir $da_output_dir --pred-only --grayscale
conda deactivate
cd ../scripts/inference/

conda activate yolov10
input_image=$1
python Integrated.py --img-name $input_image
conda deactivate