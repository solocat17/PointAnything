# run this script in the PointAnything directory.
# bash scripts/build_env/DepthAnything.sh

source activate base

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda init bash

cd YOLO-World
conda create -n yolo-world python=3.9
conda activate yolo-world
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -e .

pip install -U openmim
mim install mmcv==2.0.0
mim install mmdet==3.0.0
mim install mmengine==0.10.3
mim install mmyolo==0.6.0

pip install ultralytics
pip install openai-clip
pip install supervision==0.18.0
pip install transformers==4.38.1

cd ../models
wget https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain_800ft-9df82e55.pth

conda deactivate