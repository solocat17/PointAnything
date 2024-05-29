# run this script in the PointAnything directory.
# bash scripts/build_env/pyskl.sh

source activate base

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda init bash

cd pyskl
conda env create -f pyskl.yaml
conda activate pyskl
pip install -e .

# download the pre-trained model to the models directory
cd ../models
wget https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth
wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth

# move skeleton.py to the demo directory
cd ../scripts/inference
cp skeleton.py ../../pyskl/demo/
conda deactivate