# bash scripts/build_env/PointAnything.sh
git clone https://github.com/kennymckormick/pyskl.git
git clone https://github.com/LiheYoung/Depth-Anything.git
git clone https://github.com/THU-MIG/yolov10.git

# create folder for input, output, intermediate data and models
mkdir -p data/output
mkdir -p data/intermediate/images
mkdir -p data/intermediate/products/skeleton
mkdir -p data/intermediate/products/depth
mkdir -p models

# pyskl
cd pyskl
conda env create -f pyskl.yaml
conda activate pyskl
pip install -e .
cd ../models
wget https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth
wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth
cd ../scripts/build_env
cp skeleton.py ../../pyskl/demo/
conda deactivate

# Depth-Anything
cd ../Depth-Anything
conda create -n depth-anything python=3.9
conda activate depth-anything
pip install -r requirements.txt
conda deactivate

# YOLOv10
cd yolov10
conda create -n yolov10 python=3.9
conda activate yolov10
pip install -r requirements.txt
pip install -q supervision
pip install -e .
cd ../models
wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt
conda deactivate