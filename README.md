# PointAnything: Pointing Estimation from single RGB image

2024 Spring NYCU AI Final Project

## Usage

### Build environments

Make sure that [Anaconda](https://www.anaconda.com/download) is installed in advance.

```bash
# Under the root directory of PointAnything/
bash scripts/build_env/PointAnything.sh
```

The script above will automatically create 3 environments and download 3 pre-trained models.

### Run the project with your own image

```bash
# Under the directory of PointAnything/scripts/inference/
bash Integrated.sh <input_image_in_input_folder> <number_of_usable_GPUs>
```

The script above takes seconds to minutes to estimate the object being pointed in the image.

## Acknowledgement

We sincerely thank [PYSKL](https://github.com/kennymckormick/pyskl), [Depth Anything](https://github.com/LiheYoung/Depth-Anything), [YOLOv10](https://github.com/THU-MIG/yolov10), and others for providing their wonderful code to the community!

*This project is still under construction*
