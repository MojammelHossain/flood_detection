      # Map Floodwater from Radar Imagery


```Tensorflow.keras``` Implementation

## Introduction

The image segmentation model can be used to extract real-world objects from images, blur backgrounds, create self-driving automobiles and perform other image processing tasks. The goal of this research is to create a mask that shows floodwater in a given location based on .....

## Dataset

The dataset collected from .....

* Water: 1
* NON-Water: 0
* unlabeled: 255

Some examples from dataset.

![Alternate text](/readme/img_id_ayt01.png)
![Alternate text](/readme/img_id_jja60.png)
![Alternate text](/readme/img_id_kuo02.png)


## Models

In this repository we implement UNET using `Keras-TensorFLow` framework.

| Model | Name | Reference |
|:---------------|:----------------|:----------------|
| `unet`      | U-net           | [Ronneberger et al. (2015)](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) |

## Setup

First clone the github repo in your local or server machine by following:
```
git clone 
```

Create a new environment and install dependency from `requirement.txt` file. Before start training check the variable inside config.yaml i.e. `height`, `in_channels`. Keep the above mention dataset in the data folder that give you following structure:

```
--data
    --train
        --train-org-img
        --train-org-img
    --test
        --test-org-img
        --test-org-img
    --val
        --val-org-img
        --val-org-img
    DATASET-VERSION-NOTE-v1.0.txt.txt
```

## Experiments

After setup the required folders and package run the following experiment. There are two experiments based on combination of parameters passing through `argparse` and `config.yaml`. Combination of each experiments given below. 

When you run the following code based on different experiments, some new directories will be created;
1. csv_logger (save all evaluation result in csv format)
2. logs (tensorboard logger)
3. model (save model checkpoint)
4. prediction (validation and test prediction png format)

* **Patchify**: In this experiment we take all the patch images for each image.

```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --index -1 \
    --experiment phr \
    --patchify True \
    --patch_size 256 \
    --weights False \ 
```

* **Patchify Class Balance (P-CB)**: In this experiment we take a threshold value (39%) of water class and remove the patch images for each chip that are less than threshold value.

```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --index -1 \
    --experiment phr_cb \
    --patchify True \
    --patch_size 256 \
    --weights False \
    --patch_class_balance True
```

## Testing

Run following model for evaluating train model on test dataset.

* **PHR and PHR-CB Experiment**

```
python train.py \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --load_model_name my_model.hdf5 \
    --plot_single False \
    --index -1 \
    --patchify True \
    --patch_size 256 \
    --experiment phr \
```

## Result

We train models for all different experiments mention above. Some best and worst prediction result shown below.
Best             |
:-------------------------:
![Alternate text](/readme/best.png)
Worst           |
![Alternate text](/readme/worst.png)

## Overview of the .py files

The following figures are the overview of the important .py files in this repo.

![Alternate text](/readme/fullpipeline.png)

![Alternate text](/readme/dataset.png)

![Alternate text](/readme/utils.png)

![Alternate text](/readme/visualization.png)