# FLOOD DETECTION USING SEMANTIC SEGMENTATION

## Introduction

Floods are one of the worst natural disasters that affect the civilization anywhere in the world. The project aims at identifying floods based on images supplied. In the past decade, reported annual losses from floods have reached tens of billions of US dollars and thousands of people were killed each year. Natural disasters are frequent in this time and era. Flash floods are becoming common now due to climate change. We aim to make planning after a flood easy for the authorities. This allows for easy flood relief operations and rescue.

## Dataset

After Hurricane Harvey, the data is collected using a small UAV platform, DJI Mavic Pro quadcopters. On August 5, 2017, Hurricane Harvey made landfall as a Category 4 hurricane close to Texas and Louisiana. The Harvey dataset is made up of video and images captured during several flights that were made between August 30 and September 4 in Texas's Ford Bend County and other directly affected locations.
There are two reasons why the dataset is special. One is fidelity: it contains imagery from an unmanned aerial vehicle (UAV) captured by emergency responders during the response phase; as a result, the data represents current best practices and may reasonably be anticipated to be gathered during a disaster. Second, it is the only database of sUAV imagery for disasters that is currently known.
The dataset was obtained from the FloodNet challenge. This includes 1445 photos for training, 450 images for validation, and 448 images for the test set, for a total of 2343 images segmented into 10 classes.

Below are some example from our dataset.
![Alternate text](/other/sample.jpg)

## Data Preprocessing

The data preprocessing approach we employ is NAP, which was first proposed in [FAPNET](https://www.mdpi.com/1424-8220/22/21/8245) article. From each image in the dataset, we extract patches of a specific size (512 X 512) using the NAP augmentation module. Following the use of the NAP module, we normalized the pixel value between (0,1). Mask has also been split into two categories. The first is flooded streets, buildings, and water (Indicated as 1). The remainder will be changed to 0.

## Models

In this repository we implement the following models.

| Model | Name | Reference | FrameWork |
|:---------------|:----------------|:----------------|:----------------|
| `unet`      | U-net           | [Ronneberger et al. (2015)](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) | PyTorch|
| `unet`      | U-net           | [Ronneberger et al. (2015)](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) | Keras-TensorFlow|
| `linknet`     | LINK-Net         | [Chaurasia et al. (2017)](https://arxiv.org/pdf/1707.03718.pdf) | [segmentation_models](https://github.com/qubvel/segmentation_models)


## Training

First, download the mentioned dataset and pass the data path in the `config.yaml` file. Also, remember to check the variable inside the file. Now, run the following in cmd for train and test respectively. While evaluating the model remember to pass the variable under the Evaluation section in `config.yaml`.

```
python train.py
```
```
python test.py
```

## Result

Following are the model results after five epochs on test dataset. 

| Model | MeanIoU |
|:----------------|:----------------|
| `unet`      | 0.4865          |
| `linknet`      | 0.4266         |

Some prediction on test dataset.
![Alternate text](/other/pred.png)