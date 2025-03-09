# Detection models

In this section, we provide the code to test Convolution-based and Transformer-based models on the CoMix dataset. 
You can find the weights [here](https://drive.google.com/drive/folders/1RBhVKPscDuycDqiD8zHgfSu35xHM11Ty).

Create a subfolder `weights` inside `benchmarks`. Now inside `weights` create three subdirectories `dass`, `faster-rcnn` and `yolov8`.

Regarding `dass` and `faster-rcnn` put all the `.pth` files in the corresponding folders. Instead, in `yolov8` put ONLY `yolov8x-best.pt`, `yolov8x-c100.pt`, `yolov8x-m109.pt` and `yolov8x-mix.pt`.

All predictions will be stored in `data/predicts.coco`.
For simplicity, we are going to create a `conda` environment for all the scripts that are needed and in order to install every dependency.

### DASS

First of all under `CoMix` create a folder `modules`and inside of it clone this [repository](https://github.com/emanuelevivoli/DASS_Det_Inference).

Then, create the `conda` environment and activate it (remeber to deactivate the conda environment you were using before!):
```bash
$ conda create -n dass python=3.8
$ conda activate dass
```
Now, install dependencies:
```bash
$ pip install numpy
$ pip install chainercv
$ pip install -e .
$ pip install loguru
$ pip install xmltodict
$ pip install requests-doh
```
Run the following command for every dataset and for every weights file:
```bash
$ python benchmarks/detections/dass.py -n dataset_name -s split_name -pd weights_file
```
Where:
dataset_name -> choices = ['eBDtheque', 'DCM', 'comics', 'popmanga']

split_name -> choices = ['val', 'test']

weights_file -> choices =['m109', 'dcm', 'mixdata']

### Faster R-CNN

Create the `conda` environment and activate it (remember to deactivate the conda environment you were using before!):
```bash
$ conda create -n faster python=3.8
$ conda activate faster
```
Now, install dependencies:
```bash
$ pip install -e .
```
Run the following command:

```bash
$ python benchmarks/detections/faster_rcnn.py -n dataset_name -s split_name -wn weights_file
```
Where:
dataset_name -> choices = ['eBDtheque', 'DCM', 'comics', 'popmanga']

split_name -> choices = ['val', 'test', 'all']

weights_file -> choices =['faster_rcnn-c100-best-10052024_092536.pth', 'faster_rcnn-c100-last-10052024_092536.pth',
'faster_rcnn-m109-best-10052024_094048.pth', 'faster_rcnn-mix-best-10052024_112437.pth']

### YOLOv8

Create the `conda` environment and activate it (remember to deactivate the conda environment you were using before!):
```bash
$ conda create -n yolo python=3.8
$ conda activate yolo
```
Now, install dependencies:
```bash
$ pip install -e .
```
Run the following command:

```bash
$ python benchmarks/detections/yolov8.py -n dataset_name -s split_name -wn weights_file
```
Where:
dataset_name -> choices = ['eBDtheque', 'DCM', 'comics', 'popmanga']

split_name -> choices = ['val', 'test', 'all']

weights_file -> choices =['yolov8x-best.pt', 'yolov8x-c100.pt',
'yolov8x-m109.pt', 'yolov8x-mix.pt']


### GroundingDINO

Create the `conda` environment and activate it (remember to deactivate the conda environment you were using before!):
```bash
$ conda create -n dino python=3.8
$ conda activate dino
```
Now, install dependencies:
```bash
$ pip install -e .
$ pip install requests-doh
```
Run the following command:

```bash
$ python benchmarks/detections/groundingdino.py -n dataset_name -s split_name
```
Where:
dataset_name -> choices = ['eBDtheque', 'DCM', 'comics', 'popmanga']

split_name -> choices = ['all', 'val', 'test']

### Magi

Create the `conda` environment and activate it (remember to deactivate the conda environment you were using before!):
```bash
$ conda create -n magi python=3.8
$ conda activate magi
```
Now, install dependencies:
```bash
$ pip install -e .
$ pip install requests-doh
$ pip install transformers==4.45.2
```
Run the following command:

```bash
$ python benchmarks/detections/magi.py -n dataset_name -s split_name
```
Where:
dataset_name -> choices = ['eBDtheque', 'DCM', 'comics', 'popmanga']

split_name -> choices = ['val', 'test']

### SSD

To be added.
