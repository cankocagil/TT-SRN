# TT-SRN: Twin Transformers with Sinusoidal Representation Networks for Video Instance Segmentation

 Video instance segmentation is the recently introduced computer vision research that aims at joint detection, segmentation, and tracking of instances in the video domain. Recent methods proposed highly sophisticated and multi-stage networks that are practically unusable. Hence, simple yet effective approaches are needed to be used in practice. To fill the gap, we propose an end-to-end transformer based video instance segmentation module with Sinusoidal Representation Networks (SPN), namely TT-SRN, to address this problem. TT-SRN views the VIS task as a direct sequence prediction problem in single-stage that enables us to aggregate temporal information with spatial one.
 
![TT-SPN](https://github.com/cankocagil/TT-SPN/blob/main/figures/Pipeline.png?raw=true)


Set of video frame features are extracted by twin transformers that then propagated to the original transformer to produce a set of instance predictions. These produced instance level information is then passed through modified SPNs to get end instance level class ids and bounding boxes and self-attended 3-D convolutions to get segmentation masks. At its core, TT-SRN is a natural paradigm that handles the instance segmentation and tracking via similarity learning that enables system to produce a fast and accurate set of predictions. TT-SRN is trained end-to-end with set-based global loss that forces unique predictions via bipartite matching. Thus, the general complexity of the pipeline is significantly decreased without sacrificing the quality of segmentation masks. For the first time, the VIS problem is addressed without implicit CNN architectures thanks to twin transformers with being one of the fastest approaches. Our method can be easily divided into its sub-components to produce separate instance masks and bounding boxes that will make it unified approach for many vision tasks.  We benchmark our results on YouTube-VIS dataset by comparing competitive baselines and show that TT-SRN outperforms the base VIS model by a significant margin.


# Installation

We provide installation quidelines for TT-SPN. 
First, clone our project page as follows.

```
git clone https://github.com/cankocagil/TT-SRN.git
```

We highly recommend the installation of PyTorch and Torchvision beforehand:

```
conda install -c pytorch pytorch torchvision
```

Then, install packages from requirements file:

```
pip install -r requirements.txt
```

Compile Deformable Convolution module(requires GCC>=5.3, cuda>=10.0)

```
cd models/dcn
python setup.py build_ext --inplace
```

# Data Preparation

Download and extract 2019 version of YoutubeVIS  train and val images with annotations from
[CodeLab](https://competitions.codalab.org/competitions/20128#participate-get_data) or [YoutubeVIS](https://youtube-vos.org/dataset/vis/).
TT-SPN expects the following directory structure.
```
TT-SPN
├── data
│   ├── train
│   ├── val
│   ├── annotations
│   │   ├── instances_train_sub.json
│   │   ├── instances_val_sub.json
├── models
...
```

# Training

Training of the model requires at least 8g memory GPU, we performed the experiment on 8g Tesla K80 card. 

To train baseline TT-SPN on a single node with n gpus for 18 epochs, run:

```
python -m torch.distributed.launch --nproc_per_node=n --use_env main.py --ytvos_path /path/to/ytvos --masks
```

# Inference

```
python inference.py --masks --model_path /path/to/model_weights --save_path /path/to/results.json
```

# Acknowledgement
We would like to thank the [VisTR](https://github.com/Epiphqny/VisTR), [DETR](https://github.com/facebookresearch/detr), [Vision Transformers](https://github.com/lucidrains/vit-pytorch) and [SPN](https://vsitzmann.github.io/siren/)  open-source projects for their awesome work, part of the code are modified from their projects.



