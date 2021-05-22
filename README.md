## TT-SPN: Twin Transformers with Sinusoidal Representation Networks for Video Instance Segmentation


"""
Pipeline
"""


Video instance segmentation is the recently introduced computer vision research are that aims joint detection, segmentation and tracking of instances in the video domain. Recent methods proposed highly sophisticated and multi-stage networks that lead to be unusable in practise. Hence, simple yet effective single stage approaches are needed to be used in practise. To fill the gap, we propose end-to-end transformer based video instance segmentation module with Sinusoidal Representation Networks (SPN), namely TT-SPN, to address this problem. TT-SPN, views the VIS task as direct sequence prediction problem in single stage that enables us to aggregate temporal information with spatial one. Set of video frame features are extracted by twin transformers that then propagated to original transformer to produce sequence of instance predictions. These produced instance level information by transformers are then passed through modified Sinusoidal Representation Networks to get end instance level class ids and bounding boxes and self-attended convolutions to get segmentation masks. At its core, TT-SPN is natural paradigm that handles the instance segmentation and tracking via similarity learning that enables system to produce fast and accurate set of predictions. TT-SPN is trained end-to-end with set-based global loss that forces unique predictions via bipartite matching. Thus, general complexity of pipeline is significantly decreased without sacrificing quality of segmentation masks. For the first time, VIS problem is addressed without implicit CNN architectures thanks to twin transformers with being one of the fastest approaches. Our method can be easily divided into its sub-components to produce separate instance masks and bounding boxes that will make it unified approach for many vision tasks.  We benchmark our results on YouTube-VIS dataset by comparing competitive baselines and show that TT-SPN outperforms the base VIS model by significant margin.


# Installation

We provide installation quidelines for TT-SPN. 
First, clone our project page as follows.
```
git clone https://github.com/cankocagil/TT-SPN.git
```
Then, install PyTorch 1.6 and torchvision 0.7:

```
pip install -r requirements.txt
```

Compile DCN module(requires GCC>=5.3, cuda>=10.0)
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
'''

# Training

Training of the model requires at least 8g memory GPU, we performed the experiment on 16g Tesla K80 card. 

To train baseline TT-SPN on a single node with n gpus for 18 epochs, run:
```
python -m torch.distributed.launch --nproc_per_node=n --use_env main.py --ytvos_path /path/to/ytvos 
```

# Inference

```
python inference.py --masks --model_path /path/to/model_weights --save_path /path/to/results.json
```

# Acknowledgement
We would like to thank the [VisTR](https://github.com/Epiphqny/VisTR) and [DETR](https://github.com/facebookresearch/detr) open-source projects for their awesome work, part of the code are modified from their projects.



