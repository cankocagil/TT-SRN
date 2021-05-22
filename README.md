## TT-SPN: Twin Transformers with Sinusoidal Representation Networks for Video Instance Segmentation



### Installation
We provide instructions how to install dependencies via conda.
First, clone the repository locally:
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

### Preparation

Download and extract 2019 version of YoutubeVIS  train and val images with annotations from
[CodeLab](https://competitions.codalab.org/competitions/20128#participate-get_data) or [YoutubeVIS](https://youtube-vos.org/dataset/vis/).
We expect the directory structure to be the following:
```
VisTR
├── data
│   ├── train
│   ├── val
│   ├── annotations
│   │   ├── instances_train_sub.json
│   │   ├── instances_val_sub.json
├── models
...
```


### Training

Training of the model requires at least 8g memory GPU, we performed the experiment on 16g Tesla K80 card. 

To train baseline TT-SPN on a single node with n gpus for 18 epochs, run:
```
python -m torch.distributed.launch --nproc_per_node=n --use_env main.py --ytvos_path /path/to/ytvos 
```

### Inference

```
python inference.py --masks --model_path /path/to/model_weights --save_path /path/to/results.json
```

### Acknowledgement
We would like to thank the [VisTR](https://github.com/Epiphqny/VisTR) open-source project for its awesome work, part of the code are modified from its project.



