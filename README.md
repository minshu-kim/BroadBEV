# BroadBEV
This repository contains the official implementation of BroadBEV, 24' ICRA: [https://arxiv.org/abs/2309.01409](https://arxiv.org/abs/2309.11119)

## Environment Setup
1. Use the below settings. To configure the other packages, refer to [environment.yaml](https://github.com/minshu-kim/BroadBEV/blob/main/environment.yaml).
- Python >= 3.8, <3.9
- OpenMPI = 4.0.4 and mpi4py = 3.0.3 (Needed for torchpack)
- Pillow = 8.4.0
- PyTorch >= 1.9, <= 1.10.2
- tqdm
- torchpack
- mmcv = 1.4.0
- mmdetection = 2.20.0
- nuscenes-dev-kit

2. Run the below command after intalling the python packages.
```
python setup.py develop
```

## Data Preparation
For Setup details, follow the guidelines of [here](https://github.com/mit-han-lab/bevfusion). <br/>
BroadBEV project folder should contain the below files.

``` bash
BroadBEV
├── mmdet3d
├── pretrained
├── configs
├── tools
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
│   │   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl 
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
```

## Pretrained Models
You can download below models on this [link](https://drive.google.com/file/d/1PdQHMZWLFiiZC4cO11nCFb5JKIdLckjI/view?usp=share_link).

## Train & Evaluation
Note that we use 4 NVIDIA A100 80GB in our experiments. <br/>
Any other configurations does not guarantee stable training.
```
bash scripts/train.sh
bash scripts/eval.sh
```

## Acknowlegment
This work is mainly based on [MIT BEVFusion](https://github.com/mit-han-lab/bevfusion), we thank the authors for the contribution.
