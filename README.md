# FREE
This repository contains the reference code for the paper "**FREE: Feature Refinement for Generalized Zero-Shot Learning**". [[arXiv]]()[[Paper]]()

![](images/pipeline.png)


## Preparing Dataset and Model
Datasets can be download from [Xian et al. (CVPR2017)](https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip) and take them into dir `data`.
## Requirements
The code implementation of **GNDAN** mainly based on [PyTorch](https://pytorch.org/). All of our experiments run in Python 3.8.8.

## Runing
Before running commands, you can set the hyperparameters in config.py. Please run the following commands and testing **FREE** on different datasets: 
```
$ python ./image-scripts/run-cub.py       #CUB
$ python ./image-scripts/run-sun.py       #SUN
$ python ./image-scripts/run-flo.py       #FLO
$ python ./image-scripts/run-awa1.py      #AWA1
$ python ./image-scripts/run-awa2.py      #AWA2
```

**Note**: All of above results are run on a server with one GPU (Nvidia 1080Ti).


## Citation
If this work is helpful for you, please cite our paper.

```
@inproceedings{Chen2021FREE,  
  title={FREE: Feature Refinement for Generalized Zero-Shot Learning},    
  author={Chen, Shiming and Wang, Wenjie and Xia, Beihao and Peng, Qinmu and You, Xinge and Zheng, Feng and Shao,  Ling},    
  booktitle={2021 IEEE/CVF International Conference on Computer Vision (ICCV)}, 
  year={2021}    
}
```
