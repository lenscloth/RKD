# Relational Knowledge Distillation

Official implementation of [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068?context=cs.LG), CVPR 2019\
This repository contains source code of experiments for metric learning.


## Quick Start

```bash
python run.py --help    
python run_distill.py --help

# Train a teacher embedding network of resnet50 (d=512)
# using triplet loss (margin=0.2) with distance weighted sampling.
python run.py --mode train \ 
               --dataset cub200 \
               --base resnet50 \
               --sample distance \ 
               --margin 0.2 \ 
               --embedding_size 512 \
               --save_dir teacher

# Evaluate the teacher embedding network
python run.py --mode eval \ 
               --dataset cub200 \
               --base resnet50 \
               --embedding_size 512 \
               --load teacher/best.pth 

# Distill the teacher to student embedding network
python run_distill.py --dataset cub200 \
                      --base resnet18 \
                      --embedding_size 64 \
                      --l2normalize false \
                      --teacher_base resnet50 \
                      --teacher_embedding_size 512 \
                      --teacher_load teacher/best.pth \
                      --dist_ratio 1  \
                      --angle_ratio 2 \
                      --save_dir student
                      
# Distill the trained model to student network
python run.py --mode eval \ 
               --dataset cub200 \
               --base resnet18 \
               --l2normalize false \
               --embedding_size 64 \
               --load student/best.pth 
            
```


##  Dependency

* Python 3.6
* Pytorch 1.0
* tqdm (pip install tqdm)
* h5py (pip install h5py)
* scipy (pip install scipy)

### Note
* Hyper-parameters that used for experiments in the paper are specified at scripts in ```exmples/```.
* Heavy teacher network (ResNet50 w/ 512 dimension) requires more than 12GB of GPU memory if batch size is 128.  
  Thus, you might have to reduce the batch size. (The experiments in the paper were conducted on P40 with 24GB of gpu memory. 
)

## Citation
In case of using this source code for your research, please cite our paper.

```
@inproceedings{park2019relational,
  title={Relational Knowledge Distillation},
  author={Park, Wonpyo and Kim, Dongju and Lu, Yan and Cho, Minsu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3967--3976},
  year={2019}
}
```
