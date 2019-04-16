# Relational Knowledge Distillation

Implementation of Relational Knowledge Distillation, CVPR 2019\
This repository contains source code of experiments for metric learning.

## Quick Start

```bash
python run.py --help    
python run_distill.py --help

# Train a teacher embedding network of googlnet (d=128)
# using triplet loss (margin=0.2) with distance weighted sampling.
python run.py --mode train \ 
               --dataset cub200 \
               --base googlenet \
               --sample distance \ 
               --margin 0.2 \ 
               --embedding_size 128 \
               --save_dir teacher

# Evaluate the teacher embedding network
python run.py --mode eval \ 
               --dataset cub200 \
               --base googlenet \
               --embedding_size 128 \
               --load teacher/best.pth 

# Distill the teacher to student embedding network
python run_distill.py --dataset cub200 \
                      --base googlnet \
                      --embedding_size 64 \
                      --teacher_base googlenet \
                      --teacher_embedding_size 128 \
                      --teacher_load teacher/best.pth \
                      --dist_ratio 1  \
                      --angle_ratio 2 \
                      --save_dir student
                      
# Distill the trained model to student network
python run.py --mode eval \ 
               --dataset cub200 \
               --base googlenet \
               --embedding_size 64 \
               --load student/best.pth 
            
```


##  Dependency

* Python 3.6
* Pytorch 1.0
* tqdm (pip install tqdm)


### Note
* Hyper-parameters that used for experiments in the paper are specified at scripts in ```exmples/```.
* Do not use distance weighted sampling ( --sample distance ) when embedding is not l2 normalized. As the sampling method assume that embedding is normalized.
