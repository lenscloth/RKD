#!/usr/bin/env bash
# Teacher Network
python run.py --dataset cub200 --epochs 40 --lr_decay_epochs 25 30 35 --lr_decay_gamma 0.5 --batch 128\
              --base resnet50 --sample distance --margin 0.2 --embedding_size 512 --save_dir cub200_resnet50_512

# Student with small embedding
python run_distill.py --dataset cub200 --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1  --batch 128\
                      --base resnet18 --embedding_size 128 --l2normalize false --dist_ratio 1 --angle_ratio 2 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load cub200_resnet50_512/best.pth \
                      --save_dir cub200_student_resnet18_128

# Self-Distillation
python run_distill.py --dataset cub200 --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1  --batch 128\
                      --base resnet50 --embedding_size 512 --l2normalize false --dist_ratio 1 --angle_ratio 2 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load cub200_resnet50_512/best.pth \
                      --save_dir cub200_student_resnet50_512
