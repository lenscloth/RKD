#!/usr/bin/env bash
# Teacher
python run.py --dataset stanford --epochs 40 --lr_decay_epochs 25 30 35 --iter_per_epoch 1000  --lr_decay_gamma 0.5 --batch 128\
              --base resnet50 --embedding_size 512 --sample distance --margin 0.2 --save_dir stanford_resnet50_512

# Student with small embedding
python run_distill.py --dataset stanford --epochs 120 --iter_per_epoch 500 --lr_decay_epochs 40 80 --lr_decay_gamma 0.1 --batch 128\
                      --base resnet18 --embedding_size 64 --l2normalize false --dist_ratio 1 --angle_ratio 2 \
                      --teacher_base resnet50--teacher_embedding_size 512 --teacher_load stanford_resnet50_512/best.pth \
                      --save_dir stanford_student_resnet18_64

# Self-Distillation
python run_distill.py --dataset stanford --epochs 120 --iter_per_epoch 500 --lr_decay_epochs 40 80 --lr_decay_gamma 0.1 --batch 128\
                      --base resnet50 --embedding_size 512 --l2normalize false --dist_ratio 1 --angle_ratio 2 \
                      --teacher_base resnet50--teacher_embedding_size 512 --teacher_load stanford_resnet50_512/best.pth \
                      --save_dir stanford_student_resnet50_512
