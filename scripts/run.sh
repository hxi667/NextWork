cd ..

LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=12 TF_ENABLE_ONEDNN_OPTS=0 python -m torch.distributed.run --master_port=21231 --nproc_per_node=4 main.py --cfgs ./configs/gaitgl/Student_GaitGL_CASIA_B.yaml --phase train --log_to_file

LOCAL_RANK=0 OMP_NUM_THREADS=12 TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.run --master_port=21232 --nproc_per_node=1 main.py --cfgs ./configs/gaitgl/Student_GaitGL_CASIA_B.yaml --phase train --log_to_file

