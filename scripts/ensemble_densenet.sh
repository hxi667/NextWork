cd ..
python main.py --gpu_id 1 --lr 0.1 --batch_size 1024 --teachers [\'vgg19_BN\',\'dpn92\',\'resnet18\',\'preactresnet18\',\'densenet_cifar\'] --student densenet_cifar --d_lr 1e-3 --fc_out 1 --pool_out avg --loss ce --adv 1 --gamma [1,1,1,1,1] --eta [1,1,1,1,1] --name vgg_test --out_layer [-1] --teacher_eval 0

# python main.py --gpu_id 0 --teachers [\'vgg19_BN\'] --student vgg19_BN --d_lr 1e-3 --fc_out 1 --pool_out avg --loss ce --adv 1 --out_layer [0,1,2,3,4] --out_dims [10000,5000,1000,500,10] --gamma [0.001,0.01,0.05,0.1,1] --eta [1,1,1,1,1] --name vgg_test
