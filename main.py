from __future__ import print_function

import torch
import torch.optim as optim

import yaml
import os
import argparse

from models import *
from models import discriminator, teachers_student
from utils import get_msg_mgr, init_seeds
from models.losses import lossmap

from data.transform import get_transform
from data.dataloader import get_loader



# ================= Arugments ================ #
parser = argparse.ArgumentParser(description='Training Gait with PyTorch')
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
parser.add_argument('--cfgs', type=str, default='./configs/default.yaml', help="path of config file")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<exp_name>/<Student_name>/<logs>/<Datetime>.txt")
#  ============================================================

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--out_layer', default="[-1]", type=str, help='the type of pooling layer of output')  # eval()


# model config
parser.add_argument('--depth', type=int, default=26)
parser.add_argument('--base_channels', type=int, default=96)
parser.add_argument('--input_shape', default="(1, 3, 32, 32)", type=str, help='the size of input data')  # eval()
parser.add_argument('--n_classes', default=10, type=int, help='the number of classes') 
parser.add_argument('--out_dims', default="[5000,1000,500,200,10]", type=str, help='the dims of output pooling layers')  # eval()
parser.add_argument('--fc_out', default=1, type=int, help='if immediate output from fc-layer')
parser.add_argument('--pool_out', default="max", type=str, help='the type of pooling layer of output')

# run config
parser.add_argument('--outdir', type=str, default="results")

# optim config
parser.add_argument('--epochs', type=int, default=1800)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--base_lr', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nesterov', type=bool, default=True)  # SGD优化器的参数，enables Nesterov momentum
parser.add_argument('--lr_min', type=float, default=0)

args = parser.parse_args()


# init SummaryWriter and logger(to tensorboard, console and file print log info)， random seeds
def initialization(cfgs, training):
    # 获得 MessageManager 类的实例对象
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg'] 

    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'], engine_cfg['exp_name'], 'Student_'+ cfgs['model_cfg']['student'])
    if training:
        # 初始化 SummaryWriter 和 logger
        msg_mgr.init_manager(output_path, args.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)

    # 写 trainer 或 evaluator 的配置信息到 console/file 
    msg_mgr.log_info(engine_cfg)

    seed = torch.distributed.get_rank()
    # 初始化随机种子
    init_seeds(seed)
    


if __name__ == '__main__':
    torch.distributed.init_process_group('nccl', init_method='env://')
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of available GPUs({}) equals to the world size({}).".format(
            torch.cuda.device_count(), torch.distributed.get_world_size()))
    
    # Cuda setup
    device = torch.distributed.get_rank()
    torch.cuda.set_device(device)

    # ================= Load Config File ================ #
    with open(args.cfgs, 'r') as stream:
        cfgs = yaml.safe_load(stream)
     
    # ================= Initialization SummaryWriter and logger， random seeds ================ #
    initialization(cfgs, training=True)
    
    # TODO
    best_acc = 0  # best test accuracy ???
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch ???
    
    msg_mgr = get_msg_mgr()

    # ================= Data Loader ================ #
    msg_mgr.log_info('==> ==> Preparing Data..')

    train_transform = get_transform(cfgs['trainer_cfg']['transform'])
    test_transform = get_transform(cfgs['evaluator_cfg']['transform'])

    train_loader = get_loader(cfgs, train=True)
    test_loader = get_loader(cfgs, train=False)
    
    cfgs['data_cfg']['steps_per_epoch'] = len(train_loader)

    # ================= Model Setup ================ #
    msg_mgr.log_info('==> ==> Building Model..')

    # get models as teachers and students
    teachers, student = teachers_student.get_teachers_student(cfgs['model_cfg'], cfgs['data_cfg']['dataset_name'], device)

    msg_mgr.log_info("Teacher(s): ")
    msg_mgr.log_info([teacher.__name__ for teacher in teachers])
    msg_mgr.log_info("Student: ")
    msg_mgr.log_info([student.__name__])
    
    # TODO
    dims = [10]
    # dims = [student.out_dims[i] for i in eval(args.out_layer)]
    msg_mgr.log_info(["student dims: ", dims])

    update_parameters = [{'student params': student.parameters()}]

    # discriminator
    if cfgs['discriminator_cfg']['adv']:
        discriminators = discriminator.Discriminators(dims, grl=cfgs['discriminator_cfg']['grl'])
        for d in discriminators.discriminators:
            d = d.to(device=torch.device("cuda", device))
            update_parameters.append({'params': d.parameters(), "lr": cfgs['discriminator_cfg']['d_lr']})

    # TODO
    # 断点继续训练
    if args.resume:
        # Load checkpoint.
        print('==> ==> Resuming student from checkpoint..')
        msg_mgr.log_info('==> ==> Building model..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/%s-generator/ckpt.t7' % "_".join(args.teachers))
        student.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']

    # ================= Loss Function  ================ #
    # for Generator
    loss = lossmap.get_loss(cfgs['loss_map']['loss'])
    # loss between student and teacher
    criterion = lossmap.betweenLoss(cfgs['loss_map']['gamma'], loss=loss)

    # for Discriminator
    if cfgs['discriminator_cfg']['adv']:
        discriminators_criterion = lossmap.discriminatorLoss(discriminators, cfgs['loss_map']['eta'])
    else:
        discriminators_criterion = lossmap.discriminatorFakeLoss() # FakeLoss


    # ================= Optimizer Setup ================ #
    if args.student == "densenet_cifar":
        optimizer = optim.SGD(update_parameters, lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150 * min(2, len(teachers)), 250 * min(2, (len(teachers)))],gamma=0.1)
        print("nesterov = True")
    elif args.student == "mobilenet":
        optimizer = optim.SGD(update_parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)  # nesterov = True, weight_decay = 1e-4，stage = 3, batch_size = 64
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150 * min(1, len(teachers)), 250 * min(1, (len(teachers)))],gamma=0.1)
    else:
        optimizer = optim.SGD(update_parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)  # nesterov = True, weight_decay = 1e-4，stage = 3, batch_size = 64
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150 * min(1, len(teachers)), 250 * min(1, (len(teachers)))],gamma=0.1)


    # ================= Training and Testing ================ #
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        msg_mgr.log_info('==> ==> Building model..')
        scheduler.step()
        student.train()
        train_loss = 0
        correct = 0
        total = 0
        discriminator_loss = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            total += targets.size(0)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Get output from student model
            outputs = student(inputs)
            # Get teacher model
            teacher = teachers_student.selector_teacher(teachers)
            # Get output from teacher model
            answers = teacher(inputs)
            # Select output from student and teacher
            outputs, answers = teachers_student.selector_output(outputs, answers, eval(args.out_layer))
            # Calculate loss between student and teacher
            loss = criterion(outputs, answers)
            # Calculate loss for discriminators
            d_loss = discriminators_criterion(outputs, answers)
            
            # Get total loss
            total_loss = loss + d_loss

            total_loss.backward()
            optimizer.step()

            train_loss += loss.item()
            discriminator_loss += d_loss.item()
            
            _, predicted = outputs[-1].max(1)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Teacher: %s | Lr: %.4e | G_Loss: %.3f | D_Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (teacher.__name__, scheduler.get_lr()[0], train_loss / (batch_idx + 1), discriminator_loss / (batch_idx + 1), 100. * correct / total, correct, total))


    def test(epoch):
        global best_acc
        student.eval()
        test_loss = 0
        correct = 0
        total = 0
        discriminator_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                total += targets.size(0)
                inputs, targets = inputs.to(device), targets.to(device)

                # Get output from student model
                outputs = student(inputs)
                # Get teacher model
                teacher = teachers_student.selector_teacher(teachers)
                # Get output from teacher model
                answers = teacher(inputs)
                # Select output from student and teacher
                outputs, answers = teachers_student.selector_output(outputs, answers, eval(args.out_layer))
                # Calculate loss between student and teacher
                loss = criterion(outputs, answers)
                # Calculate loss for discriminators
                d_loss = discriminators_criterion(outputs, answers)

                test_loss += loss.item()
                discriminator_loss += d_loss.item()
                _, predicted = outputs[-1].max(1)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Lr: %.4e | G_Loss: %.3f | D_Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (scheduler.get_lr()[0], test_loss / (batch_idx + 1), discriminator_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            best_acc = max(100. * correct / total, best_acc)

        # Save checkpoint (the best accuracy).
        if epoch % 10 == 0 and best_acc == (100. * correct / total):
            print('Saving..')
            state = {
                'net': student.state_dict(),
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            FILE_PATH = './checkpoint' + '/' + "_".join(args.teachers) + '-generator'
            if os.path.isdir(FILE_PATH):
                # print 'dir exists'generator
                pass
            else:
                # print 'dir not exists'
                os.mkdir(FILE_PATH)
            save_name = './checkpoint' + '/' + "_".join(args.teachers) + '-generator/ckpt.t7'
            torch.save(state, save_name)


    for epoch in range(start_epoch, start_epoch + args.epochs*(len(teachers))):
        train(epoch)
        test(epoch)