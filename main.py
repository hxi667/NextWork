from __future__ import print_function

import yaml
import os
import argparse
import os.path as osp
import torch

# from modeling import *
from modeling import discriminator, teachers_student, model_utils, build_models
from modeling.losses import lossmap
from modeling.loss_aggregator import LossAggregator
from modeling.model_utils import resume_ckpt

from data.transform import get_transform
from data.dataloader import get_loader
from utils.common import progress_bar, init_seeds
from utils.msg_manager import get_msg_mgr


# ================= Arugments ================ #
parser = argparse.ArgumentParser(description='Training Gait with PyTorch')
parser.add_argument('--local_rank', type=int, default=0, help="passed by torch.distributed.launch module")
parser.add_argument('--cfgs', type=str, default='./configs/default.yaml', help="path of config file")
parser.add_argument('--phase', default='train', choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true', help="log to file, default path is: output/<dataset>/<exp_name>/<Student_name>/<logs>/<Datetime>.txt")

#  ============================================================
parser.add_argument('--out_layer', default="[-1]", type=str, help='the type of pooling layer of output')  # eval()
# model config
parser.add_argument('--out_dims', default="[5000,1000,500,200,10]", type=str, help='the dims of output pooling layers')  # eval()
parser.add_argument('--fc_out', default=1, type=int, help='if immediate output from fc-layer')
parser.add_argument('--pool_out', default="max", type=str, help='the type of pooling layer of output')

args = parser.parse_args()


def build_model(cfgs, training):
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info('==> ==> Building Model..')

    # Cuda setup
    device = torch.distributed.get_rank()
    torch.cuda.set_device(device)

    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    if engine_cfg is None:
        raise Exception("Initialize a model without -Engine-Cfgs-")
    save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],
                                  cfgs['model_cfg']['student'], engine_cfg['exp_name'])
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

    update_parameters = [{'params': student.parameters()}]

    # discriminator
    if cfgs['model_cfg']['discriminator']['adv']:
        discriminators = discriminator.Discriminators(dims, grl=cfgs['model_cfg']['discriminator']['grl'])
        for d in discriminators.discriminators:
            d = d.to(device=torch.device("cuda", device))
            update_parameters.append({'params': d.parameters(), "lr": cfgs['model_cfg']['discriminator']['d_lr']})

    if training:
        # ================= Loss Function  ================ #
        # for student
        # 如果有多个 losses， 返回由多个损失组成的 ModuleDict
        loss_aggregator = LossAggregator(cfgs['loss_cfg'])
        
        # for Generator
        loss = lossmap.get_loss(cfgs['loss_map']['loss'])
        # loss between student and teacher
        criterion = lossmap.betweenLoss(cfgs['loss_map']['gamma'], loss=loss)

        # for Discriminator
        if cfgs['model_cfg']['discriminator']['adv']:
            discriminators_criterion = lossmap.discriminatorLoss(discriminators, cfgs['loss_map']['eta'])
        else:
            discriminators_criterion = lossmap.discriminatorFakeLoss() # FakeLoss
        
        # ================= Optimizer Setup ================ #
        optimizer = model_utils.get_optimizer(update_parameters, cfgs['optimizer_cfg'])
        scheduler = model_utils.get_scheduler(optimizer, cfgs['scheduler_cfg'])

        if cfgs['train_cfg']['restore_hint'] != 0:
            # 从 checkpoint 恢复
            iteration = resume_ckpt(cfgs, save_path, device, student, optimizer, scheduler, training)
    
    else:
        if cfgs['evaluator_cfg']['restore_hint'] != 0:
            # 从 checkpoint 恢复
            iteration = resume_ckpt(cfgs, save_path, device, student, None, None, training)
    
        
 
def build_data(cfgs, training):
    
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info('==> ==> Preparing Data..')
    msg_mgr.log_info(cfgs['data_cfg'])

    if training:
        train_transform = get_transform(cfgs['trainer_cfg']['transform'])
        train_loader = get_loader(cfgs, train=True)
        
        if cfgs['trainer_cfg']['with_test']:
            test_transform = get_transform(cfgs['evaluator_cfg']['transform'])    
            test_loader = get_loader(cfgs, train=False)
            return [train_transform, train_loader, test_transform, test_loader]
        
        return [train_transform, train_loader]
    else:
        test_transform = get_transform(cfgs['evaluator_cfg']['transform'])    
        test_loader = get_loader(cfgs, train=False)
        return [test_transform, test_loader]
    

# init SummaryWriter and logger(to tensorboard, console and file print log info)， random seeds
def init(cfgs, training):
    # 获得 MessageManager 类的实例对象
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg'] 

    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'], engine_cfg['exp_name'], 'Student_'+ cfgs['model_cfg']['student'])
    if training:
        # 初始化 SummaryWriter 和 logger
        msg_mgr.init_manager(output_path, args.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        # 初始化 logger
        msg_mgr.init_logger(output_path, args.log_to_file)
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
    
    # ================= Load Config File ================ #
    with open(args.cfgs, 'r') as stream:
        cfgs = yaml.safe_load(stream)

    training = (args.phase == 'train')

    # ================= Init SummaryWriter, Logger and Random Seeds ================ #
    init(cfgs, training=training)

    # data_dic = build_data(cfgs, training)

    # build_model(cfgs, training)
    build_models.BuildModel(cfgs, training)