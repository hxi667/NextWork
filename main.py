from __future__ import print_function

import yaml
import os
import argparse
import torch
import torch.nn as nn
import datetime
from modeling.build_models import BuildModel

from utils.common import init_seeds, get_ddp_module
from utils.msg_manager import get_msg_mgr


# ================= Arugments ================ #
parser = argparse.ArgumentParser(description='Training Gait with PyTorch')
# parser.add_argument('--local_rank', type=int, default=0, help="passed by torch.distributed.launch module")
parser.add_argument('--cfgs', type=str, default='./configs/default.yaml', help="path of config file")
parser.add_argument('--phase', default='train', choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true', help="log to file, default path is: output/<dataset>/<Student_name>/<save_name>/<logs>/<Datetime>.txt")

args = parser.parse_args()



# init SummaryWriter and logger(to tensorboard, console and file print log info)， random seeds
def init(cfgs, training):
    # get MessageManager class的实例对象
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg'] 

    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'], 'Student_'+ cfgs['model_cfg']['student'], engine_cfg['save_name'])
    if training:
        # init SummaryWriter 和 logger
        msg_mgr.init_manager(output_path, args.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        # init logger
        msg_mgr.init_logger(output_path, args.log_to_file)
    # write trainer or evaluator config info to console/file 
    msg_mgr.log_info(engine_cfg)

    seed = torch.distributed.get_rank()
    # init seeds
    init_seeds(seed)


def run_model(cfgs, training):

    model = BuildModel(cfgs, training)
    
    # if True, 应用 Batch Normalization synchronously， 通常在分布式训练中使用，以确保不同 GPU 上的批量归一化参数同步
    if training and cfgs['trainer_cfg']['sync_BN']:
        model.student = nn.SyncBatchNorm.convert_sync_batchnorm(model.student)
        # for teacher in model.teachers:
        #     teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        for discriminator in model.discriminators.discriminators:
            discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
    
    # if True, fix BatchNorm layer weights
    if cfgs['trainer_cfg']['fix_BN']:
        model.fix_BN(model.student)
        for teacher in model.teachers:
            model.fix_BN(teacher)
        for discriminator in model.discriminators.discriminators:
            model.fix_BN(discriminator)

    # 返回一个经过分布式数据并行 (DDP) 化处理的model   
    find_unused_parameters = cfgs['trainer_cfg']['find_unused_parameters']
    model.student = get_ddp_module(model.student, find_unused_parameters)
    # DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient.
    # for teacher in model.teachers:
    #     teacher = get_ddp_module(teacher, find_unused_parameters)
    for discriminator in model.discriminators.discriminators:
        discriminator = get_ddp_module(discriminator, find_unused_parameters)
    
    # training or testing
    if training:
        BuildModel.run_train(model)
    else:
        BuildModel.run_test(model)



if __name__ == '__main__':
    torch.distributed.init_process_group('nccl', init_method='env://', timeout=datetime.timedelta(seconds=60))
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of available GPUs({}) equals to the world size({}).".format(
            torch.cuda.device_count(), torch.distributed.get_world_size()))
    
    # Load Config File
    with open(args.cfgs, 'r') as stream:
        cfgs = yaml.safe_load(stream)

    training = (args.phase == 'train')

    # Init SummaryWriter, Logger and Random Seeds 
    init(cfgs, training)

    # Run model
    run_model(cfgs, training)
    