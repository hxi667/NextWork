from __future__ import print_function

import yaml
import os
import argparse
import torch

from modeling.build_models import BuildModel

from utils.common import init_seeds
from utils.msg_manager import get_msg_mgr


# ================= Arugments ================ #
parser = argparse.ArgumentParser(description='Training Gait with PyTorch')
parser.add_argument('--local_rank', type=int, default=0, help="passed by torch.distributed.launch module")
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
    # 写 trainer 或 evaluator 的配置信息到 console/file 
    msg_mgr.log_info(engine_cfg)

    seed = torch.distributed.get_rank()
    # init seeds
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
    init(cfgs, training)

    
    build_models = BuildModel(cfgs, training)
    
    if training:
        build_models.run_train()
    else:
        build_models.run_test()
    