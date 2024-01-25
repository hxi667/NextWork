import torch
import os.path as osp
import torch.optim as optim
from utils import get_msg_mgr
from utils import get_valid_args, get_attr_from, mkdir


def get_optimizer(update_parameters, optimizer_cfg):
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info(optimizer_cfg)
    # 从 optim 包中获取名为 optimizer_cfg['solver'] 的 Optimizer 类 
    optimizer = get_attr_from([optim], optimizer_cfg['solver'])
    valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
    # 实例化这个 Optimizer 类
    optimizer = optimizer(
        update_parameters, **valid_arg)
        # filter(lambda p: p.requires_grad, model.parameters()), **valid_arg)       
    return optimizer
    

def get_scheduler(optimizer, scheduler_cfg):
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info(scheduler_cfg)
    # 从 optim.lr_scheduler 包中获取名为 scheduler_cfg['scheduler'] 的 Scheduler 类 
    Scheduler = get_attr_from(
        [optim.lr_scheduler], scheduler_cfg['scheduler'])
    valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['scheduler'])
    # 实例化这个 Scheduler 类
    scheduler = Scheduler(optimizer, **valid_arg)
    return scheduler


# 保存 checkpoint
def save_ckpt(engine_cfg, save_path, model, optimizer, scheduler, iteration):
    if torch.distributed.get_rank() == 0:
        mkdir(osp.join(save_path, "checkpoints/"))
        save_name = engine_cfg['save_name']
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'iteration': iteration}
        torch.save(checkpoint,
                    osp.join(save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, iteration)))


#  加载 checkpoint
def load_ckpt(engine_cfg, save_name, device, model, optimizer, scheduler, training):
    msg_mgr = get_msg_mgr()

    load_ckpt_strict = engine_cfg['restore_ckpt_strict'] # boolean

    checkpoint = torch.load(save_name, map_location=torch.device(
        "cuda", device))
    model_state_dict = checkpoint['model']

    # 如果 not load_ckpt_strict 为 True，即不是严格检查checkpoint是否与定义的模型相同，
    # 则找到两个模型状态字典中共有的参数键，并将它们按照某个规则排序后打印输出
    if not load_ckpt_strict:
        msg_mgr.log_info("-------- Restored Params List --------")
        msg_mgr.log_info(sorted(set(model_state_dict.keys()).intersection(
            set(model.state_dict().keys()))))

    model.load_state_dict(model_state_dict, strict=load_ckpt_strict)
    
    if training:
        if not engine_cfg["optimizer_reset"] and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            msg_mgr.log_warning(
                "Restore NO Optimizer from %s !!!" % save_name)
        if not engine_cfg["scheduler_reset"] and 'scheduler' in checkpoint:
            scheduler.load_state_dict(
                checkpoint['scheduler'])
        else:
            msg_mgr.log_warning(
                "Restore NO Scheduler from %s !!!" % save_name)
    msg_mgr.log_info("Restore Parameters from %s !!!" % save_name)


# 从 checkpoint 恢复
def resume_ckpt(cfgs, save_path, device, model, optimizer, scheduler, training):
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    if engine_cfg is None:
        raise Exception("Initialize a model without -Engine-Cfgs-")
    
    if isinstance(engine_cfg['restore_hint'], int):
        student_name = cfgs['model_cfg']['student']
        save_name = osp.join(
            save_path, 'checkpoints/{}-{:0>5}.pt'.format(student_name, engine_cfg['restore_hint']))
        iteration = engine_cfg['restore_hint']
    # 也可以直接指定 checkpoint 文件的目录
    elif isinstance(engine_cfg['restore_hint'], str):
        save_name = engine_cfg['restore_hint']
        iteration = 0
    else:
        raise ValueError(
            "Error type for -Restore_Hint-, supported: int or string.")
    load_ckpt(engine_cfg, save_name, device, model, optimizer, scheduler, training)
    return iteration