import os.path as osp
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import torch.utils.data as tordata
import torch.optim as optim

from utils.msg_manager import get_msg_mgr
from utils.common import get_valid_args, get_attr_from, mkdir, get_ddp_module, params_count


from modeling.teachers_student import get_teachers_student
from modeling.loss_aggregator import LossAggregator
from modeling.losses import get_loss
from modeling.losses.discriminatorloss import discriminatorLoss, discriminatorFakeLoss
from modeling.losses.betweenloss import betweenLoss
from modeling.discriminator import Discriminators

from data.transform import get_transform
from data.collate_fn import CollateFn
from data.dataset import DataSet
import data.sampler as Samplers


class BuildModel():
    def __init__(self, cfgs, training):
        self.cfgs = cfgs
        self.iteration = 0
        
        self.msg_mgr = get_msg_mgr()

        self.engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
        if self.engine_cfg is None:
            raise Exception("Initialize a model without -Engine-Cfgs-")

        # ================= Data Loader ================ #
        self.msg_mgr.log_info('==> ==> Preparing Data..')
        self.msg_mgr.log_info(cfgs['data_cfg'])

        if training:
            self.train_transform = get_transform(cfgs['trainer_cfg']['transform'])
            self.train_loader = self.get_loader(cfgs['data_cfg'], train=True)
        if not training or self.engine_cfg['with_test']:
            self.evaluator_transform = get_transform(cfgs['evaluator_cfg']['transform'])    
            self.test_loader = self.get_loader(cfgs['data_cfg'], train=False)

        # ================= Model Setup ================ #
        self.msg_mgr.log_info('==> ==> Building Model..')

        if training and self.engine_cfg['enable_float16']:
            # float16
            self.Scaler = GradScaler()
        self.save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],
                                  cfgs['model_cfg']['student'], self.engine_cfg['save_name'])
        
        # Cuda setup
        self.device = torch.distributed.get_rank()
        torch.cuda.set_device(self.device)
        
        # get models as teachers and students
        self.teachers, self.student = get_teachers_student(cfgs['model_cfg'], cfgs['data_cfg']['dataset_name'], self.device)

        self.msg_mgr.log_info("Teacher(s): ")
        self.msg_mgr.log_info([teacher.__name__ for teacher in self.teachers])
        self.msg_mgr.log_info("Student: ")
        self.msg_mgr.log_info([self.student.__name__])

        
        # TODO
        dims = [10]
        # dims = [student.out_dims[i] for i in eval(args.out_layer)]
        self.msg_mgr.log_info(["student dims: ", dims])

        self.update_parameters = [{'params': self.student.parameters()}]

        # discriminator
        if cfgs['model_cfg']['discriminator']['adv']:
            self.discriminators = Discriminators(dims, grl=cfgs['model_cfg']['discriminator']['grl'])
            for d in self.discriminators.discriminators:
                d = d.to(device=torch.device("cuda", self.device))
                self.update_parameters.append({'params': d.parameters(), "lr": cfgs['model_cfg']['discriminator']['d_lr']})
        
        # init parameters
        self.init_parameters(self.student)
        for teacher in self.teachers:
            self.init_parameters(teacher) 
        for discriminator in self.discriminators.discriminators:
            self.init_parameters(discriminator) 

        # count teachers and  discriminators parameters
        for teacher in self.teachers:
            self.msg_mgr.log_info(params_count(teacher))
        for discriminator in self.discriminators.discriminators:
            self.msg_mgr.log_info(params_count(discriminator))

        # ================= Loss Function Setup ================ #
        if training:
            # for student
            # 如果有多个 losses， 返回由多个损失组成的 ModuleDict
            self.loss_aggregator = LossAggregator(cfgs['loss_cfg'])
            
            # for Generator
            loss_generator = get_loss(cfgs['loss_map']['loss'])
            # loss between student and teacher
            self.criterion = betweenLoss(cfgs['loss_map']['gamma'], loss=loss_generator)

            # for Discriminator
            if cfgs['model_cfg']['discriminator']['adv']:
                self.discriminators_criterion = discriminatorLoss(self.discriminators, cfgs['loss_map']['eta'])
            else:
                self.discriminators_criterion = discriminatorFakeLoss() # FakeLoss
            
            # ================= Optimizer Setup ================ #
            self.optimizer = self.get_optimizer(self.update_parameters, cfgs['optimizer_cfg'])
            self.scheduler = self.get_scheduler(cfgs['scheduler_cfg'])


        # if "training" == true, training model, if not, evaluation model
        self.student.train(training)
        for teacher in self.teachers:
            teacher.train(training)
        for discriminator in self.discriminators.discriminators:
            discriminator.train(training)

        restore_hint = self.engine_cfg['restore_hint']
        if restore_hint != 0:
            # 从 checkpoint 恢复
            self.resume_ckpt(restore_hint, self.student)
        

        # 如果为 True, 应用 Batch Normalization synchronously， 通常在分布式训练中使用，以确保不同 GPU 上的批量归一化参数同步
        if training and cfgs['trainer_cfg']['sync_BN']:
            self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
            for teacher in self.teachers:
                teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
            for discriminator in self.discriminators.discriminators:
                discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
        
        # if True, fix BatchNorm layer weights
        if cfgs['trainer_cfg']['fix_BN']:
             self.fix_BN(self.student)
             for teacher in self.teachers:
                self.fix_BN(teacher)
             for discriminator in self.discriminators.discriminators:
                self.fix_BN(discriminator)

        # 经过分布式数据并行 (DDP) 化处理的model
        find_unused_parameters = cfgs['trainer_cfg']['find_unused_parameters']  
        self.student = get_ddp_module(self.student, find_unused_parameters)
        # for teacher in self.teachers:
        #     teacher = get_ddp_module(teacher, find_unused_parameters)
        for discriminator in self.discriminators.discriminators:
            discriminator = get_ddp_module(discriminator, find_unused_parameters)


        self.msg_mgr.log_info("Build Model Finished!")


    def init_parameters(self, model):
        for m in model.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    # get data loader
    def get_loader(self, data_cfg, train=True):
        sampler_cfg = self.cfgs['trainer_cfg']['sampler'] if train else self.cfgs['evaluator_cfg']['sampler']
        # DataSet
        dataset = DataSet(data_cfg, train)
        
        # 从 Samplers 包中获取名为 sampler_cfg['type'] 的 Samplers 类 
        Sampler = get_attr_from([Samplers], sampler_cfg['type'])
        vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=[
            'sample_type', 'type'])
        # 实例化这个 Sampler 类
        sampler = Sampler(dataset, **vaild_args)

        loader = tordata.DataLoader(
            dataset=dataset,
            batch_sampler=sampler, # batch_sampler: 每次返回一个批次的索引
            collate_fn=CollateFn(dataset.label_set, sampler_cfg),
            num_workers=data_cfg['num_workers'])
        return loader
    
    # get optimizer
    def get_optimizer(self, update_parameters, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        # 从 optim 包中获取名为 optimizer_cfg['solver'] 的 Optimizer 类 
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        # 实例化这个 Optimizer 类
        optimizer = optimizer(
            update_parameters, **valid_arg)
            # filter(lambda p: p.requires_grad, self.parameters()), **valid_arg)
        return optimizer
    
    # get scheduler
    def get_scheduler(self, scheduler_cfg):
        self.msg_mgr.log_info(scheduler_cfg)
        # 从 optim.lr_scheduler 包中获取名为 scheduler_cfg['scheduler'] 的 Scheduler 类 
        Scheduler = get_attr_from(
            [optim.lr_scheduler], scheduler_cfg['scheduler'])
        valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['scheduler'])
        # 实例化这个 Scheduler 类
        scheduler = Scheduler(self.optimizer, **valid_arg)
        return scheduler
    
    # save checkpoint
    def save_ckpt(self, model, iteration):
        if torch.distributed.get_rank() == 0:
            mkdir(osp.join(self.save_path, "checkpoints/"))
            save_name = self.engine_cfg['save_name']
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'iteration': iteration}
            torch.save(checkpoint,
                        osp.join(self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, iteration)))

    # load checkpoint
    def _load_ckpt(self, save_name, model):

        load_ckpt_strict = self.engine_cfg['restore_ckpt_strict'] # boolean

        checkpoint = torch.load(save_name, map_location=torch.device(
            "cuda", self.device))
        model_state_dict = checkpoint['model']

        # 如果 not load_ckpt_strict 为 True，即不是严格检查checkpoint是否与定义的模型相同，
        # 则找到两个模型状态字典中共有的参数键，并将它们按照某个规则排序后打印输出
        if not load_ckpt_strict:
            self.msg_mgr.log_info("-------- Restored Params List --------")
            self.msg_mgr.log_info(sorted(set(model_state_dict.keys()).intersection(
                set(model.state_dict().keys()))))

        model.load_state_dict(model_state_dict, strict=load_ckpt_strict)
        
        if self.training:
            if not self.engine_cfg["optimizer_reset"] and 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Optimizer from %s !!!" % save_name)
            if not self.engine_cfg["scheduler_reset"] and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(
                    checkpoint['scheduler'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Scheduler from %s !!!" % save_name)
        self.msg_mgr.log_info("Restore Parameters from %s !!!" % save_name)

    # resume checkpoint
    def resume_ckpt(self, restore_hint, model):
        
        if isinstance(restore_hint, int):
            save_name = self.engine_cfg['save_name']
            save_name = osp.join(
                self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, restore_hint))
            self.iteration = restore_hint
        # 也可以直接指定 checkpoint 文件的目录
        elif isinstance(restore_hint, str):
            save_name = restore_hint
            self.iteration = 0
        else:
            raise ValueError(
                "Error type for -Restore_Hint-, supported: int or string.")
        self._load_ckpt(save_name, save_name, model)

    # fix BatchNorm layer weights
    def fix_BN(self, model):
        for module in model.modules():
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                module.eval()
    @ staticmethod
    def run_train(model):
        return
    @ staticmethod
    def run_test(model):
        return