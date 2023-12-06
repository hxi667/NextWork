"""The base model definition.

This module defines the abstract meta model class and base model class. In the base model,
 we define the basic model functions, like get_loader, build_network, and run_train, etc.
 The api of the base model is run_train and run_test, they are used in `opengait/main.py`.

Typical usage:

BaseModel.run_train(model)
BaseModel.run_test(model)
"""
import torch
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata

from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from abc import ABCMeta
from abc import abstractmethod

from . import backbones
from .loss_aggregator import LossAggregator
from data.transform import get_transform
from data.collate_fn import CollateFn
from data.dataset import DataSet
import data.sampler as Samplers
from utils import Odict, mkdir, ddp_all_gather
from utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
# from evaluation import evaluator as eval_functions
from utils import NoOp
from utils import get_msg_mgr

# __all__ = ['BaseModel']


class MetaModel(metaclass=ABCMeta):
    """The necessary functions for the base model.

    This class defines the necessary functions for the base model, in the base model, we have implemented them.
    """
    @abstractmethod
    def get_loader(self, data_cfg):
        """Based on the given data_cfg, we get the data loader."""
        raise NotImplementedError

    @abstractmethod
    def build_network(self, model_cfg):
        """Build your network here."""
        raise NotImplementedError

    @abstractmethod
    def init_parameters(self):
        """Initialize the parameters of your network."""
        raise NotImplementedError

    @abstractmethod
    def get_optimizer(self, optimizer_cfg):
        """Based on the given optimizer_cfg, we get the optimizer."""
        raise NotImplementedError

    @abstractmethod
    def get_scheduler(self, scheduler_cfg):
        """Based on the given scheduler_cfg, we get the scheduler."""
        raise NotImplementedError

    @abstractmethod
    def save_ckpt(self, iteration):
        """Save the checkpoint, including model parameter, optimizer and scheduler."""
        raise NotImplementedError

    @abstractmethod
    def resume_ckpt(self, restore_hint):
        """Resume the model from the checkpoint, including model parameter, optimizer and scheduler."""
        raise NotImplementedError

    @abstractmethod
    def inputs_pretreament(self, inputs):
        """Transform the input data based on transform setting."""
        raise NotImplementedError

    @abstractmethod
    def train_step(self, loss_num) -> bool:
        """Do one training step."""
        raise NotImplementedError

    @abstractmethod
    def inference(self):
        """Do inference (calculate features.)."""
        raise NotImplementedError

    @abstractmethod
    def run_train(model):
        """Run a whole train schedule."""
        raise NotImplementedError

    @abstractmethod
    def run_test(model):
        """Run a whole test schedule."""
        raise NotImplementedError


class BaseModel(MetaModel, nn.Module):
    """Base model.

    This class inherites the MetaModel class, and implements the basic model functions, like get_loader, build_network, etc.

    Attributes:
        msg_mgr: the massage manager.
        cfgs: the configs.
        iteration: the current iteration of the model.
        engine_cfg: the configs of the engine(train or test).
        save_path: the path to save the checkpoints.

    """

    def __init__(self, cfgs, model_cfg):
        """Initialize the base model.

        Complete the model initialization, including the data loader, the network, the optimizer, the scheduler, the loss.
        完成模型初始化, 包括data loader、network、optimizer、scheduler 和 loss
        Args:
        cfgs:
            All of the configs.
        training:
            Whether the model is in training mode.
        """

        super(BaseModel, self).__init__()
        # 获得 MessageManager 类的实例对象
        self.msg_mgr = get_msg_mgr()
        self.cfgs = cfgs
        self.iteration = 0
        self.engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
        if self.engine_cfg is None:
            raise Exception("Initialize a model without -Engine-Cfgs-")

        if training and self.engine_cfg['enable_float16']:
            # float16, 梯度缩放，加快训练速度
            self.Scaler = GradScaler()
        self.save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],
                                  cfgs['model_cfg']['model'], self.engine_cfg['save_name'])
        
        # 如果子类实现了这个方法， 就调用子类的 "build_network" 方法
        # 否则调用父类的 "build_network" 方法
        self.build_network(cfgs['model_cfg'])
        self.init_parameters()
        
        # trainer transform 实例
        self.trainer_trfs = get_transform(cfgs['trainer_cfg']['transform'])
        # 写 dataset 配置信息到 console/file
        self.msg_mgr.log_info(cfgs['data_cfg'])
        
        # loader data
        if training:
            self.train_loader = self.get_loader(
                cfgs['data_cfg'], train=True)
        if not training or self.engine_cfg['with_test']:
            self.test_loader = self.get_loader(
                cfgs['data_cfg'], train=False)
            # evaluator transform 实例
            self.evaluator_trfs = get_transform(
                cfgs['evaluator_cfg']['transform'])

        self.device = torch.distributed.get_rank()
        torch.cuda.set_device(self.device)
        self.to(device=torch.device(
            "cuda", self.device))
        
        # 设置 Loss 函数, optimizer, scheduler
        if training:
            # 如果有多个 losses， 返回由多个损失组成的 ModuleDict
            self.loss_aggregator = LossAggregator(cfgs['loss_cfg'])
            self.optimizer = self.get_optimizer(self.cfgs['optimizer_cfg'])
            self.scheduler = self.get_scheduler(cfgs['scheduler_cfg'])
        
        # 如果 "training" 为 true, 则为 training 模式, 否则 evaluation 模式
        self.train(training)
        
        restore_hint = self.engine_cfg['restore_hint']
        if restore_hint != 0:
            # 从 checkpoint 恢复
            self.resume_ckpt(restore_hint)
    
    # 获取 backbone model
    def get_backbone(self, backbone_cfg):
        """Get the backbone of the model."""
        # dict 类型
        if is_dict(backbone_cfg):
            # 从 backbones 包中获取名为 backbone_cfg['type'] 的 Backbone 类 
            Backbone = get_attr_from([backbones], backbone_cfg['type'])
            # 验证 Backbone 类的参数
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            # 返回此 Backbone 类实例 
            return Backbone(**valid_args)
        # list 类型
        if is_list(backbone_cfg):
            # 递归调用 get_backbone(), 并将结果封装成一个 nn.ModuleList
            Backbone = nn.ModuleList([self.get_backbone(cfg)
                                      for cfg in backbone_cfg])
            return Backbone
        raise ValueError(
            "Error type for -Backbone-Cfg-, supported: (A list of) dict.")
    
    # 如果子类没有实现这个方法, 调用 backbone model.
    def build_network(self, model_cfg):
        if 'backbone_cfg' in model_cfg.keys():
            self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
    
    # 初始化模型参数
    def init_parameters(self):
        for m in self.modules():
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

    # 获取 data loader
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

    #  获取 optimizer
    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        # 从 optim 包中获取名为 optimizer_cfg['solver'] 的 Optimizer 类 
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        # 实例化这个 Optimizer 类
        optimizer = optimizer(
            filter(lambda p: p.requires_grad, self.parameters()), **valid_arg)
        return optimizer
    
    #  获取 scheduler
    def get_scheduler(self, scheduler_cfg):
        self.msg_mgr.log_info(scheduler_cfg)
        # 从 optim.lr_scheduler 包中获取名为 scheduler_cfg['scheduler'] 的 Scheduler 类 
        Scheduler = get_attr_from(
            [optim.lr_scheduler], scheduler_cfg['scheduler'])
        valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['scheduler'])
        # 实例化这个 Scheduler 类
        scheduler = Scheduler(self.optimizer, **valid_arg)
        return scheduler

    # 保存 checkpoint
    def save_ckpt(self, iteration):
        if torch.distributed.get_rank() == 0:
            mkdir(osp.join(self.save_path, "checkpoints/"))
            save_name = self.engine_cfg['save_name']
            checkpoint = {
                'model': self.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'iteration': iteration}
            torch.save(checkpoint,
                       osp.join(self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, iteration)))

    #  加载 checkpoint
    def _load_ckpt(self, save_name):
        load_ckpt_strict = self.engine_cfg['restore_ckpt_strict'] # boolean

        checkpoint = torch.load(save_name, map_location=torch.device(
            "cuda", self.device))
        model_state_dict = checkpoint['model']

        # 如果为 True，则检查checkpoint是否与定义的模型相同
        if not load_ckpt_strict:
            self.msg_mgr.log_info("-------- Restored Params List --------")
            self.msg_mgr.log_info(sorted(set(model_state_dict.keys()).intersection(
                set(self.state_dict().keys()))))

        self.load_state_dict(model_state_dict, strict=load_ckpt_strict)
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

    # 从 checkpoint 恢复
    def resume_ckpt(self, restore_hint):
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
        self._load_ckpt(save_name)

    # 固定所有 `BatchNorm` 层的权重
    def fix_BN(self):
        for module in self.modules():
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                module.eval()

    # 对输入数据 inputs 进行 transforms
    def inputs_pretreament(self, inputs):
        """Conduct transforms on input data.

        Args:
            inputs: the input data.
        Returns:
            tuple: training data including inputs, labels, and some meta data.
        """

        # seqs_batch -> [[No.1(seqL_batch, 3, 128, 128), No.2(seqL_batch, 3, 128, 128), ..., No.batchsize(seqL_batch, 3, 128, 128)], [No.1(seqL_batch, 128, 128), No.2(seqL_batch, 128, 128), ..., No.batchsize(seqL_batch, 128, 128)], ...]
        # labs_batch -> [1, ...]
        # typs_batch -> ['bg-01', ...]
        # vies_batch -> ['000', ...]
        # seqL_batch -> 如果为 fixed -> 则为 None, 如果为 unfixed -> np.asarray: seq length of each batch 
        seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
        
        seq_trfs = self.trainer_trfs if self.training else self.evaluator_trfs
        if len(seqs_batch) != len(seq_trfs):
            raise ValueError(
                "The number of types of input data and transform should be same. But got {} and {}".format(len(seqs_batch), len(seq_trfs)))
        
        requires_grad = bool(self.training)
        
        # 一个 transform 实例（single 或 compose ）对应一个序列类型
        # seqs -> [torch.Size([batch, seqL, 3, 128, 128]), torch.Size([batch, seqL, 128, 128]), ...]
        seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                for trf, seq in zip(seq_trfs, seqs_batch)]

        typs = typs_batch
        vies = vies_batch
        labs = list2var(labs_batch).long()

        if seqL_batch is not None:
            seqL_batch = np2var(seqL_batch).int()
        seqL = seqL_batch

        if seqL is not None:
            seqL_sum = int(seqL.sum().data.cpu().numpy())
            ipts = [_[:, :seqL_sum] for _ in seqs]
        else:
            ipts = seqs
        del seqs
        # ipts -> [torch.Size([batch, seqL, 3, 128, 128]), torch.Size([batch, seqL, 128, 128]), ...]
        # labs -> tensor([1, ...]
        # typs -> ['bg-01', ...]
        # vies -> ['000', ...]
        # seqL -> 如果为 fixed -> 则为 None, 如果为 unfixed -> np.asarray: seq length of each batch 
        return ipts, labs, typs, vies, seqL

    # 进行 loss_sum.backward(), self.optimizer.step() and self.scheduler.step()
    def train_step(self, loss_sum) -> bool:
        """Conduct loss_sum.backward(), self.optimizer.step() and self.scheduler.step().

        Args:
            loss_sum:The loss of the current batch.
        Returns:
            bool: True if the training is finished, False otherwise.
        """

        self.optimizer.zero_grad()
        if loss_sum <= 1e-9:
            self.msg_mgr.log_warning(
                "Find the loss sum less than 1e-9 but the training process will continue!")

        # float16, 梯度缩放，加快训练速度
        if self.engine_cfg['enable_float16']:
            self.Scaler.scale(loss_sum).backward()
            self.Scaler.step(self.optimizer)
            scale = self.Scaler.get_scale()
            self.Scaler.update()
            # Warning caused by optimizer skip when NaN
            # 当出现 NaN 时，optimizer 跳过引起的警告
            # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/5
            if scale != self.Scaler.get_scale():
                self.msg_mgr.log_debug("Training step skip. Expected the former scale equals to the present, got {} and {}".format(
                    scale, self.Scaler.get_scale()))
                return False
        else:
            loss_sum.backward()
            self.optimizer.step()

        self.iteration += 1
        self.scheduler.step()
        return True

    # model inference
    def inference(self, rank):
        """Inference all the test data.

        Args:
            rank: the rank of the current process.Transform
        Returns:
            Odict: contains the inference results.
        """
        total_size = len(self.test_loader)
        if rank == 0:
            pbar = tqdm(total=total_size, desc='Transforming')
        else:
            pbar = NoOp()
        batch_size = self.test_loader.batch_sampler.batch_size
        rest_size = total_size
        # info_dict: 推理的结果
        info_dict = Odict()
    
        for inputs in self.test_loader:
            # 对输入数据 inputs 进行 transforms
            ipts = self.inputs_pretreament(inputs)
            with autocast(enabled=self.engine_cfg['enable_float16']):
                # model 推理
                # 调用 model 子类的 "forward" 方法
                retval = self.forward(ipts)
                inference_feat = retval['inference_feat']
                for k, v in inference_feat.items():
                    # 分布式
                    inference_feat[k] = ddp_all_gather(v, requires_grad=False)
                del retval
            for k, v in inference_feat.items():
                inference_feat[k] = ts2np(v)
            info_dict.append(inference_feat)
            rest_size -= batch_size
            if rest_size >= 0:
                update_size = batch_size
            else:
                update_size = total_size % batch_size
            pbar.update(update_size)
        pbar.close()
        for k, v in info_dict.items():
            v = np.concatenate(v)[:total_size]
            info_dict[k] = v
        return info_dict

    @ staticmethod
    def run_train(model):
        """Accept the instance object(model) here, and then run the train loop."""

        # inputs[0] -> [[No.1(seqL_batch, 3, 128, 128), No.2(seqL_batch, 3, 128, 128), ..., No.batchsize(seqL_batch, 3, 128, 128)], [No.1(seqL_batch, 128, 128), No.2(seqL_batch, 128, 128), ..., No.batchsize(seqL_batch, 128, 128)], ...]
        # inputs[1] -> [1, ...]
        # inputs[2] -> ['bg-01', ...]
        # inputs[3] -> ['000', ...]
        # inputs[4] -> batch[4] ->->  如果为fixed -> 则为None, 如果为 unfixed -> np.asarray: seq length of each batch 
        for inputs in model.train_loader:
            
            # 对输入数据进行 transforms
            # len(ipts): 5
            ipts = model.inputs_pretreament(inputs)
            with autocast(enabled=model.engine_cfg['enable_float16']):
                # 运行 model
                retval = model(ipts)
                training_feat, visual_summary = retval['training_feat'], retval['visual_summary']
                del retval
            # 计算 loss
            loss_sum, loss_info = model.loss_aggregator(training_feat)
            ok = model.train_step(loss_sum)
            if not ok:
                # 跳出当前循环
                continue

            visual_summary.update(loss_info) # 更新 "loss_info" to "visual_summary" dict
            visual_summary['scalar/learning_rate'] = model.optimizer.param_groups[0]['lr']
            model.msg_mgr.train_step(loss_info, visual_summary)
            
            if model.iteration % model.engine_cfg['save_iter'] == 0:
                # 保存 checkpoint
                model.save_ckpt(model.iteration)

                # 如果 "with_test" 为 true， 运行 test 步骤
                if model.engine_cfg['with_test']:
                    model.msg_mgr.log_info("Running test...")
                    model.eval()
                    result_dict = BaseModel.run_test(model)
                    model.train()
                    if model.cfgs['trainer_cfg']['fix_BN']:
                        model.fix_BN()
                    if result_dict:
                        model.msg_mgr.write_to_tensorboard(result_dict)
                    model.msg_mgr.reset_time()
            if model.iteration >= model.engine_cfg['total_iter']:
                break

    @ staticmethod
    def run_test(model):
        """Accept the instance object(model) here, and then run the test loop."""

        rank = torch.distributed.get_rank()
        with torch.no_grad():
            info_dict = model.inference(rank)
        if rank == 0:
            loader = model.test_loader
            label_list = loader.dataset.label_list
            types_list = loader.dataset.types_list
            views_list = loader.dataset.views_list

            info_dict.update({
                'labels': label_list, 'types': types_list, 'views': views_list})

            if 'eval_func' in model.cfgs["evaluator_cfg"].keys():
                eval_func = model.cfgs['evaluator_cfg']["eval_func"]
            else:
                eval_func = 'identification'
            # 获取 evaluate 函数
            eval_func = getattr(eval_functions, eval_func)
            valid_args = get_valid_args(
                eval_func, model.cfgs["evaluator_cfg"], ['metric'])
            
            try:
                dataset_name = model.cfgs['data_cfg']['test_dataset_name']
            except:
                dataset_name = model.cfgs['data_cfg']['dataset_name']
            # 评估
            return eval_func(info_dict, dataset_name, **valid_args)