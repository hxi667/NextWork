import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import torch.utils.data as tordata
import torch.optim as optim
from torch.cuda.amp import autocast

from tqdm import tqdm

from utils.msg_manager import get_msg_mgr
from utils.common import NoOp, Odict, get_valid_args, get_attr_from, np2var, list2var, ts2np
from utils.common import params_count, mkdir, ddp_all_gather


from modeling.teachers_student import get_teachers_student, selector_teacher, selector_output
from modeling.loss_aggregator import LossAggregator
from modeling.losses import get_loss
from modeling.losses.discriminatorloss import discriminatorLoss, discriminatorFakeLoss
from modeling.losses.betweenloss import betweenLoss
from modeling.discriminator import Discriminators
from modeling.evaluation import evaluator as eval_functions

from data.transform import get_transform
from data.collate_fn import CollateFn
from data.dataset import DataSet
import data.sampler as Samplers


class BuildModel():
    def __init__(self, cfgs, training):
        self.cfgs = cfgs
        self.training = training
        self.iteration = 0
        
        self.msg_mgr = get_msg_mgr()

        if self.training:
            self.engine_cfg = cfgs['trainer_cfg']
            self.msg_mgr.log_info('==> ==> Initialize a model with -Trainer-Cfgs-')
        else:
            self.engine_cfg = cfgs['evaluator_cfg']
            self.msg_mgr.log_info('==> ==> Initialize a model with -Evaluator-Cfgs-')
        if self.engine_cfg is None:
            raise Exception("Initialize a model without -Engine-Cfgs-")

        # ================= Data Loader ================ #
        self.msg_mgr.log_info('==> ==> Preparing Data...')
        self.msg_mgr.log_info(cfgs['data_cfg'])

        if self.training:
            self.train_transform = get_transform(cfgs['trainer_cfg']['transform'])
            self.train_loader = self.get_loader(cfgs['data_cfg'], train=True)
        if not self.training or self.engine_cfg['with_test']:
            self.evaluator_transform = get_transform(cfgs['evaluator_cfg']['transform'])    
            self.test_loader = self.get_loader(cfgs['data_cfg'], train=False)

        # ================= Model Setup ================ #
        self.msg_mgr.log_info('==> ==> Building Model..')

        if self.training and self.engine_cfg['enable_float16']:
            # float16
            self.Scaler = GradScaler()
        self.save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],
                                  'Student_'+ cfgs['model_cfg']['student'], self.engine_cfg['save_name'])
        
        # Cuda setup
        self.device = torch.distributed.get_rank()
        torch.cuda.set_device(self.device)
        
        # get models as teachers and students
        self.teachers, self.student = get_teachers_student(cfgs['model_cfg'], cfgs['data_cfg']['dataset_name'], self.device)

        self.msg_mgr.log_info("Teacher(s): ")
        self.msg_mgr.log_info([teacher.__name__ for teacher in self.teachers])
        self.msg_mgr.log_info("Student: ")
        self.msg_mgr.log_info([self.student.__name__])

        # #
        dims = [self.student.model_cfg['out_dims'][i] for i in self.cfgs['model_cfg']['out_layer']]
        self.msg_mgr.log_info(["student dims: ", dims])

        self.update_parameters = [{'params': self.student.parameters()}]

        # get discriminator
        if cfgs['model_cfg']['discriminator']['adv']:
            self.discriminators = Discriminators(dims, grl=cfgs['model_cfg']['discriminator']['grl'])
            for d in self.discriminators.discriminators:
                d = d.to(device=torch.device("cuda", self.device))
                self.update_parameters.append({'params': d.parameters(), "lr": cfgs['model_cfg']['discriminator']['d_lr']})
        
        # init parameters
        self.init_parameters(self.student)
        for discriminator in self.discriminators.discriminators:
            self.init_parameters(discriminator) 

        restore_hint = self.engine_cfg['restore_hint']
        if restore_hint != 0:
            # resume checkpoint
            self.resume_ckpt(restore_hint, self.student)

        # count teachers and discriminators parameters
        for teacher in self.teachers:
            self.msg_mgr.log_info(params_count(teacher))
        for discriminator in self.discriminators.discriminators:
            self.msg_mgr.log_info(params_count(discriminator))

        # ================= Loss Function Setup ================ #
        if self.training:
            # for student
            # If there are multiple loss, return a ModuleDict consisting of multiple loss
            self.loss_aggregator = LossAggregator(cfgs['loss_cfg'])
            
            # for Generator
            loss_generator = get_loss(cfgs['loss_zoo']['loss'])
            # loss between student and teacher
            self.between_criterion = betweenLoss(cfgs['loss_zoo']['gamma'], loss=loss_generator)

            # for Discriminator
            if cfgs['model_cfg']['discriminator']['adv']:
                self.discriminators_criterion = discriminatorLoss(self.discriminators, cfgs['loss_zoo']['eta'], enable_float16=self.engine_cfg['enable_float16'])
            else:
                self.discriminators_criterion = discriminatorFakeLoss() # FakeLoss
            
            # ================= Optimizer Setup ================ #
            self.optimizer = self.get_optimizer(self.update_parameters, cfgs['optimizer_cfg'])
            self.scheduler = self.get_scheduler(cfgs['scheduler_cfg'])


        # if "training" == true, training model, if not, evaluation model
        self.student.train(self.training)
        for discriminator in self.discriminators.discriminators:
            discriminator.train(self.training)

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
        
        # Get the Samplers class named sampler_cfg['type'] from the Samplers package 
        Sampler = get_attr_from([Samplers], sampler_cfg['type'])
        vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=[
            'sample_type', 'type'])
        # Instantiate this Sampler class
        sampler = Sampler(dataset, **vaild_args)

        loader = tordata.DataLoader(
            dataset=dataset,
            batch_sampler=sampler, # batch_sampler: Returns the index of one batch at a time
            collate_fn=CollateFn(dataset.label_set, sampler_cfg),
            num_workers=data_cfg['num_workers'])
        return loader
    
    # get optimizer
    def get_optimizer(self, update_parameters, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        # Get the Optimizer class named optimizer_cfg['solver'] from the optim package 
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        # Instantiate this Optimizer class
        optimizer = optimizer(
            update_parameters, **valid_arg)
            # filter(lambda p: p.requires_grad, self.parameters()), **valid_arg)
        return optimizer
    
    # get scheduler
    def get_scheduler(self, scheduler_cfg):
        self.msg_mgr.log_info(scheduler_cfg)
        # Get the Scheduler class named scheduler_cfg['scheduler'] from the optim.lr_scheduler package 
        Scheduler = get_attr_from(
            [optim.lr_scheduler], scheduler_cfg['scheduler'])
        valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['scheduler'])
        # Instantiate this Scheduler class
        scheduler = Scheduler(self.optimizer, **valid_arg)
        return scheduler
    
    # save checkpoint
    def save_ckpt(self, model, iteration):
        if torch.distributed.get_rank() == 0:
            mkdir(osp.join(self.save_path, "checkpoints/"))
            save_name = self.engine_cfg['save_name']
            model_state_dict = model.module.state_dict() # if self.is_parallel else model.state_dict()
            checkpoint = {
                'model': model_state_dict,
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

        # if not load_ckpt_strict is True, not strictly checking that the checkpoint is the same as the module of the defined model,
        # Find the keys that are common to both model state dictionaries and print them out after ordering them 
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
        # You can also specify the directory of the checkpoint file directly
        elif isinstance(restore_hint, str):
            save_name = restore_hint
            self.iteration = 0
        else:
            raise ValueError(
                "Error type for -Restore_Hint-, supported: int or string.")
        self._load_ckpt(save_name, model)

    # fix BatchNorm layer weights
    def fix_BN(self, model):
        for module in model.modules():
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                module.eval()
    
    # transforms the input data
    def inputs_pretreament(self, inputs, training):
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
        
        seq_trfs = self.train_transform if training else self.evaluator_transform
        if len(seqs_batch) != len(seq_trfs):
            raise ValueError(
                "The number of types of input data and transform should be same. But got {} and {}".format(len(seqs_batch), len(seq_trfs)))
        
        requires_grad = bool(training)
        
        # A transform instance (single or compose) corresponds to a sequence type
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
        # seqL -> if fixed -> None, if unfixed -> np.asarray: seq length of each batch 
        return ipts, labs, typs, vies, seqL

    # exec loss_sum.backward(), self.optimizer.step() and self.scheduler.step()
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

        # float16, gradient scaling to speed up training
        if self.engine_cfg['enable_float16']:
            self.Scaler.scale(loss_sum).backward()
            self.Scaler.step(self.optimizer)
            scale = self.Scaler.get_scale()
            self.Scaler.update()
            # Warning caused by optimizer skip when NaN
            # The optimizer skips warnings raised when NaN occurs
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
        # info_dict: The result of the inference
        info_dict = Odict()
    
        for inputs in self.test_loader:
            # Transforms the input data
            ipts = self.inputs_pretreament(inputs, training=False)
            with autocast(enabled=self.engine_cfg['enable_float16']):
                # model inference
                # Calling the "forward" method of a subclass of model
                retval = self.student(ipts, training=False)
                inference_feat = retval['inference_feat']
                for k, v in inference_feat.items():
                    # distributed
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
        model.msg_mgr.log_info("Run train...")
        # inputs[0] -> [[No.1(seqL_batch, 3, 128, 128), No.2(seqL_batch, 3, 128, 128), ..., No.batchsize(seqL_batch, 3, 128, 128)], [No.1(seqL_batch, 128, 128), No.2(seqL_batch, 128, 128), ..., No.batchsize(seqL_batch, 128, 128)], ...]
        # inputs[1] -> [1, ...]
        # inputs[2] -> ['bg-01', ...]
        # inputs[3] -> ['000', ...]
        # inputs[4] -> batch[4] ->->  if fixed -> None, if unfixed -> np.asarray: seq length of each batch 
        for inputs in model.train_loader:
            
            # Transforms the input data
            # len(ipts): 5
            ipts = model.inputs_pretreament(inputs, training=True)
            with autocast(enabled=model.engine_cfg['enable_float16']):
                # run model
                student_retval = model.student(ipts)
                student_training_feat, visual_summary, student_between_feat= student_retval['training_feat'], student_retval['visual_summary'], student_retval['between_feat']
                del student_retval
                
                # Get teacher model
                teacher = selector_teacher(model.teachers)
                # Get output from teacher model
                teacher_retval = teacher(ipts)
                teacher_between_feat = teacher_retval['between_feat']
                del teacher_retval

                # Select output from student and teacher
                student_embedding, teacher_embedding = selector_output(student_between_feat,
                                                                        teacher_between_feat,
                                                                        model.cfgs['model_cfg']['out_layer'])
            
            # Calculate student loss
            student_loss_sum, student_loss_info = model.loss_aggregator(student_training_feat)
            # Calculate loss between student and teacher
            between_loss = model.between_criterion(student_embedding, teacher_embedding)
            # Calculate loss for discriminators
            d_loss = model.discriminators_criterion(student_embedding, teacher_embedding)
            # Get total loss
            total_loss = student_loss_sum + between_loss + d_loss
        
            ok = model.train_step(total_loss)
            if not ok:
                # jump out of the current loop
                continue
            
            loss_info = student_loss_info
            loss_info['scalar/between/loss'] = between_loss
            loss_info['scalar/discriminators/loss'] = d_loss
            
            visual_summary.update(loss_info) # update "loss_info" to "visual_summary" dict
            visual_summary['scalar/learning_rate'] = model.optimizer.param_groups[0]['lr']
            model.msg_mgr.train_step(loss_info, visual_summary)
            
            if model.iteration % model.engine_cfg['save_iter'] == 0:
                # save checkpoint
                model.save_ckpt(model.student, model.iteration)

                # if "with_test" is True， run test step
                if model.engine_cfg['with_test']:
                    model.msg_mgr.log_info("Running test...")
                    model.student.eval()
                    result_dict = BuildModel.run_test(model)
                    model.student.train()
                    if model.cfgs['trainer_cfg']['fix_BN']:
                        model.fix_BN(model.student)
                        for teacher in model.teachers:
                            model.fix_BN(teacher)
                        for discriminator in model.discriminators.discriminators:
                            model.fix_BN(discriminator)
                        
                    if result_dict:
                        model.msg_mgr.write_to_tensorboard(result_dict)
                    model.msg_mgr.reset_time()
            if model.iteration >= model.engine_cfg['total_iter']:
                break
    
    @ staticmethod
    def run_test(model):
        """Accept the instance object(model) here, and then run the test loop."""
        model.msg_mgr.log_info("Run test...")
        
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
            # Get the evaluate function
            eval_func = getattr(eval_functions, eval_func)
            valid_args = get_valid_args(
                eval_func, model.cfgs["evaluator_cfg"], ['metric'])
            
            try:
                dataset_name = model.cfgs['data_cfg']['test_dataset_name']
            except:
                dataset_name = model.cfgs['data_cfg']['dataset_name']
            # evaluate
            return eval_func(info_dict, dataset_name, **valid_args)
    

