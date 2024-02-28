import torch

from modeling.models import gaitGL_CASIAB, gaitSet, gaitPart, baseline_ResNet9
import numpy as np

from utils.msg_manager import get_msg_mgr


# load teachers checkpoint
def load_teachers_ckpt(device, model, save_name):
    
    msg_mgr = get_msg_mgr()

    checkpoint = torch.load(save_name, map_location=torch.device("cuda", device))
    model_state_dict = checkpoint['model'] # checkpoint['optimizer'], checkpoint['scheduler']

    # if not load_ckpt_strict is True, not strictly checking that the checkpoint is the same as the module of the defined model,
    # Find the keys that are common to both model state dictionaries and print them out after ordering them 
    if not model.load_ckpt_strict:
        msg_mgr.log_info("-------- Restored Params List --------")
        msg_mgr.log_info(sorted(set(model_state_dict.keys()).intersection(
            set(model.state_dict().keys()))))
    
    # The parameter "strict" is True by default, then you need to load the parameter strictly according to the Key value of the parameter in the model, if you add or delete the layer of the model, or change the parameter in the original layer, the loading will report an error.
    # If the parameter "strict" is Flase, only the parameter of the layer with the same name is allowed to be loaded, and the randomly value is be loaded to the modified layer.
    # It is important to note that if you just change the parameters of the original layer, but not the name, you will still get an error. This is because after finding the corresponding layer based on the key value and assigning it, the parameters are found to be mismatched.
    # At this point you can change the name of the original layer and load it again without reporting an error. Finally, you should note that "strict"=Flase should be used with caution, because it is very likely that you will not load in any parameters at all!
    model.load_state_dict(model_state_dict, strict=model.load_ckpt_strict)
    
    msg_mgr.log_info("Restore Parameters from %s !!!" % save_name)


# load teachers model from checkpoint
def resume_teachers_ckpt(device, model, dataset_name):
    # if type(restore_hint) == int
    if isinstance(model.restore_hint, int):
        save_name = './checkpoint/{}/{}/{}-{:0>5}.pt'.format(dataset_name, model.__name__, model.__name__, model.restore_hint)
    # if type(restore_hint) == string
    elif isinstance(model.restore_hint, str):
        save_name = model.restore_hint
    else:
        raise ValueError(
            "Error type for -Restore_Hint-, supported: int or string.")
    load_teachers_ckpt(device, model, save_name)



def get_teachers_student(model_cfg, dataset_name, device):

    model_map = {"GaitSet": gaitSet,
                 "GaitPart": gaitPart,
                 "GaitGL_CASIAB": gaitGL_CASIAB,
                 'Baseline_ResNet9': baseline_ResNet9}
    
    # Student setup
    assert model_cfg["student"] in model_map, "Student must be in %s" % " ".join(model_map.keys)
    student = model_map[model_cfg["student"]](model_cfg) # eg: GaitSet(model_cfg)
    student.__name__ = model_cfg["student"]
    student = student.to(device=torch.device("cuda", device))

    # Teachers setup
    teachers = []
    # Add teachers models into teacher model list
    for t in model_cfg["teachers"]:
        if t['name'] in model_map:
            net = model_map[t['name']](model_cfg)  # eg: GaitSet(model_cfg)
            net.__name__ = t['name']
            net.restore_hint = t['restore_hint']
            net.load_ckpt_strict = t['load_ckpt_strict']
            teachers.append(net)

    assert len(teachers) > 0, "Teachers must be in %s" % " ".join(model_map.keys)

    for i, teacher in enumerate(teachers):
        for p in teacher.parameters():
            p.requires_grad = False
        teacher = teacher.to(device=torch.device("cuda", device))
        teachers[i].__name__ = teacher.__name__
        teachers[i].restore_hint = teacher.restore_hint
        teachers[i].load_ckpt_strict = teacher.load_ckpt_strict

    # Load parameters in teacher models
    for teacher in teachers:
        resume_teachers_ckpt(device, teacher, dataset_name)
 
    if model_cfg["teacher_eval"]:
        for teacher in teachers:
            teacher.eval()
    
    return teachers, student



def selector_teacher(teachers):
    idx = np.random.randint(len(teachers))
    return teachers[idx]


# Select output from student and teacher
def selector_output(outputs, answers, idx):
    return [outputs[i] for i in idx], [answers[i] for i in idx]
    