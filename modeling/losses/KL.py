import torch.nn.functional as F
import torch

def KL(outputs, targets):
    '''
    F.log_softmax: 
        A log_softmax operation was performed on the output of the model to convert it to log probability.
    
    F.softmax:
        A softmax operation was performed on the target value to convert it to a probability distribution.


    '''
    log_softmax_outputs = F.log_softmax(outputs, dim=-1)
    softmax_targets = F.softmax(targets, dim=-1)

    return F.kl_div(log_softmax_outputs, softmax_targets, reduction='sum')
    # return F.kl_div(log_softmax_outputs, softmax_targets, reduction='mean')
