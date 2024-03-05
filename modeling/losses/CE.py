import torch.nn.functional as F


def CrossEntropy(outputs, targets):
    '''
    F.log_softmax: 
        A log_softmax operation was performed on the output of the model to convert it to log probability.
    
    F.softmax:
        A softmax operation was performed on the target value to convert it to a probability distribution.

    CrossEntropy:
        The log probability and the probability distribution of the target value are multiplied and the results for each sample are summed, and finally the average is taken and the negative sign is taken.

        This process actually calculates the cross-entropy loss for each sample and then averages the loss over all samples. This method serves to measure the difference between the model's predicted probability distribution and the probability distribution of the true label. During training, the model parameters are optimised by minimising the cross-entropy loss, which makes the model's predictions closer to the distribution of the true labels.

        It is important to note that this approach is usually effective for multiclassification problems because it takes into account the difference between the probability distribution of the model output and the probability distribution of the true labels.
    '''
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)

    return -(log_softmax_outputs*softmax_targets).sum(dim=1).mean()