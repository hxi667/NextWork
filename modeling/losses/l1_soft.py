import torch.nn.functional as F


def L1_soft(outputs, targets):
    '''
    F.l1_loss: 
        (MAE) Calculation of the average of the absolute errors between the predicted and target values.

    L1_soft:
        The F.softmax() function takes the input tensor and performs a softmax operation to convert it into a probability distribution. This means that both softmax_outputs and softmax_targets will be probability distributions with the sum of all their elements equal to one.
    
        These two probability distributions are then passed as arguments to the F.l1_loss() function, which calculates the mean absolute error between them. This has the effect of introducing a comparison of the probability distributions between the predicted and target values, rather than directly comparing the raw values. This may make sense in some specific scenarios, such as classification tasks where you care about the similarity of the class distributions between the predicted and target values, not just the numerical values.

    '''
    softmax_outputs = F.softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)

    return F.l1_loss(softmax_outputs, softmax_targets)