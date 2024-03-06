import torch.nn.functional as F


def L2_soft(outputs, targets):
    '''
    F.mse_loss:
        (MSE) Calculation of the average of the mean squared errors between the predicted and target values.
        
        In regression tasks, F.mse_loss is commonly used to measure the difference between the model's predicted and true values. Compared to F.l1_loss, F.mse_loss is more sensitive to large errors because it uses squared error instead of absolute error. This means that F.mse_loss may be more affected if there are outliers.
    
    L2_soft:
        The F.softmax() function takes the input tensor and performs a softmax operation to convert it into a probability distribution. This means that both softmax_outputs and softmax_targets will be probability distributions with the sum of all their elements equal to one.

        These two probability distributions are then passed as arguments to the F.mse_loss() function, which calculates the mean squared error between them. This has the effect of introducing a comparison of the probability distributions between the predicted and target values, rather than directly comparing the raw values. This may make sense in some specific scenarios, such as classification tasks where you care about the similarity of the class distributions between the predicted and target values, not just the numerical values.
    '''
    softmax_outputs = F.softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)

    return F.mse_loss(softmax_outputs, softmax_targets)