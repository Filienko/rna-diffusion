import torch
from torch.cuda.amp import autocast

print("Functions/Losses.py")

# 'simple' loss
def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, 
                          y: torch.Tensor = None,
                          keepdim=False):
    '''
    model: a PyTorch model that takes in an input tensor and returns an output tensor.
    x0: batch de samples sans bruit (x0).
    t: timestep/index de bruit pour chaque sample du batch.
    e: a PyTorch tensor of shape (batch_size, channels, height, width) that represents a random noise.
    b: Le vecteur des betas (échelles de bruit selon le timestep).
    '''

    # a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt() # noising

    # Runs the forward pass with autocasting.
    # Loss is computed under autocast env
    with autocast():
        output = model(x, t, y) # predicting noise

        if keepdim:
            # Returns the individual losses for each sample in the batch
            loss = (e - output).square().sum(dim=(1))
            avg = []
        else:
            # Average loss across the entire batch
            avg = (e - output).abs().mean(dim=(1)).mean(dim=0)
            loss = (e - output).square().sum(dim=(1)).mean(dim=0)  
    
    return loss, avg


loss_registry = {
    'simple': noise_estimation_loss,
}