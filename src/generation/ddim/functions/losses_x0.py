import torch
from torch.cuda.amp import autocast

print("Functions/Losses.py")

def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    '''
    model: a PyTorch model that takes in an input tensor and returns an output tensor.
    x0: batch de samples sans bruit (x0).
    t: timestep/index de bruit pour chaque sample du batch.
    e: a PyTorch tensor of shape (batch_size, channels, height, width) that represents a random noise.
    b: Le vecteur des betas (échelles de bruit selon le timestep).
    '''

    # a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1)
    xt = x0 * a.sqrt() + e * (1.0 - a).sqrt() # noising

    # Runs the forward pass with autocasting.
    # Loss is computed under autocast env
    with autocast():
        et = model(xt, t) # predicting noise
        x0_t = (xt - et * (1 - a).sqrt()) / a.sqrt() #denoising to estimate x_0

        if keepdim:
            # Returns the individual losses for each sample in the batch
            loss = (e - et).square().sum(dim=(1))
        else:
            # Average loss across the entire batch
            avg = (e - et).abs().mean(dim=(1)).mean(dim=0)
            loss = (x0 - x0_t).square().sum(dim=(1)).mean(dim=0)
    
    return loss, avg

loss_registry = {
    'simple': noise_estimation_loss,
}