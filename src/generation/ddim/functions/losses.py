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

    #index_select = on selectionne les cumprod de b selon t
    #view = reshape vers (batch_size, 1, 1, 1)
    #alphas (a) = 1 - betas (b) //en gros c'est le alpha du papier (= alpha barre du DDPM) 
    #EQUA (60) & (61) Pg.17 du papier.

    x = x0 * a.sqrt() + e * (1.0 - a).sqrt() #Definition d'un batch de x_t (x) selon alpha_t (a) et x0 (x0 = batch des samples sans bruit) et e_t (e)
    #EQUA (4) Pg.3 du papier.

    # Runs the forward pass with autocasting.
    # Loss is computed under autocast env
    with autocast():
        output = model(x, t, y)
        #On fait la prediction, estimation du bruit

        if keepdim:
            # Returns the individual losses for each sample in the batch
            # return (e - output).square().sum(dim=(1, 2, 3)) #loss = diff entre bruit réel e et la prediction du bruit
            loss = (e - output).square().sum(dim=(1))
            avg = []
            #EQUA (5) Pg.3 du papier.
        else:
            # Average loss across the entire batch
            # return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
            # return (e - output).square().sum(dim=(1)).mean(dim=0)
            avg = (e - output).abs().mean(dim=(1)).mean(dim=0)
            loss = (e - output).square().sum(dim=(1)).mean(dim=0)  #EQUA (5) Pg.3 du papier.
    
    return loss, avg


loss_registry = {
    'simple': noise_estimation_loss,
}