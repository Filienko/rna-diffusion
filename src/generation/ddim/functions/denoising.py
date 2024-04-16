import torch
from torch.cuda.amp import autocast


print("Functions/Denoising.py")

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1)

    return a


def generalized_steps(x, seq, model, b, y=None, **kwargs):
    with torch.no_grad():
        n = x.size(0) # batch size
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        # iterating over pairs of t and t-1 backward (i.e. xT = noisy image, x0 = denoised image)
        for i, j in zip(reversed(seq), reversed(seq_next)): 
            t = (torch.ones(n) * i).to(x.device) 
            next_t = (torch.ones(n) * j).to(x.device) 
            at = compute_alpha(b, t.long()) # alpha_t
            at_next = compute_alpha(b, next_t.long()) # alpha_t-1

            xt = xs[-1].to(x.device) # last prediction of x_t

            # Runs the forward pass with autocasting.
            # Loss is computed under autocast env
            with autocast():    
                et = model(xt, t, y) # predicting noise e_t
                # predicting x_0 given x_t and predicted noise in x_t
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() # estimation of x_0
                x0_preds.append(x0_t.to('cpu')) 

                 # denoising to get x_{t-1}
                c1 = (
                    kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                xs.append(xt_next.to('cpu'))

    # lists of denoised x_{t-1} and predicted x_0
    return xs, x0_preds

# DDPM with ETA > 1
def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device) 
            next_t = (torch.ones(n) * j).to(x.device) 
            at = compute_alpha(betas, t.long()) # alpha_t
            atm1 = compute_alpha(betas, next_t.long()) # alpha_t-1
            beta_t = 1 - at / atm1
            x = xs[-1].to(x.device)

            # Runs the forward pass with autocasting.
            # Loss is computed under autocast env
            with autocast():
                output = model(x, t)
            
            e = output # predicted noise e_t
            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e # predicting x_0
            x0_from_e = torch.clamp(x0_from_e, -1, 1) # clipping to avoid outliers above -1 and 1
            x0_preds.append(x0_from_e.to('cpu')) 
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x) # random noise
            mask = 1 - (t == 0).float() # masking values for which t=0 (i.e. no noise)
            mask = mask.view(-1, 1)
            logvar = beta_t.log() 
            sample = mean + mask * torch.exp(0.5 * logvar) * noise # denoising
            xs.append(sample.to('cpu')) 
    return xs, x0_preds