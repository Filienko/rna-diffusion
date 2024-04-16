import math
import torch
import torch.nn as nn


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1 # 1D tensor (batch_size)

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)  
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    # exponential decreasing curve with embedding_dim/2 points
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :] # each timestep is multiplied by the curve
    # We end up with different exponential curves, one per timestep, with different amplitutes
    # emb is of size (batch_size, embedding_dim/2) 
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1) 
    if embedding_dim % 2 == 1:  # zero pad if dimension is not a multiple of 2  
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0)) # pad to the right by 1
    return emb

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.LayerNorm(in_channels, eps=1e-6, elementwise_affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer = nn.Linear(in_channels, in_channels*2)

    def forward(self, x):
        x = self.layer(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer = nn.Linear(in_channels, in_channels//2)

    def forward(self, x):
        x = self.layer(x)
        return x
    
class ResLinear(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.5, temb_channels=4, num_classes=0, float_precision='single'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        # Projection of x for dimensions matching in res connexion
        if (in_channels != out_channels):
            self.x_proj = torch.nn.Linear(in_channels + num_classes, out_channels) 

        if float_precision=='single':
            self.batch_norm_temb1 = Normalize(in_channels + 1)
            self.batch_norm1 = Normalize(out_channels)
            self.batch_norm_temb2 = Normalize(out_channels + 1)
            self.batch_norm2 = Normalize(out_channels)
       
        self.w1 = nn.Linear(in_channels + num_classes + 1, out_channels) # first linear projection
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(out_channels + num_classes + 1, out_channels) # second linear projection

    def forward(self, x, temb, label=None):
        y = torch.cat((x, temb), dim=1) # Timestep embedding concatenation
        y = self.batch_norm_temb1(y) # Normalize
        if (label is not None):
            y = torch.cat((y, label), dim=1) # Conditionning on tissue type
        y = self.w1(y) # Linear
        y = self.batch_norm1(y) # Normalize
        y = self.relu(y) # Non-linearity
        y = self.dropout(y) # Dropout
        y = torch.cat((y, temb), dim=1) # Timestep embedding concatenation
        y = self.batch_norm_temb2(y) # Normalize
        if (label is not None):
            y = torch.cat((y, label), dim=1) # Conditionning on tissue type
        y = self.w2(y) # Linear
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)
        # Match dimensions for residual connexion
        if (self.in_channels != self.out_channels):
            if (label is not None):
                x = torch.cat((x, label), dim=1) # Conditionning on tissue type
            x = self.x_proj(x)

        return x + y # Residual connexion

# Attention Block
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)

        self.q = torch.nn.Linear(in_channels, in_channels)
        self.k = torch.nn.Linear(in_channels, in_channels)
        self.v = torch.nn.Linear(in_channels, in_channels)

        self.proj_out = torch.nn.Linear(in_channels, in_channels)  

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        # q = h_ * self.q
        # k = h_ * self.k
        # v = h_ * self.v

        # compute attention
        b, w = q.shape #batch, width(hw)
        # q = q.reshape(b, w, 1) # b,hw,c
        # k = k.reshape(b, w, 1) # b,hw,c
        # k = k.permute(0, 2, 1)   # b,1,hw
        q = torch.unsqueeze(q, dim=-1) #b,hw,1

        k = torch.unsqueeze(k, dim=-1) #b,hw,1  #POUR CDIST
        w_ = torch.cdist(q, k, p=2) #b,hw,hw
        w_ = torch.nn.functional.softmin(w_, dim=2)

        # k = torch.unsqueeze(k, dim=1) #b,1,hw  #POUR BMM
        # w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # w_ = torch.nn.functional.softmax(w_, dim=2)

        # w_ = w_ * (int(c)**(-0.5))
        # w_ = w_ * (int(w)**(-0.5))

        # attend to values
        # v = v.reshape(b, 1, w)
        v = torch.unsqueeze(v, dim=1) #b,1,hw
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        # h_ = h_.reshape(b, 1, w)
        # h_ = h_.permute(0, 2, 1) # b, hw, 1
        #remove last dimension
        # h_ = h_.squeeze(2)

        h_ = h_.squeeze(1)

        h_ = self.proj_out(h_)

        return x+h_

class ModelDDIM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # get params
        dropout = config.model.dropout # dropout
        self.dim_t = config.model.dim_t # timestep embedding size
        self.temb_ch = self.dim_t * 4 # timestep embedding size as input of dense layers
        self.is_time_embed = config.model.is_time_embed # whether to use embedding for time steps
        self.is_y_cond = config.model.is_y_cond # whether to condition on given target label
        self.num_classes = config.data.num_classes # number of classes in target labels
        self.num_resolutions = config.model.num_res_blocks # how many ResLinear blocks
        self.resolution = config.data.image_size # samples resolutions (number of genes)
        self.middle_resolution = self.resolution // (2 ** (self.num_resolutions)) # mid resolution
        self.attn_resolutions = config.model.attn_resolutions # attention resolution
        self.d_layers = config.model.d_layers # number of layers in block

        self.timestep_max = config.diffusion.num_diffusion_timesteps # number of diffusion steps
        self.with_attn = config.model.with_attn # whether to use attention

        self.use_y_emb = config.model.use_y_emb # whether to embe the target labels
        self.dim_y_emb = config.model.dim_y_emb # size of target labels embedding

        self.precision = config.model.precision # outdated: this code uses automatically mixed precision in float16

        if self.use_y_emb and self.is_y_cond:
            self.y_emb = nn.Embedding(self.num_classes, self.dim_y_emb)

        # timestep embedding
        if self.is_time_embed:
            self.temb_dense = nn.ModuleList([
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.dim_t),
            ])

        # sizes
        if self.is_y_cond:
            if self.use_y_emb:
                y_num_classes_temp = self.num_classes * self.dim_y_emb
            else:
                y_num_classes_temp = self.num_classes
        else:
            y_num_classes_temp = 0

        self.mid = nn.ModuleList(
            [
                ResLinear(
                    in_channels=self.d_layers[i - 1] if i else self.resolution, 
                    out_channels=self.d_layers[i], 
                    temb_channels=self.temb_ch, 
                    dropout=dropout,
                    num_classes=y_num_classes_temp,
                    float_precision=self.precision,
                )
                for i in range(len(self.d_layers))
            ]
        )


        if self.precision=='single':
            self.norm_out = Normalize(self.d_layers[-1]) 

        self.lin_out = torch.nn.Linear(self.d_layers[-1], self.resolution)


    def forward(self, x, t, y=None):
        # timestep embedding
        if self.is_time_embed:
            temb = get_timestep_embedding(t, self.temb_ch) 
            temb = self.temb_dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb_dense[1](temb) # final shape: (batch size x 1)
        else:
            temb = t/self.timestep_max
            temb = torch.unsqueeze(temb, dim=1) # final shape: (batch size x 1)

        if self.use_y_emb and self.is_y_cond:
            y = self.y_emb(y)
            # flatten y
            y = y.view(y.shape[0], -1)
            y = nonlinearity(y)
        
        hs = [x]
        #middle
        h = hs[-1]
        for block in self.mid:
            if self.is_y_cond:
                h = block(h, temb, y)
            else:
                h = block(h, temb)

        # output
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.lin_out(h)

        return h