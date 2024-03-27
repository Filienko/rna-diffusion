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
    assert len(timesteps.shape) == 1 #1D tensor (batch_size)

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)  #log naturel de 10000 = 9.21
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    #avec ca ^ (fleche du haut) on genere une courbe exponentiellement decroissante de 1 a 1E10-4, avec embedding_dim/2 points
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :] #on multiplie chaque timestep par la courbe exponentielle.
    #On a donc plusieurs courbes exponentielles, une par timestep, de différentes amplitudes
    #emb est donc un tensor de taille (batch_size, embedding_dim/2)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1) #on concatene sin et cos en emb, les deux faisaient half_dim
    #ca fait un paquet de sinus et cosinus, avec des amplitudes différentes, et des phases différentes
    #emb est donc un tensor de taille (batch_size, embedding_dim - embedding_dim%2)
    if embedding_dim % 2 == 1:  # zero pad      #si la dim est impaire, on rajoute un 0, pour obtenir la dim voulue
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0)) #pad to the right by 1. (left, right, top, bottom)
    return emb

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.LayerNorm(in_channels, eps=1e-6, elementwise_affine=True)
    # return torch.nn.LayerNorm(in_channels, eps=1e-6, elementwise_affine=True, dtype=torch.float16) #HALF

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
            # self.batch_norm1 = torch.nn.LayerNorm(in_channels, eps=1e-6, elementwise_affine=True)
            # self.batch_norm_temb = torch.nn.LayerNorm(in_channels, eps=1e-6, elementwise_affine=True)
            self.batch_norm_temb1 = Normalize(in_channels + 1)
            self.batch_norm1 = Normalize(out_channels)
            self.batch_norm_temb2 = Normalize(out_channels + 1)
            self.batch_norm2 = Normalize(out_channels)
        # elif float_precision=='half':
        #     self.batch_norm1 = torch.nn.LayerNorm(out_channels, eps=1e-6, elementwise_affine=True, dtype=torch.float16)
        #     self.batch_norm_temb = torch.nn.LayerNorm(out_channels + 1, eps=1e-6, elementwise_affine=True, dtype=torch.float16)
        #     self.batch_norm2 = torch.nn.LayerNorm(out_channels, eps=1e-6, elementwise_affine=True, dtype=torch.float16)

        self.w1 = nn.Linear(in_channels + num_classes + 1, out_channels) # first linear projection
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(out_channels + num_classes + 1, out_channels) # second linear projection
        
    # def forward(self, x, temb, label=None):
    #     y = self.batch_norm1(x) # Normalize
    #     y = self.relu(y) # Non-linearity
    #     if (label is not None):
    #         y = torch.cat((y, label), dim=1) # Conditionning on tissue type
    #     y = self.w1(y) # Linear
    #     y = torch.cat((y, temb), dim=1) # Timestep embedding concatenation
    #     y = self.batch_norm_temb(y) # Normalize
    #     y = self.relu(y) # Non-linearity
    #     y = self.dropout(y) # Dropout
    #     if (label is not None):
    #         y = torch.cat((y, label), dim=1) # Conditionning on tissue type
    #     y = self.w2(y) # Linear

    #     # Match dimensions for residual connexion
    #     if (self.in_channels != self.out_channels):
    #         if (label is not None):
    #             x = torch.cat((x, label), dim=1) # Conditionning on tissue type
    #         x = self.x_proj(x)

    #     return x + y # Residual connexion

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

#Res Block by Romain: Linear + BatchNorm + ReLU + Dropout + Residual
class OldResLinear(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.5, temb_channels=512, num_classes=0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        if (in_channels != out_channels):
            self.x_proj = torch.nn.Linear(in_channels, out_channels) #projection de x

        self.batch_norm_temb = Normalize(out_channels + 1)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.w1 = nn.Linear(in_channels, out_channels)
        self.batch_norm1 = Normalize(out_channels)

        self.w2 = nn.Linear(out_channels + 1 + num_classes, out_channels)
        self.batch_norm2 = Normalize(out_channels)

    def forward(self, x, temb, label=None):
        #Linear + BatchNorm + ReLU + Dropout
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = torch.cat((y, temb), dim=1) #Ajout du timestep embedding
        y = self.batch_norm_temb(y)

        if (label is not None):
            y = torch.cat((y, label), dim=1)

        #Linear + BatchNorm + ReLU + Dropout
        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        #Residual
        if (self.in_channels != self.out_channels):
            x = self.x_proj(x)

        out = x + y #Residual

        return out

#Attention Block
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)

        self.q = torch.nn.Linear(in_channels, in_channels)
        self.k = torch.nn.Linear(in_channels, in_channels)
        self.v = torch.nn.Linear(in_channels, in_channels)

        self.proj_out = torch.nn.Linear(in_channels, in_channels)  #Projection par un dense layer

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

        #get params
        dropout = config.model.dropout #dropout
        self.dim_t = config.model.dim_t #taille de la génération du timestep embedding
        self.temb_ch = self.dim_t * 4 #taille du timestep embedding en entrée de ses dense layers
        self.is_time_embed = config.model.is_time_embed
        self.is_y_cond = config.model.is_y_cond #si on utilise le label conditionnel
        self.num_classes = config.data.num_classes #nombre de classes
        self.num_resolutions = config.model.num_res_blocks #combien de ResLinear par downsample/upsample
        self.resolution = config.data.image_size #resolution de l'image (1024 pour nous)
        self.middle_resolution = self.resolution // (2 ** (self.num_resolutions)) #resolution du milieu
        self.attn_resolutions = config.model.attn_resolutions #resolution de l'attention
        self.d_layers = config.model.d_layers #nombre de dense layers dans le middle block

        self.timestep_max = config.diffusion.num_diffusion_timesteps #nombre de timesteps max

        self.with_attn = config.model.with_attn #si on utilise l'attention

        self.use_y_emb = config.model.use_y_emb #utilisation d'un layer d'embedding pour les y
        self.dim_y_emb = config.model.dim_y_emb #taille de l'embedding des y

        self.precision = config.model.precision # float32 or 16
        #print("Float precision:", self.precision)

        if self.use_y_emb and self.is_y_cond:
            self.y_emb = nn.Embedding(self.num_classes, self.dim_y_emb)

        # timestep embedding
        if self.is_time_embed:
            # self.temb = nn.Module()
            self.temb_dense = nn.ModuleList([
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.dim_t),
            ])

        #downsampling
        # self.down = nn.ModuleList()
        # for i_level in range(self.num_resolutions):
        #     block = nn.Module()
        #     res_in = self.resolution // (2 ** (i_level))
        #     block.reslin = ResLinear(res_in, res_in, dropout, self.temb_ch)
        #     if self.with_attn and res_in in attn_resolutions:
        #         block.attn = AttnBlock(res_in)
        #     block.downsample = Downsample(res_in)
        #     self.down.append(block)

        # middle
        # self.mid = nn.Module()
        # self.mid.block_1 = ResLinear(in_channels=self.middle_resolution,
        #                                out_channels=self.middle_resolution,
        #                                temb_channels=self.temb_ch,
        #                                dropout=dropout)
        # if self.with_attn and self.middle_resolution in attn_resolutions:
        #     self.mid.attn_1 = AttnBlock(self.middle_resolution)
        # self.mid.block_2 = ResLinear(in_channels=self.middle_resolution,
        #                                out_channels=self.middle_resolution,
        #                                temb_channels=self.temb_ch,
        #                                dropout=dropout)

        #taille du vecteur y concaténé
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
                # AttnBlock(d_layers[i]) if self.with_attn and i in attn_resolutions #JSP CMMENT FAIRE PR LINSTANT
                for i in range(len(self.d_layers))
            ]
        )

        #upsampling
        # self.up = nn.ModuleList()
        # for i_level in range(self.num_resolutions):
        #     block = nn.Module()
        #     res_in = self.resolution // (2 ** (self.num_resolutions - i_level))
        #     block.reslin = ResLinear(res_in, res_in, dropout, self.temb_ch)
        #     if self.with_attn and res_in in attn_resolutions:
        #         block.attn = AttnBlock(res_in)
        #     block.upsample = Upsample(res_in)
        #     self.up.append(block)
        if self.precision=='single':
            # self.norm_out = torch.nn.LayerNorm(self.d_layers[-1], eps=1e-6, elementwise_affine=True)
            self.norm_out = Normalize(self.d_layers[-1]) 
        # elif self.precision=='half':
        #     self.norm_out = torch.nn.LayerNorm(self.d_layers[-1], eps=1e-6, elementwise_affine=True, dtype=torch.float16)

        self.lin_out = torch.nn.Linear(self.d_layers[-1], self.resolution)

        # self.norm_out = Normalize(self.resolution)
        # self.lin_out = torch.nn.Linear(self.resolution, self.resolution)


    def forward(self, x, t, y=None):
        # timestep embedding
        if self.is_time_embed:
            temb = get_timestep_embedding(t, self.temb_ch) # shape: (?)
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

        #downsampling
        hs = [x]
        # for i_level in range(self.num_resolutions):
        #     h = self.down[i_level].reslin(hs[-1], temb)
        #     if self.with_attn and h.shape[1] in self.attn_resolutions:
        #         h = self.down[i_level].attn(h)
        #     h = self.down[i_level].downsample(h)
        #     hs.append(h)


        #middle
        h = hs[-1]
        for block in self.mid:
            if self.is_y_cond:
                h = block(h, temb, y)
            else:
                h = block(h, temb)

        #upsampling
        # for i_level in range(self.num_resolutions):
        #     h = h + hs.pop() #Residual v2.2 
        #     h = self.up[i_level].reslin(h, temb)
        #     if self.with_attn and h.shape[1] in self.attn_resolutions:
        #         h = self.up[i_level].attn(h)
        #     h = self.up[i_level].upsample(h)

        # output
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.lin_out(h)

        return h