import torch
from torch.cuda.amp import autocast


print("Functions/Denoising.py")

def compute_alpha(beta, t):
    #littéralement le meme calcul que dans la loss
    #juste je comprend pas pourquoi on concatene un zero au debut puis on fait t+1 
    #on pourrait surement centraliser le calcul plus tard
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    # a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1)

    return a

#generalised step, avec le ETA (DDIM, etc...)
#(DDPM with a larger variance)
#ETA = 1 = DDPM, ETA = 0 = DDIM
def generalized_steps(x, seq, model, b, y=None, **kwargs):
    with torch.no_grad():
        #n = taille du batch
        n = x.size(0)
        #on ajoute -1 au debut de la liste seq sans le dernier element, mm taille que seq
        #on shift la sequence vers la gauche
        seq_next = [-1] + list(seq[:-1])
        #liste des predis pour x0
        x0_preds = []
        #listes des predis des x_t-1
        xs = [x]
        #iterateur sur les binomes des t et t-1 à l'envers (t décroissants) (xT = max bruit, x0 = image nette)
        for i, j in zip(reversed(seq), reversed(seq_next)): 
            #seq_next est donc tjr une étape plus proche du t=0 que seq.
            t = (torch.ones(n) * i).to(x.device) #t actuel, vecteur de t, un pour chaque sample du batch
            next_t = (torch.ones(n) * j).to(x.device) #t-1, vecteur de t-1, un pour chaque sample du batch
            at = compute_alpha(b, t.long()) #alpha_t
            at_next = compute_alpha(b, next_t.long()) #alpha_t-1

            # xt = xs[-1].to('cuda:1') #On recupere la derniere prediction de x_t
            xt = xs[-1].to(x.device) #On recupere la derniere prediction de x_t

            # Runs the forward pass with autocasting.
            # Loss is computed under autocast env
            with autocast():    
                et = model(xt, t, y) #On fait la prediction de e_t

                #convert to half precision
                # et = model(xt.half(), t.half(), y).float()

                #Prediction de x0 avec x_t et le bruit estimé de x_t
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() #On enleve le bruit de x_t pour estimer x0
            
                x0_preds.append(x0_t.to('cpu')) #On ajoute la prediction de x0 à la liste des preds

                c1 = (
                    kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                #EQUA Fin Page page 18 du papier.
                #c1 = sigma_t(eta), Première EQUA Page 19 du papier.
                #c2 = derniere racine de la Dernière EQUA Page 18 du papier.

                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et #on estime x_t-1
                #EQUA Fin Page page 18 du papier.

                #On ajoute la prediction de x_t-1 à la liste des preds
                xs.append(xt_next.to('cpu'))

    #On retourne la liste des preds des x_t-1 et la liste des preds de x0 selon les x_t
    return xs, x0_preds

#DDPM avec ETA > 1 (sigma^ (fin page 6 du papier))
def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        #n = taille du batch
        n = x.size(0)
        #on ajoute -1 au debut de la liste seq sans le dernier element, mm taille que seq
        #on shift la sequence vers la gauche
        seq_next = [-1] + list(seq[:-1])
        #preds de x_t-1 et x0
        xs = [x]
        x0_preds = []
        #cette ligne est ultra useless bravo :)
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            #seq_next est donc tjr une étape plus proche du t=0 que seq.
            t = (torch.ones(n) * i).to(x.device) #t actuel, vecteur de t, un pour chaque sample du batch
            next_t = (torch.ones(n) * j).to(x.device) #t-1, vecteur de t-1, un pour chaque sample du batch
            at = compute_alpha(betas, t.long()) #alpha_t
            atm1 = compute_alpha(betas, next_t.long()) #alpha_t-1
            beta_t = 1 - at / atm1
            #EQUA (61) Page 17 du papier. (Pourquoi??? On utilise liitéralement les beta_t pour calculer les alpha puis on utilise les alpha pour calculer les beta_t...???)

            # x = xs[-1].to('cuda:1')
            x = xs[-1].to(x.device)

            # Runs the forward pass with autocasting.
            # Loss is computed under autocast env
            with autocast():
                output = model(x, t)
            
            e = output #On fait la prediction de e_t
            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e #On enleve le bruit estimé e de x_t pour estimer x0
            x0_from_e = torch.clamp(x0_from_e, -1, 1) #On clip les valeurs de x0 entre -1 et 1 pour éviter les valeurs aberrantes
            x0_preds.append(x0_from_e.to('cpu')) #On ajoute la prediction de x0 à la liste des preds
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x) #On genere un bruit gaussien
            mask = 1 - (t == 0).float() #On masque les valeurs qui ont un t=0 (bruit nul)
            # mask = mask.view(-1, 1, 1, 1)
            mask = mask.view(-1, 1)
            logvar = beta_t.log() #On calcule le log de beta_t
            sample = mean + mask * torch.exp(0.5 * logvar) * noise #On estime x_t-1

            #On ajoute la prediction de x_t-1 à la liste des preds
            xs.append(sample.to('cpu')) 
    return xs, x0_preds