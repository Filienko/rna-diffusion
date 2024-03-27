import torch.optim as optim
import pytorch_warmup as warmup

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim.optimizer))

def get_scheduler(config, optimizer):
    if config.optim.scheduler == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=config.optim.scheduler_step_size, gamma=config.optim.scheduler_gamma)
    elif config.optim.scheduler == 'MultiStepLR':
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.optim.scheduler_milestones, gamma=config.optim.scheduler_gamma)
    elif config.optim.scheduler == 'ExponentialLR':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.optim.scheduler_gamma)
    elif config.optim.scheduler == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.optim.scheduler_factor, patience=config.optim.scheduler_patience, verbose=True)
    elif config.optim.scheduler =='Cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    elif config.optim.scheduler =='OneCycle':
        print(f"steps per epoch: {config.n_batchs}")
        print(f"n epochs: {config.training.n_epochs}")
        print(f"maximum LR reached at step: 1000.")
        lr_peak_prop = 1000/(config.n_batchs*config.training.n_epochs)
        # lr_peak_prop = 0.5
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.optim.lr, steps_per_epoch=config.n_batchs, epochs=config.training.n_epochs, pct_start = lr_peak_prop)
    
    else:
        raise NotImplementedError(
            'Scheduler {} not understood.'.format(config.optim.scheduler))

def get_scheduler_warmup(config, optimizer):
    if config.optim.scheduler_warmup == 'LinearWarmup':
        return warmup.LinearWarmup(optimizer, warmup_period=config.optim.warmup_period, last_step=config.optim.warmup_last_step)
    elif config.optim.scheduler_warmup == 'ExponentialWarmup':
        return warmup.ExponentialWarmup(optimizer, warmup_period=config.optim.warmup_period, last_step=config.optim.warmup_last_step)
    elif config.optim.scheduler_warmup == 'UntunedLinearWarmup':
        return warmup.UntunedLinearWarmup(optimizer, last_step=config.optim.warmup_last_step)
    elif config.optim.scheduler_warmup == 'UntunedExponentialWarmup':
        return warmup.UntunedExponentialWarmup(optimizer, last_step=config.optim.warmup_last_step)
    else:
        raise NotImplementedError(
            'Warmup Scheduler {} not understood.'.format(config.optim.scheduler_warmup))
