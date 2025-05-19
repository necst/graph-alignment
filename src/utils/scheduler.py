import math
import numpy as np

def cosine_decay_lr(step, warmup_steps, tot_steps, min_gain=0.0001) -> float:
    r"""
    Define the policy to update the learning rate with linear warmup epochs
    and cosine decay starting at a specific epoch.

    Args:
        step (int): current training step.
        warmup_steps(int): number of warmup steps.
        tot_steps (int): total number of steps for training.

    Returns:
        lr_gain (float): a gain to multiply the default learning rate with.
    """
    if step < warmup_steps:
        # linear warmup
        lr_gain = (step + 1) / warmup_steps
        return lr_gain
    
    elif step >= warmup_steps:
        # cosine decay
        decay_steps = tot_steps - warmup_steps
        progress = (step - warmup_steps) / decay_steps
        lr_gain = 0.5 * (1 + math.cos(math.pi * progress))
        return max(lr_gain, min_gain)
    else:
        # constant learning rate 
        lr_gain = 1
        return lr_gain




def exponential_decay_lr():
    pass



def cosine_increase_law(initial_param:float, final_param:float, tot_steps:float):
    r"""
    Create a lookup table to increase a parameter (tau, wd...) following a cosine law policy.

    Args:
        initial_param (float): initial value.
        final_param(float): final value.
        tot_steps (float): total number of steps.

    Returns:
        schedule (np.ndarray): lut with the values of the parameter for the specified number of steps.
    """
    steps = np.arange(tot_steps)
    schedule = initial_param + 0.5 * (final_param - initial_param) * (1 - np.cos(np.pi * steps / tot_steps))
    return schedule


def linear_increase_law(initial_param:float, final_param:float, tot_steps:float):
    r"""
    Create a lookup table to increase a parameter (tau, wd...) following a cosine law policy.

    Args:
        initial_param (float): initial value.
        final_param(float): final value.
        tot_steps (float): total number of steps.

    Returns:
        schedule (np.ndarray): lut with the values of the parameter for the specified number of steps.
    """
    steps = np.arange(tot_steps)
    schedule = initial_param + (final_param - initial_param) * (steps / tot_steps)
    return schedule