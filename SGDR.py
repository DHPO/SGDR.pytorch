import torch
import math


def SGDR(optimizer, T_0, T_mul):
    def schedule_func(epoch):
        if not hasattr(schedule_func, 'T_start'):
            schedule_func.T_start = 0
            schedule_func.T_length = T_0
            schedule_func.T_next = T_0
        if epoch == schedule_func.T_next:
            schedule_func.T_start = schedule_func.T_next
            schedule_func.T_length *= T_mul
            schedule_func.T_next += schedule_func.T_length
        T_cur = epoch - schedule_func.T_start
        lr_coef = 0.5 * (1 + math.cos(T_cur / schedule_func.T_length * math.pi))
        print("Epoch {}, lr_coef == {}".format(epoch, lr_coef))
        return lr_coef
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_func)
