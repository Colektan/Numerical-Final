import numpy as np
import torch
from scipy.stats import multivariate_normal

np.random.seed(1687)

def inverse_sample(F_inv):
    p = np.random.rand()
    return F_inv(p)

def reject_sample(f, kg, kg_sample):
    cal_count = 0
    while 1:
        cal_count += 1
        x = kg_sample()
        u = np.random.rand()
        if f(x) / kg(x) > 1:
            print(f"Invalid Point {x}")
        if u < f(x) / kg(x):
            yield (x, cal_count)
        

def langevin(f, epsi, x_0):
    x = torch.tensor(x_0, requires_grad=True)
    while 1:
        x.requires_grad_(True)
        y = torch.log(f(x))
        y.backward()
        z = torch.normal(0, 1, x.shape)
        with torch.no_grad():
            x = x + epsi ** 2 / 2 * x.grad + epsi * z
        yield x.numpy()

def langevin_adjusted(f, epsi, x_0):
    x = torch.tensor(x_0, requires_grad=True)
    eye = np.eye(len(x_0)) * epsi
    cal_count = 0
    while 1:
        cal_count += 1
        x.requires_grad_(True)
        y = f(x)
        out = torch.log(y)
        out.backward()
        z = torch.normal(0, 1, x.shape)
        with torch.no_grad():
            x_new = x + epsi ** 2 / 2 * x.grad + epsi * z
            y_new = f(x_new)
            q_up = multivariate_normal.pdf(x.numpy(), mean=x_new.numpy(), cov=eye)
            q_down = multivariate_normal.pdf(x_new.numpy(), mean=x.numpy(), cov=eye)
            a = np.min([1.0, y_new.numpy() * q_up / y.numpy() / q_down])
            u = np.random.rand()
            x = x_new
            if u < a:
                yield x.numpy(), cal_count

def langevin_adjusted_annealing(f, epsi, x_0):
    def sin_func(time_step):
        return 1 + 0.5 * np.cos(time_step / 500)
    x = torch.tensor(x_0, requires_grad=True)
    eye = np.eye(len(x_0))
    cal_count = 0
    while 1:
        scaling = sin_func(cal_count)
        cal_count += 1
        x.requires_grad_(True)
        y = f(x)
        out = torch.log(y)
        out.backward()
        z = torch.normal(0, 1, x.shape)
        with torch.no_grad():
            temp_epsi = epsi 
            temp_eye = eye * temp_epsi
            x_new = x + temp_epsi ** 2 / 2 * x.grad + temp_epsi * z
            y_new = f(x_new)
            q_up = multivariate_normal.pdf(x.numpy(), mean=x_new.numpy(), cov=temp_eye)
            q_down = multivariate_normal.pdf(x_new.numpy(), mean=x.numpy(), cov=temp_eye)
            a = np.min([1.0, y_new.numpy() * q_up / y.numpy() / q_down])
            u = np.random.rand()
            x = x_new
            if u < a:
                yield x.numpy(), cal_count