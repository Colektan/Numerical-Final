from random_generator import *
from matplotlib import pyplot as plt
from scipy.integrate import quad
import numpy as np
from tqdm import tqdm
import time

def BoxMuller():
    F_inv = lambda x : np.sqrt(-2*np.log(x))
    r = inverse_sample(F_inv)
    rad = np.random.rand() * 2 * np.pi

    x1 = r * np.cos(rad)
    x2 = r * np.sin(rad)

    return (x1, x2)


class test_pdf_1:
    def __init__(self, mean=None, cov=None):
        if mean is None:
            self.mean = [0, 0]
        else:
            self.mean = mean
        if cov is None:
            self.cov = np.eye(2)
        else:
            self.cov = cov
    
    def f(self, x):
        return multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)

class test_pdf_2:
    def __init__(self):
        self.mean1 = np.array([-1, -1])
        self.mean2 = np.array([1, 1])
        self.cov= np.array([[0.5, 0], [0, 0.1]])
        self.mv_1 = torch.distributions.MultivariateNormal(torch.tensor(self.mean1), torch.tensor(self.cov))
        self.mv_2 = torch.distributions.MultivariateNormal(torch.tensor(self.mean2), torch.tensor(self.cov))

    def f(self, x):
        if isinstance(x, torch.Tensor):
            a = torch.exp(self.mv_1.log_prob(x))
            b = torch.exp(self.mv_2.log_prob(x))
        else:
            a = multivariate_normal.pdf(x, mean=self.mean1, cov=self.cov)
            b = multivariate_normal.pdf(x, mean=self.mean2, cov=self.cov)
        return 0.5 * a + 0.5 * b

def plot_results_2D(sampled_points, pdf, title, t_spent):
    """
    sampled_points: [N, 2] numpy array
    """
    start, end = [-4, 4]
    plt.figure(figsize=(10, 10))

    # plot on x_axis
    plt.subplot(224)
    plt.xlabel("value of X")
    plt.ylabel("Frequency")
    plt.xlim([start, end])
    plt.title("Marginal Statistics on X")
    # hist
    plt.hist(sampled_points[:, 0], bins=20, range=[start, end], density=True, color='royalblue')
    # target marginal pdf
    def marginal_pdf_x(x):
        result, _ = quad(lambda y: pdf(np.array([x, y])), -50, 50)
        return result
    input_x = np.linspace(start, end, 50)
    values = [marginal_pdf_x(x) for x in input_x]
    plt.plot(input_x, values, color="lightcoral")

    # plot on y_axis
    plt.subplot(221)
    plt.ylabel("value of Y")
    plt.ylim([start, end])
    plt.xlabel("Frequency")
    plt.title("Marginal Statistics on Y")
    plt.hist(sampled_points[:, 1], bins=20, range=[start, end], orientation="horizontal", density=True, color='royalblue')
    # target marginal pdf
    def marginal_pdf_y(y):
        result, _ = quad(lambda x: pdf(np.array([x, y])), -50, 50)
        return result
    input_x = np.linspace(start, end, 50)
    values = [marginal_pdf_y(x) for x in input_x]
    plt.plot(values, input_x, color="lightcoral")

    # plot on 2D samples
    plt.subplot(222)
    x = np.arange(start,end,0.1)
    y = np.arange(start,end,0.1)
    X,Y = np.meshgrid(x,y)
    Z = pdf(np.dstack((X, Y)))
    N = np.linspace(0, pdf([1,1]), 10)
    CS = plt.contourf(x, y, Z, N, cmap="GnBu")
    # plt.colorbar(CS)
    plt.scatter(sampled_points[:, 0],  sampled_points[:, 1], s=1, color='k')
    plt.title("Time spent per data point: %.5f ms" % t_spent)
    
    plt.suptitle(title)
    plt.savefig(f"{title}.svg")

def test_BoxMuller():
    mv = test_pdf_1()
    
    sampled_points = []
    
    N = 2000
    for i in range(N):
        sampled_points.append(BoxMuller())
    sampled_points = np.array(sampled_points)

    t_start = time.time()
    N = 100000
    for i in range(N):
        BoxMuller()
    t_end = time.time()
    t_spent = (t_end - t_start) / N * 1000 # unit: ms

    plot_results_2D(sampled_points, mv.f, "The result of Box-Muller Sampling", t_spent)

def test_reject_sample_1():
    mv = test_pdf_2()
    def kg(x):
        return multivariate_normal.pdf(x, mean=[0, 0], cov=np.eye(2)) * 11
    
    def kg_sample():
        return multivariate_normal.rvs(mean=[0, 0], cov=np.eye(2))
    
    gen = reject_sample(mv.f, kg, kg_sample)
    N = 1000
    sampled_points = []
    for i in tqdm(range(N)):
        p, _ = next(gen)
        sampled_points.append(p)
    sampled_points = np.array(sampled_points)

    gen = reject_sample(mv.f, kg, kg_sample)
    N = 50000
    t_start = time.time()
    for i in range(N):
        _, cal_count = next(gen)
    t_end = time.time()
    t_spent = (t_end - t_start) / N * 1000 # unit: ms

    print(f"Total calculate count with small var: {cal_count}")
    plot_results_2D(sampled_points, mv.f, "The result of Reject Sampling(Small Var)", t_spent)
 
def test_reject_sample_2():
    mv = test_pdf_2()
    def kg(x):
        return multivariate_normal.pdf(x, mean=[0, 0], cov=np.eye(2)*4) * 12
    
    def kg_sample():
        return multivariate_normal.rvs(mean=[0, 0], cov=np.eye(2)*4)
    
    gen = reject_sample(mv.f, kg, kg_sample)
    N = 1000
    sampled_points = []
    for i in tqdm(range(N)):
        p, _ = next(gen)
        sampled_points.append(p)
    sampled_points = np.array(sampled_points)

    gen = reject_sample(mv.f, kg, kg_sample)
    N = 50000
    t_start = time.time()
    for i in range(N):
        _, cal_count = next(gen)
    t_end = time.time()
    t_spent = (t_end - t_start) / N * 1000 # unit: ms
    
    print(f"Total calculate count with large var: {cal_count}")
    plot_results_2D(sampled_points, mv.f, "The result of Reject Sampling(Large Var)", t_spent)

def test_langevin():
    mv = test_pdf_2()
    
    epsi = 0.4
    x = np.random.rand(2)
    gen = langevin(mv.f, epsi, x)
    N = 1000
    sampled_points = []
    for i in tqdm(range(N)):
        p = next(gen)
        if np.isnan(p).any():
            break
        sampled_points.append(p)
    sampled_points = np.array(sampled_points)

    N = 50000
    t_start = time.time()
    gen = langevin(mv.f, epsi, x)
    for i in range(N):
        p = next(gen)
        if np.isnan(p).any():
            break
    t_end = time.time()
    t_spent = (t_end - t_start) / N * 1000 # unit: ms
    
    print("Time test complete.")
    plot_results_2D(sampled_points, mv.f, f"The result of Langevin Sampling(Epsi={epsi})", t_spent)

def test_langevin_adjusted():
    mv = test_pdf_2()
    
    epsi = 0.4
    x = np.random.rand(2)
    gen = langevin_adjusted(mv.f, epsi, x)
    N = 10000
    sampled_points = []
    for i in tqdm(range(N)):
        p, _ = next(gen)
        if np.isnan(p).any():
            break
        sampled_points.append(p)
    sampled_points = np.array(sampled_points)

    N = 5000
    t_start = time.time()
    gen = langevin_adjusted(mv.f, epsi, x)
    for i in range(N):
        _, cal_count = next(gen)
        if np.isnan(p).any():
            break
    t_end = time.time()
    t_spent = (t_end - t_start) / N * 1000 # unit: ms
    
    print(f"Total calculate count with adjusted langevin: {cal_count}")
    plot_results_2D(sampled_points, mv.f, f"The result of Langevin-Adjusted Sampling(Epsi={epsi})", t_spent)

def test_langevin_adjusted_anealing():
    mv = test_pdf_2()
    
    epsi = 0.5
    x = np.random.rand(2)
    gen = langevin_adjusted_annealing(mv.f, epsi, x)
    N = 1000
    sampled_points = []
    for i in tqdm(range(N)):
        p, _ = next(gen)
        if np.isnan(p).any():
            break
        sampled_points.append(p)
    sampled_points = np.array(sampled_points)

    N = 50
    t_start = time.time()
    gen = langevin_adjusted(mv.f, epsi, x)
    for i in range(N):
        _, cal_count = next(gen)
        if np.isnan(p).any():
            break
    t_end = time.time()
    t_spent = (t_end - t_start) / N * 1000 # unit: ms
    
    print(f"Total calculate count with adjusted langevin: {cal_count}")
    plot_results_2D(sampled_points, mv.f, f"The result of Langevin-Adjusted-Sin Sampling(Epsi={epsi})", t_spent)

if __name__ == "__main__":
    # test_BoxMuller()
    # test_reject_sample_1()
    # test_reject_sample_2()
    # test_langevin()
    test_langevin_adjusted()
    # test_langevin_adjusted_anealing()
