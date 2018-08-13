import numpy as np
from data import *

def GM_pdf(x, mu, sigma):
    coef = ((2 * np.pi) ** (-x.shape[1]/2)) * (np.linalg.det(sigma) ** (-0.5))
    ind = -0.5 * (x - mu) * sigma.I * (x - mu).T
    return coef * np.exp(ind)

class GMM:
            
    def __init__(self, x, K = 4, round = 500, eps=1e-15):
        # x.shape: [#data, #dim]
        n = x.shape[0]
        D = x.shape[1]
        # mu.shape: [K, #dim]
        self.K = K
        self.mu = np.mat(np.random.random((K, D)) * np.mean(x, axis=0))
        x = np.mat(x)
        self.x = x
        # sigma.shape: [K, #dim, #dim]
        self.sigma = [np.mat(np.eye(D)) for _ in range(K)]
        # pi.shape: [K]
        self.pi = np.random.random(K)
        self.pi = self.pi / np.sum(self.pi)
        # phi.shape: [K, #data]
        # gamma.shape: [K, #data]
        phi = np.zeros((K, n))
        gamma = np.zeros((K, n))
        last_pi = np.random.random(K)
        # until converge or max round
        for r in range(round):
            print("STEP: ", r)
            print("Distribution: ", self.pi)
            # s.shape: [#data]
            s = np.zeros(n)
            # E-step
            for k in range(K):
                for j in range(n):
                    phi[k, j] = GM_pdf(x[j], self.mu[k], self.sigma[k])
                    gamma[k, j] = phi[k, j] * self.pi[k]
                    s[j] += gamma[k, j]
            for k in range(K):
                for j in range(n):
                    gamma[k, j] /= s[j]
            # M-step
            # s.shape: [K]
            s = np.zeros(K)
            # update mu
            for k in range(K):
                self.mu[k, :] = 0
                for j in range(n):
                    self.mu[k] += gamma[k, j] * x[j]
                    s[k] += gamma[k, j]
            for k in range(K):
                self.mu[k] /= s[k]
            # update sigma
            for k in range(K):
                self.sigma[k][:, :] = 0
                for j in range(n):
                    self.sigma[k] += (x[j] - self.mu[k]).T * (x[j] - self.mu[k]) * gamma[k, j]
            for k in range(K):
                self.sigma[k] /= s[k]
            # update pi
            for k in range(K):
                self.pi[k] = s[k] / n

            diff = np.mean(np.abs(last_pi - self.pi))
            print(diff)
            if diff < eps:
                break
            last_pi[:] = self.pi


    def predict(self, data):
        if not isinstance(data, np.matrix):
            data = np.mat(data)
        rate = np.zeros(data.shape[0])
        for k in range(self.K):
            for j in range(data.shape[0]):
                rate[j] += GM_pdf(data[j], self.mu[k], self.sigma[k]) * self.pi[k]
        return rate