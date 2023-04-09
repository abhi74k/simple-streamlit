import numpy as np
import cvxpy as cp
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

class TimeSeriesCurveFitter:

    def __init__(self, data, smoothening_weight = 1, period=365, ar_order=5):
        self.m = data.shape[0]
        self.data = data.reshape(-1)
        self.smoothening_weight = smoothening_weight
        self.period = period
        self.ar_order = ar_order
        self.residual = np.zeros_like(self.data)
        self.final_estimate = np.zeros_like(self.data)

    def __fit_cyclic(self, train_data):
        m = train_data.shape[0]
        lam = self.smoothening_weight
        period = self.period

        ones = np.ones(m)
        L = sp.sparse.spdiags([-ones, ones], np.array([0, 1]), m-1, m).toarray()

        s = cp.Variable(m)

        #Define objective function
        obj1 = cp.sum_squares(s - train_data)
        obj2 = cp.sum_squares(L @ s)
        obj = obj1 + lam * obj2

        constraints = []
        for i in range(period, m):
            constraints.append(s[i] == s[i-period])

        # Solve the optimization problem
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve()
        
        return s.value
    
    def __fit_trend(self, train_data):
        m = train_data.shape[0]
        ar_order = self.ar_order
        theta = cp.Variable(self.ar_order)
        
        A = np.zeros((m-ar_order, ar_order))

        for i in range(ar_order):
            A[:, i] = train_data[i:i+m-ar_order]

        # Least squares problem
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ theta - train_data[ar_order:])))
        prob.solve()

        trend_pred = A @ theta.value
        trend = train_data.copy()
        trend[ar_order:] = trend_pred

        return trend
        
    def fit(self):
        train_data = self.data
        self.cyclic = self.__fit_cyclic(train_data)
        self.trend = self.__fit_trend(train_data - self.cyclic)

        self.final_estimate = self.trend + self.cyclic
        self.residual = train_data - self.trend - self.cyclic
        
        rms = np.sqrt(np.mean((train_data - self.final_estimate)**2))

        return self.final_estimate, self.residual, \
            self.trend, self.cyclic, rms
    
    def generate_plots(self):
        data = self.data
        cyclic = self.cyclic
        trend = self.trend
        final_estimate = self.final_estimate
        residual = self.residual

        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(data, label='train_data')
        plt.plot(cyclic, label='cyclic')
        plt.title('Cyclic Component')
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(data[:256] - cyclic[:256], label='train_data - cyclic')
        plt.plot(trend[:256], label='trend')
        plt.title(f'Trend Component(ar_order={self.ar_order})')
        plt.legend()
        plt.subplot(2, 2, 3)
        plt.plot(data, label='train_data')
        plt.plot(final_estimate, label='trend + cyclic')
        plt.title('Final fit')
        plt.legend()
        plt.subplot(2, 2, 4)
        plt.plot(residual)
        plt.title('Residual')
        plt.legend()

        return plt
    
if __name__ == "__main__":
    

    data = pd.DataFrame(np.loadtxt('temperature.txt'))
    data = np.array(data).reshape(-1)

    smoothing_weight = 50
    ar_order = 1
    period = 350

    decomposer = TimeSeriesCurveFitter(data, period=period, smoothening_weight=smoothing_weight, ar_order=ar_order)
    final_estimate, residual, trend, cyclic, rms = decomposer.fit(data)

    print(f'Root Mean Square Error: {rms}')

    plt = decomposer.generate_plots()
    plt.show()