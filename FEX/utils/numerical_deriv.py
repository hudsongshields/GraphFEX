import torch

def NumericalDeriv(timeseries, dt):
    return ((
        timeseries[:-4, :, :]         # t-2
        - 8 * timeseries[1:-3, :, :]  # t-1
        + 8 * timeseries[3:-1, :, :]  # t+1
        - timeseries[4:, :, :]        # t+2
    ) / (12 * dt))