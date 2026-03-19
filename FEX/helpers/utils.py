from argparse import __all__
import torch

def NumericalDeriv(timeseries, dt):
    dx_dt = torch.zeros_like(timeseries)
    dx_dt = dx_dt[2:-2] # subtract 4 from time dimension for 4th order method
    for dim in range(timeseries.shape[2]):
        x_dim = timeseries[:, :, dim]

        x_PlusTwo = x_dim[4:, :]
        x_MinusTwo = x_dim[:-4, :]

        x_PlusOne = x_dim[3:-1, :]
        x_MinusOne = x_dim[1:-3, :]

        dim_dxdt = (x_MinusTwo - 8 * x_MinusOne + 8 * x_PlusOne - x_PlusTwo) / (12 * dt)
        dx_dt[:, :, dim] = dim_dxdt
    return dx_dt