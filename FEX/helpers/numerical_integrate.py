import torch

# x0: x0, y0, z0, ...
# func: f(x, t) -> dx/dt
def NumericalIntegrate(func, x0: torch.Tensor, dt: float, t: float, method: str = 'euler'):
    shape = (t, x0.shape[0])
    xt = torch.zeros(shape, device=x0.device)
    xt[0] = x0
    for i in range(1, t):
        if method == 'euler':
            xt[i] = xt[i-1] + func(xt[i-1], (i-1) * dt) * dt
        elif method == 'rk4':
            k1 = func(xt[i-1], (i-1)*dt) * dt
            k2 = func(xt[i-1] + 0.5 * k1, (i-1)*dt + 0.5 * dt) * dt
            k3 = func(xt[i-1] + 0.5 * k2, (i-1)*dt + 0.5 * dt) * dt
            k4 = func(xt[i-1] + k3, (i-1)*dt + dt) * dt
            xt[i] = xt[i-1] + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        else:
            raise ValueError(f"Unknown integration method: {method}")
    return xt