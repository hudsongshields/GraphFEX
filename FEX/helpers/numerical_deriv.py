import torch

def five_point(timeseries, dt):
    return ((
        timeseries[:-4, :, :]         # t-2
        - 8 * timeseries[1:-3, :, :]  # t-1
        + 8 * timeseries[3:-1, :, :]  # t+1
        - timeseries[4:, :, :]        # t+2
    ) / (12 * dt))

import numpy as np
import pysindy as ps
def smoothed_five_point(timeseries, dt):
    device = timeseries.device
    dtype = timeseries.dtype

    # convert timeseries to numpy for pysindy compatability
    x_np = timeseries.detach().cpu().numpy()

    smoothed_fd = ps.SmoothedFiniteDifference(
        smoother_kws={
            "window_length": 11,
            "polyorder": 3,
        },
        order=4,
        d=1,
        axis=0,
        is_uniform=True,
        drop_endpoints=True,
        save_smooth=True,
    )

    x_dot_np = smoothed_fd(x_np, dt)
    x_smooth_np = np.asarray(smoothed_fd.smoothed_x_)

    # drop enpoints in x
    x_smooth_np = x_smooth_np[2:-2]
    x_dot_np = x_dot_np[2:-2]

    # convert back to torch tensors
    x_smooth = torch.from_numpy(x_smooth_np).to(
        device=device,
        dtype=dtype,
    )
    x_dot = torch.from_numpy(x_dot_np).to(
        device=device,
        dtype=dtype,
    )

    return x_smooth, x_dot