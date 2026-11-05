import pickle
from pathlib import Path

import h5py
import torch

from FEX.helpers import numerical_deriv


DATA_DIR = Path(__file__).parent
DEFAULT_DT = 5.0 / 60.0


def _sensor_id(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def make_data(
    data_dir: str | Path = DATA_DIR,
    dt: float = DEFAULT_DT,
    timesteps: int | None = None,
    smoothing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load METR-LA speeds and return adjacency, speeds, and derivatives.

    Speeds are in mph. With the default five-minute timestep, derivatives are in
    mph per hour. The derivative drops the first and last two samples.
    If ``timesteps`` is set, only the first requested raw samples are returned.
    If ``smoothing`` is True, a smoothed finite difference (Savitzky-Golay
    window of 11) is used and the returned speeds are the smoothed speeds;
    otherwise the unsmoothed five-point derivative is used.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    min_timesteps = 11 if smoothing else 5
    if timesteps is not None and timesteps < min_timesteps:
        raise ValueError(f"timesteps must be at least {min_timesteps} for the derivative")

    data_dir = Path(data_dir)
    with (data_dir / "adj_mx.pkl").open("rb") as adjacency_file:
        sensor_ids, _, adjacency = pickle.load(adjacency_file, encoding="latin1")

    with h5py.File(data_dir / "metr-la.h5", "r") as speed_file:
        dataset = speed_file["df"]
        speeds = dataset["block0_values"][:]
        speed_sensor_ids = [_sensor_id(sensor_id) for sensor_id in dataset["block0_items"][:]]

    sensor_ids = [_sensor_id(sensor_id) for sensor_id in sensor_ids]
    if len(set(speed_sensor_ids)) != len(speed_sensor_ids):
        raise ValueError("METR-LA speed data contains duplicate sensor IDs")
    speed_indices = {sensor_id: index for index, sensor_id in enumerate(speed_sensor_ids)}
    missing_sensors = [sensor_id for sensor_id in sensor_ids if sensor_id not in speed_indices]
    if missing_sensors:
        raise ValueError(
            f"METR-LA speed data is missing {len(missing_sensors)} adjacency sensors"
        )

    if timesteps is not None:
        if timesteps > speeds.shape[0]:
            raise ValueError(
                f"requested {timesteps} timesteps, but only {speeds.shape[0]} are available"
            )
        speeds = speeds[:timesteps]

    adjacency = torch.as_tensor(adjacency, dtype=torch.float32)

    speeds = torch.as_tensor(speeds[:, [speed_indices[sensor_id] for sensor_id in sensor_ids]])
    speeds = speeds.to(dtype=torch.float32).unsqueeze(-1)
    if smoothing:
        timeseries, derivatives = numerical_deriv.smoothed_five_point(speeds, dt=dt)
    else:
        timeseries, derivatives = numerical_deriv.five_point(speeds, dt=dt)

    return adjacency, timeseries, derivatives
