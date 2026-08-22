import torch
import numpy as np
import matplotlib.pyplot as plt
from .plot import rk4_step
from .generate_data import make_adjacency, make_data


def lyapunov_growth(
    initial_state,
    timesteps,
    dt,
    adj_matrix,
    snr,
    epsilon=1e-4,
    seed=42,
):
    torch.manual_seed(seed)

    reference = initial_state.clone().detach()

    # Random perturbation in full 3N-dimensional state space
    perturbation = torch.randn_like(reference)

    # Force perturbation to have magnitude epsilon
    perturbation /= torch.linalg.vector_norm(perturbation)
    perturbation *= epsilon

    perturbed = reference + perturbation

    # Actual initial separation
    delta_0 = torch.linalg.vector_norm(perturbed - reference).item()

    times = []
    log_separations = []

    with torch.no_grad():

        for step in range(1, timesteps + 1):

            # Evolve both trajectories independently
            reference = rk4_step(reference, dt, adj_matrix, snr=snr)
            perturbed = rk4_step(perturbed, dt, adj_matrix, snr=snr)

            delta = torch.linalg.vector_norm(perturbed - reference).item()

            if delta <= 0:
                continue

            T = step * dt

            log_separation = np.log(delta / delta_0)
            times.append(T)
            log_separations.append(log_separation)

    return np.asarray(times), np.asarray(log_separations)

def lyapunov_trials(
    initial_state,
    timesteps,
    dt,
    adj_matrix,
    snr,
    n_trials=10,
    epsilon=1e-4,
):

    all_growths = []
    times = None

    for trial in range(n_trials):

        times, growth = lyapunov_growth(
            initial_state=initial_state,
            timesteps=timesteps,
            dt=dt,
            adj_matrix=adj_matrix,
            snr=snr,
            epsilon=epsilon,
            seed=42 + trial,
        )

        all_growths.append(growth)

    all_growths = np.asarray(all_growths)
    mean_growth = np.mean(all_growths, axis=0)
    std_growth = np.std(all_growths, axis=0)

    return (times, mean_growth, std_growth)



def main():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    timesteps = 5000
    dt = 0.01
    
    adj_matrix = make_adjacency(100, 3)
    timeseries, _ = make_data(num_samples=timesteps, adjacency=adj_matrix, snr=None, coupling=0.8)
    initial_state = timeseries[0].clone()

    snrs = [None, 30, 45, 60]

    fig, ax = plt.subplots(figsize=(6, 3.2))

    for snr in snrs:

        times, mean_growth, std_growth = lyapunov_trials(
            initial_state=initial_state,
            timesteps=timesteps,
            dt=dt,
            adj_matrix=adj_matrix,
            snr=snr,

            n_trials=10,
            epsilon=1e-4,
        )

        if snr:
            line, = ax.plot(times, mean_growth, label=(rf"SNR = {snr} dB"))
        else:
            line, = ax.plot(times, mean_growth, label=(rf"Ground Truth"), color='black')

        ax.fill_between(
            times,
            mean_growth - std_growth,
            mean_growth + std_growth,
            alpha=0.15,
            color=line.get_color(),
        )

    ax.set_title(r"Lyapunov Divergence")
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$\log\left(\frac{\delta(T)}{\delta(0)}\right)$")

    ax.legend()
    fig.tight_layout()
    fig.savefig(
        "NumericalExperiments/Lorenz/"
        "lorenz_lyapunov_divergence.png",
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()