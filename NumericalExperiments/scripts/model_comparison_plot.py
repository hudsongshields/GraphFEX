import matplotlib.pyplot as plt
import numpy as np


HR_fex_smapes = list(reversed([0.011587, 0.258068, 1.377027]))
HR_scindy_smapes = [16.056085742169948, 3.8690776529165825, 0.8521754253334978]
HR_two_phase_smapes = [140.86607235074248/2, 99.25732088318102/2, 104.62122917017113/2, 104.4325547226511/2, 59.34508384679936/2, 100.25841759459676/2, 0.006341143304284227/2] # fake numbers

Lorenz_fex_smapes = list(reversed([0.007000787597725743, 0.16296467509479837, 4.474004358155896]))
Lorenz_scindy_smapes = [0.28918069440828326, 0.005000250012497853, 0.0]
Lorenz_two_phase_smapes = [1.4267456638211855/2, 0.4158661870302396/2, 0.1453436609323649/2, 0.043153394181105295/2, 0.021661687693803618/2, 0.005912044840026999/2, 0.0034173545975779065/2] # fake numbers

Kuramoto_fex_smapes = [2.3, 1.5, 0.7] # fake numbers
Kuramoto_scindy_smapes = [1.8, 1.2, 0.5] # fake numbers
Kuramoto_two_phase_smapes = [1.0, 0.7, 0.3] # fake numbers

vortex_fex_smapes = [1.5, 1.0, 0.5] # fake numbers
vortex_scindy_smapes = [1.2, 0.8, 0.4] # fake numbers
vortex_two_phase_smapes = [0.9, 0.6, 0.3] # fake numbers

snr_levels = [30, 45, 60]
two_phase_snr_levels = [30, 35, 40, 45, 50, 55, 60]

fig, ax = plt.subplots(2, 2, figsize=(8, 6))
HR_plot = ax[0, 0]
Lorenz_plot = ax[0, 1]
Kuramoto_plot = ax[1, 0]
vortex_plot = ax[1, 1]

HR_plot.plot(snr_levels, HR_fex_smapes, 'o-', label='FEX', linewidth=1)
HR_plot.plot(snr_levels, HR_scindy_smapes, 'o-', label='SINDy', linewidth=1)
HR_plot.plot(two_phase_snr_levels, HR_two_phase_smapes, 'o-', label='Two-Phase', linewidth=1)
HR_plot.set_title(r'(a) Hindmarsh Rose')
HR_plot.set_xlabel('SNR Noise Level')
HR_plot.set_ylabel('sMAPE')
HR_plot.legend()

Lorenz_plot.plot(snr_levels, Lorenz_fex_smapes, 'o-', label='FEX', linewidth=1)
Lorenz_plot.plot(snr_levels, Lorenz_scindy_smapes, 'o-', label='SINDy', linewidth=1)
Lorenz_plot.plot(two_phase_snr_levels, Lorenz_two_phase_smapes, 'o-', label='Two-Phase', linewidth=1)
Lorenz_plot.set_title(r'(b) Lorenz Attractor')
Lorenz_plot.set_xlabel('SNR Noise Level')
Lorenz_plot.set_ylabel('sMAPE')
Lorenz_plot.legend()

Kuramoto_plot.plot(snr_levels, Kuramoto_fex_smapes, 'o-', label='FEX', linewidth=1)
Kuramoto_plot.plot(snr_levels, Kuramoto_scindy_smapes, 'o-', label='SINDy', linewidth=1)
Kuramoto_plot.plot(snr_levels, Kuramoto_two_phase_smapes, 'o-', label='Two-Phase', linewidth=1)
Kuramoto_plot.set_title(r'(c) Kuramoto Model')
Kuramoto_plot.set_xlabel('SNR Noise Level')
Kuramoto_plot.set_ylabel('sMAPE')
Kuramoto_plot.legend()

vortex_plot.plot(snr_levels, vortex_fex_smapes, 'o-', label='FEX', linewidth=1)
vortex_plot.plot(snr_levels, vortex_scindy_smapes, 'o-', label='SINDy', linewidth=1)
vortex_plot.plot(snr_levels, vortex_two_phase_smapes, 'o-', label='Two-Phase', linewidth=1)
vortex_plot.set_title(r'(d) Vortex Dynamics')
vortex_plot.set_xlabel('SNR Noise Level')
vortex_plot.set_ylabel('sMAPE')
vortex_plot.legend()


plt.tight_layout()
plt.show()
