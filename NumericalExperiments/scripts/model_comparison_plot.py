import matplotlib.pyplot as plt
import numpy as np

HR_fex_smapes = [1.377027/2, 0.258068/2, 0.4967490093283841, 0.011587/2]
HR_sindy_smapes = [13.56496143173078, 0.7376175902424891, 22.199659228668253, 0.10130996412087195, 0.0] # removed 22.199659228668253 at snr 52.5
HR_two_phase_smapes = [38.79107516418194, 52.269179100617485, 0.015848390263636023, 0.0020, 25.7160]

Lorenz_fex_smapes = list(reversed([0.007000787597725743/2, 0.16296467509479837/2, 2.0655122331675804/2]))
Lorenz_sindy_smapes = [0.28918069440828326/2, 0.005000250012497853/2, 0.0/2]
Lorenz_two_phase_smapes = [1.4267456638211855/2, 0.4158661870302396/2, 0.1453436609323649/2, 0.043153394181105295/2, 0.021661687693803618/2, 0.005912044840026999/2, 0.0034173545975779065/2]

stuart_fex_smapes = [0.18771058303669375, 0.052805560825567185, 0.024479846989509727, 0.04102773549228936, 0.009340769075084967]
stuart_sindy_smapes = [26.424291509488906, 7.979116327660904, 3.1155130896345486, 1.9256002101887089, 0.5661781426376994]
stuart_two_phase_smapes = [] 

fractional_fex_smapes = [] # fake numbers
fractional_sindy_smapes = [] # fake numbers
fractional_two_phase_smapes = [] # fake numbers

snr_levels = [30, 45, 60]
two_phase_snr_levels = [30, 35, 40, 45, 50, 55, 60]
new_snr_levels = [37.5, 45, 52.5, 60, 67.5]


fig, ax = plt.subplots(2, 2, figsize=(8, 6))
plt.xticks(two_phase_snr_levels)
HR_plot = ax[0, 0]
Lorenz_plot = ax[0, 1]
Kuramoto_plot = ax[1, 0]
vortex_plot = ax[1, 1]

hr_temp_arr = [30, 45, 52.5, 60]
sindy_temp_arr = [37.5, 45, 60, 67.5]
HR_plot.plot(hr_temp_arr, HR_fex_smapes, 'o-', label='FEX', linewidth=1)
HR_plot.plot(new_snr_levels, HR_sindy_smapes, 'o-', label='SINDy', linewidth=1)
HR_plot.plot(new_snr_levels, HR_two_phase_smapes, 'o-', label='Two-Phase', linewidth=1)
HR_plot.set_title(r'(a) Hindmarsh Rose')
HR_plot.set_xlabel('SNR Noise Level')
HR_plot.set_ylabel('sMAPE')
HR_plot.legend()

Lorenz_plot.plot(snr_levels, Lorenz_fex_smapes, 'o-', label='FEX', linewidth=1)
Lorenz_plot.plot(snr_levels, Lorenz_sindy_smapes, 'o-', label='SINDy', linewidth=1)
Lorenz_plot.plot(two_phase_snr_levels, Lorenz_two_phase_smapes, 'o-', label='Two-Phase', linewidth=1)
Lorenz_plot.set_title(r'(b) Lorenz Attractor')
Lorenz_plot.set_xlabel('SNR Noise Level')
Lorenz_plot.set_ylabel('sMAPE')
Lorenz_plot.legend()

Kuramoto_plot.plot(new_snr_levels, stuart_fex_smapes, 'o-', label='FEX', linewidth=1)
Kuramoto_plot.plot(new_snr_levels, stuart_sindy_smapes, 'o-', label='SINDy', linewidth=1)
"""
Kuramoto_plot.plot(new_snr_levels, stuart_two_phase_smapes, 'o-', label='Two-Phase', linewidth=1)
"""
Kuramoto_plot.set_title(r'(c) Stuart-Landau Model')
Kuramoto_plot.set_xlabel('SNR Noise Level')
Kuramoto_plot.set_ylabel('sMAPE')
# Kuramoto_plot.legend()

"""vortex_plot.plot(snr_levels, fractional_fex_smapes, 'o-', label='FEX', linewidth=1)
vortex_plot.plot(snr_levels, fractional_sindy_smapes, 'o-', label='SINDy', linewidth=1)
vortex_plot.plot(snr_levels, fractional_two_phase_smapes, 'o-', label='Two-Phase', linewidth=1)"""
vortex_plot.set_title(r'(d) Fractional non-local pde Dynamics')
vortex_plot.set_xlabel('SNR Noise Level')
vortex_plot.set_ylabel('sMAPE')
# vortex_plot.legend()


plt.tight_layout()
plt.show()
