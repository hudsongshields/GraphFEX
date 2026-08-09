import matplotlib.pyplot as plt

def plot_dynamics(true_x, true_y, true_z, predicted_states, *, elev=30, azim=45):
    fig = plt.figure(figsize=(24, 6))
    gs = fig.add_gridspec(3, 3)
    ax1 = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax2 = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    ax3 = fig.add_subplot(gs[:, 2], projection='3d')
    ax1[0].plot(true_x, color='blue')
    ax1[0].set_ylabel(r'$x_{i,1}$')
    ax1[1].plot(true_y, color='blue')
    ax1[1].set_ylabel(r'$x_{i,2}$') 
    ax1[2].plot(true_z, color='blue')
    ax1[2].set_ylabel(r'$x_{i,3}$') 
    ax1[2].set_title('(a) True Dynamics', loc='center', y=-0.5)
    ax1[2].set_xlabel('Time')

    ax2[0].plot(predicted_states[:, 0], color='red')
    ax2[1].plot(predicted_states[:, 1], color='red')
    ax2[2].plot(predicted_states[:, 2], color='red')
    ax2[2].set_title('(b) FEX Dynamics', loc='center', y=-0.5)
    ax2[2].set_xlabel('Time')

    ax3.plot(true_x, true_y, true_z, label='True Dynamics', color='blue')
    ax3.plot(predicted_states[:, 0], predicted_states[:, 1], predicted_states[:, 2], label='FEX', linestyle='--', color='red')
    ax3.legend()
    ax3.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax3.set_xlabel(r'$x_{i,1}$')
    ax3.set_ylabel(r'$x_{i,2}$')
    ax3.zaxis.set_rotate_label(False) 
    ax3.set_zlabel(r'$x_{i,3}$', rotation=90)
    ax3.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax3.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax3.view_init(elev=elev, azim=azim)
    ax3.set_box_aspect(None, zoom=0.85)
    ax3.set_title('(c) 3D Visualization', loc='center', y=-0.156)
    return fig



plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",

    "font.size": 20,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})


def plot_panel(nonePred, noneTrue, snr60Pred, snr60True, snr45Pred, snr45True, snr30Pred, snr30True, *, elev=30, azim=45):
    fig = plt.figure(figsize=(12, 7.0))
    gs_outline = fig.add_gridspec(2, 2, wspace=0.18, hspace=0.1)
    snr_labels = [r'(a)', r'(b)', r'(c)', r'(d)']
    data_pairs = [(nonePred, noneTrue), (snr60Pred, snr60True), (snr45Pred, snr45True), (snr30Pred, snr30True)]

    for i, (predicted_states, true_states) in enumerate(data_pairs):

        inner_sq = gs_outline[i // 2, i % 2].subgridspec(
            3, 2,
            width_ratios=[1.0, 0.7],
            wspace=0.05,
            hspace=0.08
        )
        ax1 = [fig.add_subplot(inner_sq[i, 0]) for i in range(3)]
        ax2 = fig.add_subplot(inner_sq[:, 1], projection='3d')
        ax1[0].plot(true_states[:, 0], color='blue')
        ax1[0].set_ylabel(r'$x_{i,1}$')
        ax1[0].set_xticks([])  # Hide x-ticks for the first two plots
        ax1[1].set_xticks([])  # Hide x-ticks for the
        ax1[1].plot(true_states[:, 1], color='blue')
        ax1[1].set_ylabel(r'$x_{i,2}$') 
        ax1[2].plot(true_states[:, 2], color='blue')
        ax1[2].set_ylabel(r'$x_{i,3}$') 
        if i < 2:
            ax1[2].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        else:
            ax1[2].set_xlabel(r'Time')

        ax1[0].set_title(snr_labels[i], loc='left', y=0.71, x=-0.29, fontweight='bold') # top left corner
        # align x-dim labels (ylabels for the 3 plots)
        for ax in ax1: 
            ax.label_outer()

        ax1[0].plot(predicted_states[:, 0], color='red', linestyle='--')
        ax1[1].plot(predicted_states[:, 1], color='red', linestyle='--')
        ax1[2].plot(predicted_states[:, 2], color='red', linestyle='--')

        ax2.plot(true_states[:, 0], true_states[:, 1], true_states[:, 2], label=r'True Dynamics', color='blue')
        ax2.plot(predicted_states[:, 0], predicted_states[:, 1], predicted_states[:, 2], label=r'FEX', linestyle='--', color='red')
        ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax2.set_xlabel(r'$x_{i,1}$')
        ax2.set_ylabel(r'$x_{i,2}$')
        ax2.zaxis.set_rotate_label(False) 
        ax2.set_zlabel(r'$x_{i,3}$', rotation=90)
        ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax2.view_init(elev=elev, azim=azim)

        # remove tick labels and lines
        ax2.set_axis_off()
        ax2.set_box_aspect(None, zoom=1.15)

        

    fig.legend(
        labels=[r'True Dynamics', r'FEX'], 
        loc='center', 
        bbox_to_anchor=(0.5, 0.05),
        ncol=1
    )

    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.16)
    fig.savefig("HR/test_panel_dynamics.png", bbox_inches="tight")
    return fig
