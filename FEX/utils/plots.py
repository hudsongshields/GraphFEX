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