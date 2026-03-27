from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from models import model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob


def block_lengths():
    section = 'east_coast_main_line'
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    block_lengths = data['distances']
    block_bearings = np.rad2deg(data['bearings'])
    block_number = np.arange(0, len(block_bearings), 1)

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])
    ax0.hist(block_bearings, bins=100)
    plt.show()


def misoperations_ws_framegen():
    section = 'west_coast_main_line'
    bearings = np.deg2rad(np.array([150, 330]))
    e_values = np.linspace(0, 20, 201)
    for bearing in bearings:
        ex_uni = e_values * np.cos(bearing)
        ey_uni = e_values * np.sin(bearing)
        axles_a_all = np.load(f'data/axle_positions/{section}_train_end_axles_midpoint_a.npy')
        axles_a = np.concatenate(axles_a_all[9::9, :])
        output = model(section_name=section, ex_uniform=ex_uni, ey_uniform=ey_uni, axle_pos_a=axles_a, y_trac=1.6)
        ia_all = output['i_relays_a']
        idx_occupied = np.argwhere((ia_all[:, 0] < 0.055) & (ia_all[:, 0] > -0.055))
        idx_unoccupied = np.argwhere((ia_all[:, 0] > 0.055) | (ia_all[:, 0] < -0.055))
        ia_occupied = ia_all[idx_occupied[:, 0]]
        ia_unoccupied = ia_all[idx_unoccupied[:, 0]]

        idx_frame = 0
        for ei in range(0, len(e_values)):
            plt.rcParams['font.size'] = '15'
            fig = plt.figure(figsize=(16, 8))
            gs = GridSpec(1, 1, bottom=0.25)
            ax0 = fig.add_subplot(gs[0])
            colors_unoccupied = np.where(((ia_unoccupied[:, ei] < 0.055) & (ia_unoccupied[:, ei] > -0.055)), 'orangered', 'white')
            edge_colors_unoccupied = np.where(((ia_unoccupied[:, ei] < 0.055) & (ia_unoccupied[:, ei] > -0.055)), 'firebrick', 'limegreen')
            colors_occupied = np.where(((ia_occupied[:, ei] > 0.081) | (ia_occupied[:, ei] < -0.081)), 'limegreen', 'white')
            edge_colors_occupied = np.where(((ia_occupied[:, ei] > 0.081) | (ia_occupied[:, ei] < -0.081)), 'seagreen', 'orangered')
            ax0.scatter(idx_unoccupied, ia_unoccupied[:, ei], facecolor=colors_unoccupied, edgecolors=edge_colors_unoccupied, s=15, marker='o')
            ax0.scatter(idx_occupied, ia_occupied[:, ei], facecolor=colors_occupied, edgecolors=edge_colors_occupied, s=15, marker='X')

            legend_elements = [Line2D([0], [0], linestyle='None', marker='o', label='Unoccupied Block', markerfacecolor='white', markeredgecolor='limegreen',  markersize=5),
                               Line2D([0], [0], linestyle='None', marker='o', label='Right Side Failure', markerfacecolor='orangered', markeredgecolor='firebrick',  markersize=5),
                               Line2D([0], [0], linestyle='None', marker='X', label='Occupied Block', markerfacecolor='white', markeredgecolor='orangered',  markersize=5),
                               Line2D([0], [0], linestyle='None', marker='X', label='Wrong Side Failure', markerfacecolor='limegreen', markeredgecolor='seagreen',  markersize=5),
                               Line2D([0], [0], linestyle='-', label='Drop-out Current', color='orangered'),
                               Line2D([0], [0], linestyle='--', label='Pick-up Current', color='limegreen')
                               ]
            ax0.axhline(0.055, color='orangered')
            ax0.axhline(-0.055, color='orangered')
            ax0.axhline(0.081, color='limegreen', linestyle='--')
            ax0.axhline(-0.081, color='limegreen', linestyle='--')
            ax0.set_ylim(-1, 1)
            ax0.set_xlim(-1, len(ia_all))
            ax0.set_xlabel('Block Number')
            ax0.set_ylabel('Relay Current (A)')
            ax0.legend(handles=legend_elements, ncol=3, fancybox=True, loc='center', bbox_to_anchor=(0.5, -0.2))
            if idx_frame != 0:
                ax0.set_title(f'Electric Field Strength: {np.round(e_values[ei], 1)} V/km')

            if bearing == np.deg2rad(150):
                if idx_frame < 10:
                    plt.savefig(f'frames/misoperation_ws/{section}_misoperation_ws_pos_00{idx_frame}.png')
                elif idx_frame > 99:
                    plt.savefig(f'frames/misoperation_ws/{section}_misoperation_ws_pos_{idx_frame}.png')
                else:
                    plt.savefig(f'frames/misoperation_ws/{section}_misoperation_ws_pos_0{idx_frame}.png')
                plt.close()
                idx_frame += 1
            if bearing == np.deg2rad(330):
                if idx_frame < 10:
                    plt.savefig(f'frames/misoperation_ws/{section}_misoperation_ws_neg_00{idx_frame}.png')
                elif idx_frame > 99:
                    plt.savefig(f'frames/misoperation_ws/{section}_misoperation_ws_neg_{idx_frame}.png')
                else:
                    plt.savefig(f'frames/misoperation_ws/{section}_misoperation_ws_neg_0{idx_frame}.png')
                plt.close()
                idx_frame += 1


def misoperations_ws_gif():
    section = 'east_coast_main_line'
    files = sorted(glob.glob(f'frames/misoperation_ws/{section}_misoperation_ws_pos_*.png'))
    frames = [Image.open(f) for f in files]
    frames[0].save(
        f'{section}_ws_pos.gif',
        save_all=True,
        append_images=frames[1:],
        duration=100,  # ms per frame
        loop=0)

    files = sorted(glob.glob(f'frames/misoperation_ws/{section}_misoperation_ws_neg_*.png'))
    frames = [Image.open(f) for f in files]
    frames[0].save(
        f'{section}_ws_neg.gif',
        save_all=True,
        append_images=frames[1:],
        duration=100,  # ms per frame
        loop=0)


#block_lengths()
#misoperations_rs()
#misoperations_ws_framegen()
#misoperations_ws_gif()
