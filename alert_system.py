import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from models import model
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import re


def alert_map_geographic(e, index):
    e = np.round(e, 1)
    markersize = 50
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(3, 4, wspace=0.34)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1:])
    ax2 = fig.add_subplot(gs[1, 1:])
    ax3 = fig.add_subplot(gs[2, 1:])
    ax0.set_xlabel('Longitude')
    ax0.set_ylabel('Latitude')
    axs = [ax0, ax1, ax2, ax3]
    n_yellow = 0
    n_orange = 0
    n_red = 0

    for n_line in [1, 2, 3]:
        if n_line == 1:
            sec = 'glasgow_edinburgh_falkirk'
        if n_line == 2:
            sec = 'west_coast_main_line'
        if n_line == 3:
            sec = 'east_coast_main_line'

        data = np.load(f'data/axle_positions/{sec}_train_end_axles_midpoint_a.npy')[4:-4:8]
        axle_pos_a = np.concatenate(data)
        data = model(section_name=sec, axle_pos_a=axle_pos_a, axle_pos_b=[], ex_uniform=np.array([e]), ey_uniform=np.array([e]))
        ia = np.concatenate(data['i_relays_a'])

        data = np.load(f'data/rail_data/{sec}/{sec}_block_lons_lats.npz')
        lons = data['lons']
        lats = data['lats']
        ax0.scatter(lons[0], lats[0], c='lightgrey', edgecolors='grey')
        ax0.scatter(lons[-1], lats[-1], c='lightgrey', edgecolors='grey')

        data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
        block_lengths = data['distances']
        block_lengths_sum = np.cumsum(block_lengths)

        occupied_blocks = np.searchsorted(block_lengths_sum, axle_pos_a, side='right')
        occupied_blocks = np.unique(occupied_blocks)
        unoccupied_blocks = np.delete(np.arange(0, len(ia)), occupied_blocks)
        ia_occupied = ia[occupied_blocks]
        ia_unoccupied = ia[unoccupied_blocks]

        rs_yellow_locs = np.where((ia_unoccupied < 0.15) & (ia_unoccupied > -0.15))[0]
        rs_orange_locs = np.where((ia_unoccupied < 0.081) & (ia_unoccupied > -0.081))[0]
        rs_fail_locs = np.where((ia_unoccupied < 0.055) & (ia_unoccupied > -0.055))[0]
        rs_normal_locs = np.delete(np.arange(0, len(unoccupied_blocks)), rs_fail_locs)

        ws_yellow_locs = np.where((ia_occupied > 0.04) | (ia_occupied < -0.04))[0]
        ws_orange_locs = np.where((ia_occupied > 0.06) | (ia_occupied < -0.06))[0]
        ws_fail_locs = np.where((ia_occupied > 0.081) | (ia_occupied < -0.081))[0]
        ws_normal_locs = np.delete(np.arange(0, len(occupied_blocks)), ws_fail_locs)

        points = np.column_stack([lons, lats])
        segments = np.stack([points[:-1], points[1:]], axis=1)

        n_segments = len(segments)
        n_yellow += (len(rs_yellow_locs) + len(ws_yellow_locs))
        n_orange += (len(rs_orange_locs) + len(ws_orange_locs))
        n_red += (len(rs_fail_locs) + len(ws_fail_locs))

        colors = np.full(n_segments, 'lightgrey', dtype=object)
        colors[unoccupied_blocks[rs_yellow_locs]] = 'gold'
        colors[unoccupied_blocks[rs_orange_locs]] = 'darkorange'
        colors[unoccupied_blocks[rs_fail_locs]] = 'red'
        colors[occupied_blocks[ws_yellow_locs]] = 'gold'
        colors[occupied_blocks[ws_orange_locs]] = 'darkorange'
        colors[occupied_blocks[ws_fail_locs]] = 'red'
        lc = LineCollection(segments, colors=colors, linewidths=2)

        ax0.add_collection(lc)
        ax0.autoscale()

        axs[n_line].scatter(occupied_blocks[ws_fail_locs], ia_occupied[ws_fail_locs], marker='>', edgecolors='black', c='red', zorder=10)
        axs[n_line].scatter(occupied_blocks[ws_normal_locs], ia_occupied[ws_normal_locs], marker='>', edgecolors='lime', c='white', zorder=10)
        axs[n_line].scatter(unoccupied_blocks[rs_fail_locs], ia_unoccupied[rs_fail_locs], marker='.', edgecolors='black', c='lime', zorder=10)
        axs[n_line].scatter(unoccupied_blocks[rs_normal_locs], ia_unoccupied[rs_normal_locs], marker='.', edgecolors='tomato', c='white', zorder=10)
        axs[n_line].axhline(0.055, color='tomato', zorder=5)
        axs[n_line].axhline(-0.055, color='tomato', zorder=5)
        axs[n_line].axhline(0.081, color='lime', zorder=5, linestyle='--')
        axs[n_line].axhline(-0.081, color='lime', zorder=5, linestyle='--')

    legend_elements = [
        Line2D([0], [0], color='gold', lw=2, label=f'{n_yellow}'),
        Line2D([0], [0], color='darkorange', lw=2, label=f'{n_orange}'),
        Line2D([0], [0], color='red', lw=2, label=f'{n_red}'),
    ]
    ax0.legend(handles=legend_elements, loc='upper right')

    legend_elements = [Line2D([0], [0], color='white', lw=2, label=f'Glasgow to Edinburgh via Falkirk High, E = {e} V/km')]
    legend1 = ax1.legend(handles=legend_elements, loc='upper center')
    legend_elements = [Line2D([0], [0], color='white', lw=2, label=f'West Coast Main Line, E = {e} V/km')]
    legend2 = ax2.legend(handles=legend_elements, loc='upper center')
    legend_elements = [Line2D([0], [0], color='white', lw=2, label=f'East Coast Main Line, E = {e} V/km')]
    legend3 = ax3.legend(handles=legend_elements, loc='upper center')
    legend1.set_zorder(15)
    legend2.set_zorder(15)
    legend3.set_zorder(15)
    ax2.set_ylabel('Relay Current (A)')
    ax3.set_xlabel('Track Circuit Number')
    ax1.set_ylim(-0.8, 0.8)
    ax2.set_ylim(-0.8, 0.8)
    ax3.set_ylim(-0.8, 0.8)

    plt.savefig(f'frames/alert/alert_pos_{index}.jpg')
    #plt.show()
    plt.close()


def video_saver():
    filenames = glob.glob(f'frames/alert/alert_pos*.jpg')
    print(filenames)

    def frame_number(path):

        match = re.search(r'alert_pos_(\d+)', path)
        return int(match.group(1))

    filenames = sorted(filenames, key=frame_number)

    with imageio.get_writer(f"frames/animation_alert_pos.mp4", fps=10) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)


e_vals = np.arange(0, 10, 0.1)
for i in range(0, len(e_vals)):
    alert_map_geographic(e_vals[i], i)

#video_saver()
