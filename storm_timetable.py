import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from models import model
import imageio.v2 as imageio
import glob


def may2024():
    data = np.load(f'data/axle_positions/timetable/east_coast_main_line_axle_positions_timetable.npz', allow_pickle=True)
    axles_day = data['axle_pos_a_all']
    axles = np.concatenate([axles_day, axles_day])

    data = np.load(f'data/storm_e_fields/May2024/east_coast_main_line_May2024_e_blocks_v2.npz')
    exs = data['ex_blocks']
    eys = data['ey_blocks']

    data = np.load(f'data/rail_data/east_coast_main_line/east_coast_main_line_distances_bearings.npz')
    blocks = data['distances']  # Block lengths (km)
    block_sum = np.cumsum(blocks)

    for i in range(1000, 2000):
        print(i)
        plt.rcParams['font.size'] = '10'
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(1, 1)
        ax0 = fig.add_subplot(gs[0])

        occupied_blocks = []
        unoccupied_blocks = []
        ax = axles[i]
        if len(ax) > 0:
            for ax_ind in ax:
                occupied_blocks.append(np.where(block_sum >= ax_ind)[0][0])
            occupied_blocks = np.unique(occupied_blocks)
        unoccupied_blocks = np.arange(0, len(blocks), 1)
        unoccupied_blocks = np.delete(unoccupied_blocks, occupied_blocks)

        ex = exs[:, i:i + 2] / 1000
        ey = eys[:, i:i + 2] / 1000
        outputs = model('east_coast_main_line', axle_pos_a=ax, ex_blocks=ex, ey_blocks=ey)
        currents = outputs['i_relays_a'][:, 0]

        occupied_currents = currents[occupied_blocks]
        wrong_side_idx = np.where((occupied_currents > 0.081) | (occupied_currents < -0.081))
        correct_occupied_idx = np.arange(0, len(occupied_currents), 1)
        correct_occupied_idx = np.delete(correct_occupied_idx, wrong_side_idx)

        unoccupied_currents = currents[unoccupied_blocks]
        right_side_idx = np.where((unoccupied_currents < 0.055) & (unoccupied_currents > -0.055))
        correct_unoccupied_idx = np.arange(0, len(unoccupied_currents), 1)
        correct_unoccupied_idx = np.delete(correct_unoccupied_idx, right_side_idx)

        ax0.scatter(unoccupied_blocks[correct_unoccupied_idx], currents[unoccupied_blocks][correct_unoccupied_idx], s=1,
                    marker='o', color='lightgray')
        ax0.scatter(unoccupied_blocks[right_side_idx], currents[unoccupied_blocks][right_side_idx], s=1, marker='o',
                    color='limegreen')
        if len(occupied_currents) > 0:
            ax0.scatter(occupied_blocks[correct_occupied_idx], currents[occupied_blocks][correct_occupied_idx], s=1,
                        marker='>', color='lightgray')
            ax0.scatter(occupied_blocks[wrong_side_idx], currents[occupied_blocks][wrong_side_idx], s=1, marker='>',
                        color='red')
        ax0.set_xlabel('Track circuit number')
        ax0.set_ylabel('Relay current (A)')
        plt.savefig(f'frames/may2024/frame{i}')
        plt.close()


def may2024x20():
    data = np.load(f'data/axle_positions/timetable/east_coast_main_line_axle_positions_timetable.npz', allow_pickle=True)
    axles_day = data['axle_pos_a_all']
    axles = np.concatenate([axles_day, axles_day])

    data = np.load(f'data/storm_e_fields/May2024/east_coast_main_line_May2024_e_blocks_v2.npz')
    exs = data['ex_blocks']
    eys = data['ey_blocks']

    ey_tot = np.concatenate(eys)
    print(np.max(ey_tot))

    data = np.load(f'data/rail_data/east_coast_main_line/east_coast_main_line_distances_bearings.npz')
    blocks = data['distances']  # Block lengths (km)
    block_sum = np.cumsum(blocks)

    for i in range(1000, 2000):
        print(i)
        plt.rcParams['font.size'] = '10'
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(1, 1)
        ax0 = fig.add_subplot(gs[0])

        occupied_blocks = []
        ax = axles[i]
        if len(ax) > 0:
            for ax_ind in ax:
                occupied_blocks.append(np.where(block_sum >= ax_ind)[0][0])
            occupied_blocks = np.unique(occupied_blocks)
        unoccupied_blocks = np.arange(0, len(blocks), 1)
        unoccupied_blocks = np.delete(unoccupied_blocks, occupied_blocks)

        ex = exs[:, i:i+2] / 50
        ey = eys[:, i:i+2] / 50
        outputs = model('east_coast_main_line', axle_pos_a=ax, ex_blocks=ex, ey_blocks=ey)
        currents = outputs['i_relays_a'][:, 0]

        occupied_currents = currents[occupied_blocks]
        wrong_side_idx = np.where((occupied_currents > 0.081) | (occupied_currents < -0.081))
        correct_occupied_idx = np.arange(0, len(occupied_currents), 1)
        correct_occupied_idx = np.delete(correct_occupied_idx, wrong_side_idx)

        unoccupied_currents = currents[unoccupied_blocks]
        right_side_idx = np.where((unoccupied_currents < 0.055) & (unoccupied_currents > -0.055))
        correct_unoccupied_idx = np.arange(0, len(unoccupied_currents), 1)
        correct_unoccupied_idx = np.delete(correct_unoccupied_idx, right_side_idx)

        ax0.scatter(unoccupied_blocks[correct_unoccupied_idx], currents[unoccupied_blocks][correct_unoccupied_idx], s=1, marker='o', color='lightgray')
        ax0.scatter(unoccupied_blocks[right_side_idx], currents[unoccupied_blocks][right_side_idx], s=1, marker='o', color='limegreen')
        if len(occupied_currents) > 0:
            ax0.scatter(occupied_blocks[correct_occupied_idx], currents[occupied_blocks][correct_occupied_idx], s=1, marker='>', color='lightgray')
            ax0.scatter(occupied_blocks[wrong_side_idx], currents[occupied_blocks][wrong_side_idx], s=1, marker='>', color='red')
        ax0.set_xlabel('Track circuit number')
        ax0.set_ylabel('Relay current (A)')
        ax0.set_ylim(-0.4, 0.4)
        ax0.axhline(-0.081, color='green', alpha=0.5)
        ax0.axhline(0.081, color='green', alpha=0.5)
        ax0.axhline(0.055, color='red', alpha=0.5)
        ax0.axhline(-0.055, color='red', alpha=0.5)
        plt.savefig(f'frames/may2024x20/frame{i}')
        plt.close()


def video():
    # Get your files in order (adjust pattern/sorting as needed)
    filenames = sorted(glob.glob(f"frames/may2024x20/*.png"))

    with imageio.get_writer("frames/animation_may2024x20.mp4", fps=10) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

may2024x20()
#may2024()
#video()