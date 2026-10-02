import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from models import model
import imageio.v2 as imageio
import glob
import datetime
import re


def storm(sec, storm, scale):
    data = np.genfromtxt(
        f'data/storm_indices/{storm}_indices_IAGA.txt',
        skip_header=1,
        dtype=str
    )

    date_strs = data[:, 0]
    time_strs = data[:, 1]

    timestamps = np.array([
        datetime.datetime.strptime(d + ' ' + t, '%Y-%m-%d %H:%M:%S.%f')
        for d, t in zip(date_strs, time_strs)
    ])

    sym_h = data[:, 6].astype(float)

    data = np.load(f'data/axle_positions/timetable/{sec}_axle_positions_timetable.npz', allow_pickle=True)
    axles_day = data['axle_pos_a_all']
    axles = np.concatenate([axles_day, axles_day])

    data = np.load(f'data/storm_e_fields/{storm}/{sec}_{storm}_e_blocks_v2.npz')
    exs = data['ex_blocks']
    eys = data['ey_blocks']

    data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
    blocks = data['distances']  # Block lengths (km)
    block_sum = np.cumsum(blocks)

    outputs = model(section_name=sec, ex_uniform=np.array([0]), ey_uniform=np.array([0]))
    currents0 = outputs['i_relays_a'][:, 0]

    for i in range(0, len(exs[0, :])):
        print(i)
        plt.rcParams['font.size'] = '10'
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(3, 1, hspace=0.3)
        ax0 = fig.add_subplot(gs[1:])
        ax1 = fig.add_subplot(gs[0])

        occupied_blocks = []
        ax = axles[i]
        if len(ax) > 0:
            for ax_ind in ax:
                occupied_blocks.append(np.where(block_sum >= ax_ind)[0][0])
            occupied_blocks = np.unique(occupied_blocks)
        unoccupied_blocks = np.arange(0, len(blocks), 1)
        unoccupied_blocks = np.delete(unoccupied_blocks, occupied_blocks)

        ex = exs[:, i:i + 2] / 1000 * scale
        ey = eys[:, i:i + 2] / 1000 * scale

        outputs = model(section_name=sec, axle_pos_a=ax, ex_blocks=ex, ey_blocks=ey)
        currents = outputs['i_relays_a'][:, 0]

        occupied_currents = currents[occupied_blocks]
        wrong_side_idx = np.where((occupied_currents > 0.081) | (occupied_currents < -0.081))
        correct_occupied_idx = np.arange(0, len(occupied_currents), 1)
        correct_occupied_idx = np.delete(correct_occupied_idx, wrong_side_idx)

        unoccupied_currents = currents[unoccupied_blocks]
        right_side_idx = np.where((unoccupied_currents < 0.055) & (unoccupied_currents > -0.055))
        correct_unoccupied_idx = np.arange(0, len(unoccupied_currents), 1)
        correct_unoccupied_idx = np.delete(correct_unoccupied_idx, right_side_idx)

        linewidth = 0.5
        markersize = 20
        ax0.scatter(unoccupied_blocks[correct_unoccupied_idx], currents[unoccupied_blocks][correct_unoccupied_idx], s=markersize, marker='o', linewidth=linewidth, edgecolor='red', facecolor='white', zorder=5)
        ax0.scatter(unoccupied_blocks[right_side_idx], currents[unoccupied_blocks][right_side_idx], s=markersize, marker='o', linewidth=linewidth, edgecolor='red', facecolor='green', zorder=5)
        if len(occupied_currents) > 0:
            ax0.scatter(occupied_blocks[correct_occupied_idx], currents[occupied_blocks][correct_occupied_idx], s=markersize, marker='>', linewidth=linewidth, edgecolor='green', facecolor='white', zorder=5)
            ax0.scatter(occupied_blocks[wrong_side_idx], currents[occupied_blocks][wrong_side_idx], s=markersize, marker='>', linewidth=linewidth, edgecolor='green', facecolor='red', zorder=5)

        ax0.set_xlabel('Track circuit number')
        ax0.set_ylabel('Relay current (A)')
        ax0.set_ylim(-0.4, 0.4)
        ax0.axhline(-0.055, color='tomato', linewidth=1, linestyle='--')
        ax0.axhline(0.055, color='tomato', linewidth=1, linestyle='--')
        ax0.axhline(-0.081, color='limegreen', linewidth=1, linestyle='-')
        ax0.axhline(0.081, color='limegreen', linewidth=1, linestyle='-')

        ax0.scatter(unoccupied_blocks, currents0[unoccupied_blocks], edgecolors='white', facecolor='lightgray', s=20, marker='o', zorder=1)
        ax0.scatter(occupied_blocks, np.zeros(len(occupied_blocks)), edgecolors='white', facecolor='lightgray', s=20, marker='>', zorder=1)

        ax1.plot(timestamps, sym_h, linewidth=1, color='cornflowerblue')
        ax1.set_xlim(timestamps[0], timestamps[-1])
        ax1.set_ylim(-800, 100)
        ax1.set_ylabel('SYM-H (nT)')
        ax1.set_xlabel('Universal Time')
        ax1.axvline(timestamps[i], color='black', alpha=0.5, linewidth=1)

        #plt.show()
        plt.savefig(f'frames/{sec}/{storm}x{scale}/frame{i}')
        plt.close()


def video(sec, storm, scale):
    filenames = glob.glob(f'frames/{sec}/{storm}x{scale}/frame*.png')

    def frame_number(path):
        match = re.search(r'frame(\d+)', path)
        return int(match.group(1))

    filenames = sorted(filenames, key=frame_number)

    with imageio.get_writer(f"frames/animation_{sec}_{storm}x{scale}.mp4", fps=30) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)


def test_axles(sec):
    data = np.load(f'data/axle_positions/timetable/{sec}_axle_positions_timetable.npz', allow_pickle=True)
    axles_day = data['axle_pos_a_all']
    for i in range(0, len(axles_day)):
        plt.plot(axles_day[i], np.full(len(axles_day[i]), i), '.')
    plt.xlim(0, 630)
    plt.show()
    plt.close()


# storm('glasgow_edinburgh_falkirk', 'May2024', 1)
# storm('east_coast_main_line', 'May2024', 1)
# storm('glasgow_edinburgh_falkirk', 'May2024', 10)
# storm('east_coast_main_line', 'May2024', 10)
# storm('glasgow_edinburgh_falkirk', 'Mar1989', 1)
# storm('east_coast_main_line', 'Mar1989', 1)
# storm('glasgow_edinburgh_falkirk', 'Mar1989', 5)
# storm('east_coast_main_line', 'Mar1989', 5)

# video('glasgow_edinburgh_falkirk', 'May2024', 1)
# video('east_coast_main_line', 'May2024', 1)
# video('glasgow_edinburgh_falkirk', 'May2024', 10)
# video('east_coast_main_line', 'May2024', 10)
# video('glasgow_edinburgh_falkirk', 'Mar1989', 1)
# video('east_coast_main_line', 'Mar1989', 1)
# video('glasgow_edinburgh_falkirk', 'Mar1989', 5)
# video('east_coast_main_line', 'Mar1989', 5)
