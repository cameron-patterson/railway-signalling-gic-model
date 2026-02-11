import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from models import model


def run_model(section):
    """
    Runs the model.

    :param section: Section identifier ('glasgow_edinburgh_falkirk', 'west_coast_main_line', 'east_coast_main_line')
    :return: None.
    """

    # Set number of blocks based on section identifier
    blocks_by_section = {
        'glasgow_edinburgh_falkirk': 119,
        'west_coast_main_line': 936,
        'east_coast_main_line': 913,
    }
    try:
        n_blocks = blocks_by_section[section]
    except KeyError:
        print('Section not defined.')

    bearings = np.deg2rad(np.arange(0, 360, 5))
    e_values = np.linspace(0, 20, 2001)
    currents_all_e = np.full((len(bearings), n_blocks, len(e_values)), np.nan)

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(19, 8))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])
    for y_trac in [1.6, None]:
        for i in range(0, len(bearings)):
            ex_uni = e_values * np.cos(bearings[i])
            ey_uni = e_values * np.sin(bearings[i])
            output = model(section_name=section, ex_uni=ex_uni, ey_uni=ey_uni, y_trac=y_trac)
            currents_all_e[i] = np.abs(output['i_relays_a'])
            print(i)

        misop_thresholds_by_bearing = np.full((n_blocks, len(bearings)), np.nan)  # Electric field misoperation threshold for each bearing
        minimum_misop_thresholds = np.full(n_blocks, np.nan)  # Minimum electric field misoperation threshold for each block
        for idx_block in range(0, len(currents_all_e[0, :, 0])):
            for idx_bearing in range(0, len(currents_all_e[:, 0, 0])):
                misop_locs = np.where(currents_all_e[idx_bearing, idx_block, :] < 0.055)[0]
                if len(misop_locs) > 0:
                    misop_thresholds_by_bearing[idx_block, idx_bearing] = np.nanmin(e_values[misop_locs])
                    minimum_misop_thresholds[idx_block] = np.nanmin(misop_thresholds_by_bearing[idx_block, :])

        ax0.plot(minimum_misop_thresholds, '.', label=f'y_trac = {y_trac}')
        ax0.set_xlabel('Block Number')
        ax0.set_ylabel('Lowest Misoperation E')
    ax0.legend()
    plt.show()
    plt.close()


run_model('west_coast_main_line')
