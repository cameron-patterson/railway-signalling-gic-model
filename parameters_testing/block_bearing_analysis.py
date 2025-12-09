from models import model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def save_currents_all_e(section):
    data = np.load(f'../data/rail_data/{section}/{section}_distances_bearings.npz')
    blocks_bearings = np.rad2deg(data['bearings'])  # Block bearings (radians); Note: zero is directly northwards, with positive values increasing clockwise
    bearings = np.deg2rad(np.arange(0, 360, 5))
    e_values = np.linspace(0, 20, 2001)
    currents_all_e = np.full((len(bearings), len(blocks_bearings), len(e_values)), np.nan)
    for i in range(0, len(bearings)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])
        output = model(section_name=section, ex_uni=ex_uni, ey_uni=ey_uni, y_trac=1.6)
        currents_all_e[i] = np.abs(output['i_relays_a'])
        print(i)
    np.save(f'currents_all_e_{section}', currents_all_e)


def calculate_thresholds(section):
    data = np.load(f'../data/rail_data/{section}/{section}_distances_bearings.npz')
    blocks_bearings = np.rad2deg(data['bearings'])  # Block bearings (radians); Note: zero is directly northwards, with positive values increasing clockwise
    e_values = np.linspace(0, 20, 2001)
    currents_all_e = np.load(f'currents_all_e_{section}.npy')
    threshold_misoperation_e_fields = np.full(len(blocks_bearings), np.nan)
    for j in range(0, len(blocks_bearings)):
        currents_block = currents_all_e[:, j, :]
        if len(np.where(currents_block < 0.055)[1]) > 0:
            threshold_misoperation_e_fields[j] = np.min(e_values[np.where(currents_block < 0.055)[1]])
        else:
            pass

    plt.plot(threshold_misoperation_e_fields, '.')
    plt.show()


def potential_curve(section):
    data = np.load(f'../data/rail_data/{section}/{section}_distances_bearings.npz')
    blocks_bearings = np.rad2deg(data['bearings'])  # Block bearings (radians); Note: zero is directly northwards, with positive values increasing clockwise
    bearings = np.deg2rad(np.arange(0, 360, 60))
    e_values = np.array([0, 5])
    plt.rcParams['font.size'] = 15
    fig, axs = plt.subplots(2, 3, figsize=(19, 8))
    axs = axs.ravel()
    for i in range(0, int(len(bearings)/2)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])
        output = model(section_name=section, ex_uni=ex_uni, ey_uni=ey_uni, y_trac=1.6)
        v_matrix = output['v_matrix']
        node_relay_trac = output['node_locs_relay_trac_a']
        node_relay_sig = output['node_locs_relay_sig_a']
        axs[i].plot(v_matrix[:308, 1], color='red')
        axs[i].plot(node_relay_trac, v_matrix[node_relay_sig, 1], '.', color='red')
        # Create inset axis
        axs_inset = inset_axes(axs[i], width="15%", height="15%", loc="upper left")
        # Draw circle
        theta = np.linspace(0, 2 * np.pi, 300)
        x = np.cos(theta)
        y = np.sin(theta)
        axs_inset.plot(x, y, color='black', alpha=0.25)
        # Draw radial line
        x_line = [0, np.sin(bearings[i])]
        y_line = [0, np.cos(bearings[i])]
        axs_inset.plot(x_line, y_line, color='red')
        # Keep equal aspect
        axs_inset.set_aspect("equal")
        axs_inset.set_xticks([])
        axs_inset.set_yticks([])
        axs_inset.set_xlabel(f"{round(np.rad2deg(bearings[i]))}°", fontsize=8, labelpad=2)

    for i in range(int(len(bearings)/2), len(bearings)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])
        output = model(section_name=section, ex_uni=ex_uni, ey_uni=ey_uni, y_trac=1.6)
        v_matrix = output['v_matrix']
        node_relay_trac = output['node_locs_relay_trac_a']
        node_relay_sig = output['node_locs_relay_sig_a']
        axs[i].plot(v_matrix[:308, 1], color='blue')
        axs[i].plot(node_relay_trac, v_matrix[node_relay_sig, 1], '.', color='blue')
        # Create inset axis
        axs_inset = inset_axes(axs[i], width="15%", height="15%", loc="upper right")
        # Draw circle
        theta = np.linspace(0, 2 * np.pi, 300)
        x = np.cos(theta)
        y = np.sin(theta)
        axs_inset.plot(x, y, color='black', alpha=0.25)
        # Draw radial line
        x_line = [0, np.sin(bearings[i])]
        y_line = [0, np.cos(bearings[i])]
        axs_inset.plot(x_line, y_line, color='blue')
        # Keep equal aspect
        axs_inset.set_aspect("equal")
        axs_inset.set_xticks([])
        axs_inset.set_yticks([])
        axs_inset.set_xlabel(f"{round(np.rad2deg(bearings[i]))}°", fontsize=8, labelpad=2)

    plt.show()


def potential_difs(section):
    data = np.load(f'../data/rail_data/{section}/{section}_distances_bearings.npz')
    blocks_bearings = np.rad2deg(data['bearings'])  # Block bearings (radians); Note: zero is directly northwards, with positive values increasing clockwise
    bearings = np.deg2rad(np.arange(0, 360, 60))
    e_values = np.array([0, 5])
    plt.rcParams['font.size'] = 15
    fig, axs = plt.subplots(2, 3, figsize=(19, 8))
    axs = axs.ravel()
    for i in range(0, int(len(bearings)/2)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])
        output = model(section_name=section, ex_uni=ex_uni, ey_uni=ey_uni, y_trac=1.6)
        v_matrix = output['v_matrix']
        node_relay_trac = output['node_locs_relay_trac_a']
        node_relay_sig = output['node_locs_relay_sig_a']
        axs[i].plot(v_matrix[node_relay_sig, 1] - v_matrix[node_relay_trac, 1], '.', color='red')
        axs[i].set_ylim(np.min(v_matrix[node_relay_sig, 1] - v_matrix[node_relay_trac, 1]) - 1, np.max(v_matrix[node_relay_sig, 1] - v_matrix[node_relay_trac, 1]) + 5)
        # Create inset axis
        axs_inset = inset_axes(axs[i], width="15%", height="15%", loc="upper right")
        # Draw circle
        theta = np.linspace(0, 2 * np.pi, 300)
        x = np.cos(theta)
        y = np.sin(theta)
        axs_inset.plot(x, y, color='black', alpha=0.25)
        # Draw radial line
        x_line = [0, np.sin(bearings[i])]
        y_line = [0, np.cos(bearings[i])]
        axs_inset.plot(x_line, y_line, color='red')
        # Keep equal aspect
        axs_inset.set_aspect("equal")
        axs_inset.set_xticks([])
        axs_inset.set_yticks([])
        axs_inset.set_xlabel(f"{round(np.rad2deg(bearings[i]))}°", fontsize=8, labelpad=2)

    for i in range(int(len(bearings)/2), len(bearings)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])
        output = model(section_name=section, ex_uni=ex_uni, ey_uni=ey_uni, y_trac=1.6)
        v_matrix = output['v_matrix']
        node_relay_trac = output['node_locs_relay_trac_a']
        node_relay_sig = output['node_locs_relay_sig_a']
        axs[i].plot(v_matrix[node_relay_sig, 1] - v_matrix[node_relay_trac, 1], '.', color='blue')
        axs[i].set_ylim(np.min(v_matrix[node_relay_sig, 1] - v_matrix[node_relay_trac, 1]) - 1, np.max(v_matrix[node_relay_sig, 1] - v_matrix[node_relay_trac, 1]) + 5)
        # Create inset axis
        axs_inset = inset_axes(axs[i], width="15%", height="15%", loc="upper left")
        # Draw circle
        theta = np.linspace(0, 2 * np.pi, 300)
        x = np.cos(theta)
        y = np.sin(theta)
        axs_inset.plot(x, y, color='black', alpha=0.25)
        # Draw radial line
        x_line = [0, np.sin(bearings[i])]
        y_line = [0, np.cos(bearings[i])]
        axs_inset.plot(x_line, y_line, color='blue')
        # Keep equal aspect
        axs_inset.set_aspect("equal")
        axs_inset.set_xticks([])
        axs_inset.set_yticks([])
        axs_inset.set_xlabel(f"{round(np.rad2deg(bearings[i]))}°", fontsize=8, labelpad=2)

    plt.show()


def currents(section):
    data = np.load(f'../data/rail_data/{section}/{section}_distances_bearings.npz')
    blocks_bearings = np.rad2deg(data['bearings'])  # Block bearings (radians); Note: zero is directly northwards, with positive values increasing clockwise
    bearings = np.deg2rad(np.arange(0, 360, 60))
    e_values = np.array([0, 5])
    plt.rcParams['font.size'] = 15
    fig, axs = plt.subplots(2, 3, figsize=(19, 8))
    axs = axs.ravel()
    for i in range(0, int(len(bearings)/2)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])
        output = model(section_name=section, ex_uni=ex_uni, ey_uni=ey_uni, y_trac=1.6)
        ia = output['i_relays_a']
        axs[i].plot(np.abs(ia[:, 1]), '.', color='red')
        # Create inset axis
        axs_inset = inset_axes(axs[i], width="15%", height="15%", loc="upper center")
        # Draw circle
        theta = np.linspace(0, 2 * np.pi, 300)
        x = np.cos(theta)
        y = np.sin(theta)
        axs_inset.plot(x, y, color='black', alpha=0.25)
        # Draw radial line
        x_line = [0, np.sin(bearings[i])]
        y_line = [0, np.cos(bearings[i])]
        axs_inset.plot(x_line, y_line, color='red')
        # Keep equal aspect
        axs_inset.set_aspect("equal")
        axs_inset.set_xticks([])
        axs_inset.set_yticks([])
        axs_inset.set_xlabel(f"{round(np.rad2deg(bearings[i]))}°", fontsize=8, labelpad=2)

    for i in range(int(len(bearings)/2), len(bearings)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])
        output = model(section_name=section, ex_uni=ex_uni, ey_uni=ey_uni, y_trac=1.6)
        ia = output['i_relays_a']
        axs[i].plot(np.abs(ia[:, 1]), '.', color='blue')
        # Create inset axis
        axs_inset = inset_axes(axs[i], width="15%", height="15%", loc="upper center")
        # Draw circle
        theta = np.linspace(0, 2 * np.pi, 300)
        x = np.cos(theta)
        y = np.sin(theta)
        axs_inset.plot(x, y, color='black', alpha=0.25)
        # Draw radial line
        x_line = [0, np.sin(bearings[i])]
        y_line = [0, np.cos(bearings[i])]
        axs_inset.plot(x_line, y_line, color='blue')
        # Keep equal aspect
        axs_inset.set_aspect("equal")
        axs_inset.set_xticks([])
        axs_inset.set_yticks([])
        axs_inset.set_xlabel(f"{round(np.rad2deg(bearings[i]))}°", fontsize=8, labelpad=2)

    plt.show()


def misoperation_fields_bearings(section):
    e_values = np.linspace(0, 20, 2001)
    bearings = np.deg2rad(np.arange(0, 360, 5))
    currents_all_e = np.load(f'currents_all_e_{section}.npy')
    data = np.load(f'../data/rail_data/{section}/{section}_distances_bearings.npz')
    blocks_bearings = np.rad2deg(data['bearings'])  # Block bearings (radians); Note: zero is directly northwards, with positive values increasing clockwise
    blocks_bearings[np.where(blocks_bearings < 0)] += 360
    for i in range(0, len(currents_all_e[0, :, 0])):
        bearing_thresh_e = np.full(len(bearings), np.nan)
        for b in range(0, len(currents_all_e[:, 0, 0])):
            e_thresh_locs = np.where(currents_all_e[b, i, :] < 0.055)[0]
            if len(e_thresh_locs) > 0:
                bearing_thresh_e[b] = np.min(e_values[e_thresh_locs])
            else:
                pass
        plt.plot(np.rad2deg(bearings), bearing_thresh_e, 'x')
        if blocks_bearings[i] < 180:
            if i % 2 == 0:
                plt.axvline(blocks_bearings[i], color='red')
            else:
                plt.axvline(blocks_bearings[i] + 180, color='blue')
        if blocks_bearings[i] > 180:
            if i % 2 != 0:
                plt.axvline(blocks_bearings[i], color='blue')
            else:
                plt.axvline(blocks_bearings[i] - 180, color='red')
        plt.show()


def misoperation_threshold_bearings(section):
    e_values = np.linspace(0, 20, 2001)
    bearings = np.deg2rad(np.arange(0, 360, 5))
    currents_all_e = np.load(f'currents_all_e_{section}.npy')
    data = np.load(f'../data/rail_data/{section}/{section}_distances_bearings.npz')
    blocks_bearings = np.rad2deg(data['bearings'])  # Block bearings (radians); Note: zero is directly northwards, with positive values increasing clockwise
    for i in range(0, len(currents_all_e[0, :, 0])):
        misop_locs = np.where(currents_all_e[:, i, :] < 0.055)[1]
        pass


#for sec in ['glasgow_edinburgh_falkirk', 'west_coast_main_line', 'east_coast_main_line']:
#    save_currents_all_e(sec)
#calculate_thresholds('west_coast_main_line')
#potential_curve('glasgow_edinburgh_falkirk')
#potential_difs('glasgow_edinburgh_falkirk')
#currents('glasgow_edinburgh_falkirk')
misoperation_fields_bearings('west_coast_main_line')
#misoperation_threshold_bearings('glasgow_edinburgh_falkirk')
