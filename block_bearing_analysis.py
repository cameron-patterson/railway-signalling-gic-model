from models import model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Right Side Failure Analysis
def save_currents_all_e_rs(section):
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    blocks_bearings = np.rad2deg(data['bearings'])
    bearings = np.deg2rad(np.arange(0, 360, 5))
    e_values = np.linspace(0, 20, 2001)
    currents_all_e = np.full((len(bearings), len(blocks_bearings), len(e_values)), np.nan)
    for i in range(0, len(bearings)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])
        output = model(section_name=section, ex_uni=ex_uni, ey_uni=ey_uni, y_trac=1.6)
        currents_all_e[i] = output['i_relays_a']
        print(i)
    np.save(f'currents_all_e_{section}_rs', currents_all_e)


def calculate_thresholds_rs(section):
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    blocks_bearings = np.rad2deg(data['bearings'])  # Block bearings (radians); Note: zero is directly northwards, with positive values increasing clockwise
    e_values = np.linspace(0, 20, 2001)
    currents_all_e = np.abs(np.load(f'currents_all_e_{section}_rs.npy'))
    threshold_misoperation_e_fields = np.full(len(blocks_bearings), np.nan)
    for j in range(0, len(blocks_bearings)):
        currents_block = currents_all_e[:, j, :]
        if len(np.where(currents_block < 0.055)[1]) > 0:
            threshold_misoperation_e_fields[j] = np.min(e_values[np.where(currents_block < 0.055)[1]])
        else:
            pass

    plt.rcParams['font.size'] = 12
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    axs.plot(threshold_misoperation_e_fields, '.')
    axs.set_xlabel('Block Number')
    axs.set_ylabel('Misoperation Threshold Electric Field (V/km)')
    plt.show()


def potential_curve_rs(section):
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    blocks_bearings = np.rad2deg(data['bearings'])  # Block bearings (radians); Note: zero is directly northwards, with positive values increasing clockwise
    bearings = np.deg2rad(np.arange(0, 360, 60))
    e_values = np.array([0, 5])
    plt.rcParams['font.size'] = 12
    fig, axs = plt.subplots(2, 3, figsize=(19, 8))
    axs = axs.ravel()
    for i in range(0, int(len(bearings)/2)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])
        output = model(section_name=section, ex_uni=ex_uni, ey_uni=ey_uni, y_trac=1.6)
        v_matrix = output['v_matrix']
        node_relay_trac = output['node_locs_relay_trac_a']
        node_relay_sig = output['node_locs_relay_sig_a']
        axs[i].plot(v_matrix[:308, 1], color='red', label='Traction Rail')
        axs[i].plot(node_relay_trac, v_matrix[node_relay_sig, 1], '.', color='red', label='Signal Rail Relay Node')
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
        axs[i].plot(v_matrix[:308, 1], color='blue', label='Traction Rail')
        axs[i].plot(node_relay_trac, v_matrix[node_relay_sig, 1], '.', color='blue', label='Signal Rail Relay Node')
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
    axs[1].legend(loc='upper center')
    axs[4].legend(loc='upper center')
    for a in [3, 4, 5]:
        axs[a].set_xlabel('Traction Rail Node')
    for a in [0, 3]:
        axs[a].set_ylabel('Potential (V)')
    plt.show()


def potential_difs_rs(section):
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
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
    for a in [3, 4, 5]:
        axs[a].set_xlabel('Traction Rail Node')
    for a in [0, 3]:
        axs[a].set_ylabel('Potential Difference (V)')
    plt.show()


def misoperation_fields_bearings_rs(section):
    e_values = np.linspace(0, 20, 2001)
    bearings = np.deg2rad(np.arange(0, 360, 5))
    currents_all_e = np.abs(np.load(f'currents_all_e_{section}_rs.npy'))
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    blocks_bearings = np.rad2deg(data['bearings'])
    blocks_bearings[np.where(blocks_bearings < 0)] += 360
    for i in range(0, len(currents_all_e[0, :, 0])):
        bearing_thresh_e = np.full(len(bearings), np.nan)
        for b in range(0, len(currents_all_e[:, 0, 0])):
            e_thresh_locs = np.where(currents_all_e[b, i, :] < 0.055)[0]
            if len(e_thresh_locs) > 0:
                bearing_thresh_e[b] = np.min(e_values[e_thresh_locs])
            else:
                pass

        plt.rcParams['font.size'] = 15
        fig, axs = plt.subplots(1, 1, figsize=(10, 8))
        axs.plot(np.rad2deg(bearings), bearing_thresh_e, 'x')
        if blocks_bearings[i] < 180:
            if i % 2 == 0:
                axs.axvline(blocks_bearings[i], color='red')
            else:
                axs.axvline(blocks_bearings[i] + 180, color='blue')
        if blocks_bearings[i] > 180:
            if i % 2 != 0:
                axs.axvline(blocks_bearings[i], color='blue')
            else:
                axs.axvline(blocks_bearings[i] - 180, color='red')
        axs.set_xlabel('Bearing')
        axs.set_ylabel('Misoperation Electric Field Threshold (V/km)')
        axs.set_title(f'Block {i}')
        plt.show()


def misoperation_threshold_bearings_rs(section):
    e_values = np.linspace(0, 20, 2001)
    bearings = np.deg2rad(np.arange(0, 360, 5))
    currents_all_e = np.abs(np.load(f'currents_all_e_{section}_rs.npy'))
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    blocks_bearings = np.rad2deg(data['bearings'])
    #blocks_bearings[1::2] += 180
    misoperation_bearings = np.full(len(currents_all_e[0, :, 0]), np.nan)
    for i in range(0, len(currents_all_e[0, :, 0])):
        misop_locs = np.where(currents_all_e[:, i, :] < 0.055)[1]
        if len(misop_locs > 0):
            misoperation_bearings[i] = round(np.rad2deg(bearings[np.where(currents_all_e[:, i, :] < 0.055)[0][np.where(misop_locs == np.min(misop_locs))][0]]))
        else:
            pass
    plt.rcParams['font.size'] = '15'
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(np.arange(0, len(blocks_bearings)), misoperation_bearings, 'x')
    axs[0].plot(np.arange(0, len(blocks_bearings)), blocks_bearings, '.')
    axs[0].plot(np.arange(0, len(blocks_bearings))[np.where(blocks_bearings < 180)], blocks_bearings[np.where(blocks_bearings < 180)] + 180, '.')
    axs[0].plot(np.arange(0, len(blocks_bearings))[np.where(blocks_bearings >= 180)], blocks_bearings[np.where(blocks_bearings > 180)] - 180, '.')
    axs[1].plot(abs(misoperation_bearings - blocks_bearings), '.')
    axs[0].set_xlabel('Block Number')
    axs[0].set_ylabel('Bearing (°)')
    axs[1].set_xlabel('Block Number')
    axs[1].set_ylabel('Absolute Bearing Difference (°)')
    plt.show()


# Wrong Side Failure Analysis
def save_currents_all_e_ws(section):
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    blocks_bearings = np.rad2deg(data['bearings'])
    axle_data = np.load(f'data/axle_positions/block_centre/axle_positions_two_track_back_axle_at_centre_{section}.npz', allow_pickle=True)
    axles = axle_data['axle_pos_a_all']
    bearings = np.deg2rad(np.arange(0, 360, 5))
    e_values = np.linspace(0, 20, 2001)
    currents_all_e = np.full((len(bearings), len(blocks_bearings), len(e_values)), np.nan)
    for i in range(0, len(bearings)):
        print(i)
        for b in range(0, len(blocks_bearings)):
            ex_uni = e_values * np.cos(bearings[i])
            ey_uni = e_values * np.sin(bearings[i])
            output = model(section_name=section, ex_uni=ex_uni, ey_uni=ey_uni, y_trac=1.6, axle_pos_a=axles[b])
            currents_all_e[i, b, :] = output['i_relays_a'][b, :]
    np.save(f'currents_all_e_{section}_ws', currents_all_e)


def calculate_thresholds_ws(section):
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    blocks_bearings = np.rad2deg(data['bearings'])  # Block bearings (radians); Note: zero is directly northwards, with positive values increasing clockwise
    e_values = np.linspace(0, 20, 2001)
    currents_all_e = np.load(f'currents_all_e_{section}_ws.npy')
    threshold_misoperation_e_fields1 = np.full(len(blocks_bearings), np.nan)
    threshold_misoperation_e_fields2 = np.full(len(blocks_bearings), np.nan)
    for j in range(0, len(blocks_bearings)):
        currents_block = currents_all_e[:, j, :]
        if len(np.where(currents_block > 0.081)[1]) > 0:
            threshold_misoperation_e_fields1[j] = np.min(e_values[np.where(currents_block > 0.081)[1]])
        else:
            pass
        if len(np.where(currents_block < -0.081)[1]) > 0:
            threshold_misoperation_e_fields2[j] = np.min(e_values[np.where(currents_block < -0.081)[1]])
        else:
            pass

    plt.plot(threshold_misoperation_e_fields1, '.')
    plt.plot(threshold_misoperation_e_fields2, '.')
    plt.show()


def misoperation_fields_bearings_ws(section):
    e_values = np.linspace(0, 20, 2001)
    bearings = np.deg2rad(np.arange(0, 360, 5))
    currents_all_e = np.load(f'currents_all_e_{section}_ws.npy')
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    blocks_bearings = np.rad2deg(data['bearings'])  # Block bearings (radians); Note: zero is directly northwards, with positive values increasing clockwise
    blocks_bearings[np.where(blocks_bearings < 0)] += 360
    for i in range(0, len(currents_all_e[0, :, 0])):
        bearing_thresh_e = np.full(len(bearings), np.nan)
        for b in range(0, len(currents_all_e[:, 0, 0])):
            e_thresh_locs = np.where(currents_all_e[b, i, :] > 0.081)[0]
            if len(e_thresh_locs) > 0:
                bearing_thresh_e[b] = np.min(e_values[e_thresh_locs])
            else:
                pass

        plt.rcParams['font.size'] = 15
        fig, axs = plt.subplots(1, 1, figsize=(10, 8))
        axs.plot(np.rad2deg(bearings), bearing_thresh_e, 'x')
        if blocks_bearings[i] < 180:
            if i % 2 == 0:
                axs.axvline(blocks_bearings[i], color='red')
            else:
                axs.axvline(blocks_bearings[i] + 180, color='blue')
        if blocks_bearings[i] > 180:
            if i % 2 != 0:
                axs.axvline(blocks_bearings[i], color='blue')
            else:
                axs.axvline(blocks_bearings[i] - 180, color='red')
        axs.set_xlabel('Traction Rail Node')
        axs.set_ylabel('Relay Current (A)')
        plt.show()


def misoperation_threshold_bearings_ws(section):
    e_values = np.linspace(0, 20, 2001)
    bearings = np.deg2rad(np.arange(0, 360, 5))
    currents_all_e = np.load(f'currents_all_e_{section}_ws.npy')
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    blocks_bearings = np.rad2deg(data['bearings'])
    misoperation_bearings = np.full(len(currents_all_e[0, :, 0]), np.nan)
    for i in range(0, len(currents_all_e[0, :, 0])):
        misop_locs = np.where(currents_all_e[:, i, :] > 0.081)[1]
        if len(misop_locs > 0):
            misoperation_bearings[i] = round(np.rad2deg(bearings[np.where(currents_all_e[:, i, :] > 0.081)[0][np.where(misop_locs == np.min(misop_locs))][0]]))
        else:
            pass
    plt.rcParams['font.size'] = '15'
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(misoperation_bearings, 'x', zorder=2)
    axs[0].plot(blocks_bearings, '.', zorder=1, color='red')
    axs[0].plot(blocks_bearings + 180, '.', zorder=1, color='blue')
    axs[1].plot(abs(misoperation_bearings - blocks_bearings), '.')
    axs[1].plot(abs(misoperation_bearings - blocks_bearings), '.')
    axs[0].set_xlabel('Block Number')
    axs[0].set_ylabel('Bearing (°)')
    axs[1].set_xlabel('Block Number')
    axs[1].set_ylabel('Absolute Bearing Difference (°)')
    plt.show()


#for sec in ['glasgow_edinburgh_falkirk', 'west_coast_main_line', 'east_coast_main_line']:
#    save_currents_all_e_rs(sec)
#calculate_thresholds_rs('glasgow_edinburgh_falkirk')
#potential_curve_rs('glasgow_edinburgh_falkirk')
#potential_difs_rs('glasgow_edinburgh_falkirk')
#misoperation_fields_bearings_rs('glasgow_edinburgh_falkirk')
#misoperation_threshold_bearings_rs('glasgow_edinburgh_falkirk')


#for sec in ['glasgow_edinburgh_falkirk', 'west_coast_main_line', 'east_coast_main_line']:
#    save_currents_all_e_ws(sec)
#calculate_thresholds_ws('glasgow_edinburgh_falkirk')
#misoperation_fields_bearings_ws('glasgow_edinburgh_falkirk')
#misoperation_threshold_bearings_ws('glasgow_edinburgh_falkirk')
