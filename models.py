import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from sksparse.cholmod import cholesky
from datetime import datetime


# kwargs: relay_type, z_sig, z_trac, y_sig, y_trac, i_power, leakage_profile
def model(section_name, axle_pos_a=None, axle_pos_b=None, ex_blocks=None, ey_blocks=None, ex_uniform=None, ey_uniform=None, electrical_staggering=True, **kwargs):
    # Define network parameters
    if axle_pos_a is None:  # Check if 'a' direction axles are specified, if not, default to empty list
        axle_pos_a = []
    if axle_pos_b is None:  # Check if 'b' direction axles are specified, if not, default to empty list
        axle_pos_b = []
    z_sig = kwargs.get('z_sig', 0.0289)  # Signal rail series impedance (ohms/km); optional kwarg
    z_trac = kwargs.get('z_trac', 0.0289)  # Traction rail series impedance (ohms/km); optional kwarg
    i_power = kwargs.get('i_power', 10/7.2)  # Track circuit power supply equivalent current source (amps); optional kwarg
    y_power = 1/7.2  # Track circuit power supply admittance (siemens)
    y_cb = 1/1e-3  # Cross bond admittance (siemens)
    y_axle = 1/251e-4  # Axle admittance (siemens)
    relay_type = kwargs.get('relay_type', 'BR939A')  # Check if relay type is specified, if not, default to BR939A; optional kwarg
    y_relays = {  # Track circuit relay admittances (siemens) depending on relay type
        'BR939A': 1 / 20,
        'BR966F2': 1 / 9,
        'BR966F9': 1 / 60
    }
    try:
        y_relay = y_relays[relay_type]
    except KeyError:
        raise ValueError(f'Relay type {relay_type} not recognised')

    # Load in the lengths and bearings of the track circuit blocks
    data = np.load(f'data/rail_data/{section_name}/{section_name}_distances_bearings.npz')
    blocks = data['distances']  # Block lengths (km)
    print(f'Number of blocks: {len(blocks)}')
    bearings = data['bearings']  # Block bearings (radians); Note: zero is directly northwards, with positive values increasing clockwise
    blocks_sum = np.cumsum(blocks)  # Cumulative sum of block lengths (km)

    if ex_blocks is not None and ey_blocks is not None:
        pass
    elif ex_uniform is not None and ey_uniform is not None:
        # Setup uniform electric field
        ex_blocks = np.full((len(blocks), len(ex_uniform)), ex_uniform, dtype=float)
        ey_blocks = np.full((len(blocks), len(ey_uniform)), ey_uniform, dtype=float)
    else:
        raise ValueError('Check electric field inputs.')

    # Set traction rail block leakages
    y_trac = kwargs.get('y_trac', None)  # Check if traction rail leakage is specified, if not, default to None; optional kwarg
    if y_trac is None:  # Default traction rail leakage setup
        block_leakages = np.load(f'data/rail_data/{section_name}/{section_name}_block_leakage.npz')
        leakage_profile = kwargs.get('leakage_profile', 'a50')
        y_trac_block = block_leakages[leakage_profile]  # Use median leakage values
    elif isinstance(y_trac, (int, float)):
        y_trac_block = np.full(len(blocks), y_trac, dtype=float)
    elif isinstance(y_trac, (list, tuple, np.ndarray)) and len(y_trac) == 1:
        y_trac_block = np.full(len(blocks), y_trac[0], dtype=float)
    else:
        y_trac_block = y_trac

    # Set Signal rail parallel admittance for moderate conditions (siemens/km)
    y_sig = kwargs.get('y_sig', None)
    if y_sig is None:  # Default signal rail leakage setup
        y_sig_block = np.full(len(blocks), 0.1, dtype=float)
    elif isinstance(y_sig, (int, float)):
        y_sig_block = np.full(len(blocks), y_sig, dtype=float)
    elif isinstance(y_sig, (list, tuple, np.ndarray)) and len(y_sig) == 1:
        y_sig_block = np.full(len(blocks), y_sig[0], dtype=float)
    else:
        y_sig_block = y_sig

    # Calculate the electrical characteristics of the rails
    gamma_sig_block = np.sqrt(z_sig * y_sig_block)
    gamma_trac_block = np.sqrt(z_trac * y_trac_block)
    z0_sig_block = np.sqrt(z_sig / y_sig_block)
    z0_trac_block = np.sqrt(z_trac / y_trac_block)

    # Add cross bonds and axles which split the blocks into sub blocks
    pos_cb = np.arange(0.401, blocks_sum[-1], 0.4)  # Position of the cross bonds
    # Note: 'a' and 'b' are used to identify the opposite directions of travel in this network (two-track)
    trac_sub_block_sum_a = np.sort(np.insert(np.concatenate((blocks_sum, pos_cb, axle_pos_a)), 0, 0))  # Traction rail connects to axles and cross bonds
    sig_sub_block_sum_a = np.sort(np.insert(np.concatenate((blocks_sum, blocks_sum[:-1], axle_pos_a)), 0, 0))  # Signal rail connects to axles, but need to add points on either side or IRJ
    trac_sub_block_sum_b = np.sort(np.insert(np.concatenate((blocks_sum, pos_cb, axle_pos_b)), 0, 0))
    sig_sub_block_sum_b = np.sort(np.insert(np.concatenate((blocks_sum, blocks_sum[:-1], axle_pos_b)), 0, 0))
    trac_sub_blocks_a = np.diff(trac_sub_block_sum_a)
    sig_sub_blocks_a = np.diff(sig_sub_block_sum_a)
    sig_sub_blocks_a[sig_sub_blocks_a == 0] = np.nan  # Sets a nan value to indicate the IRJ gap
    trac_sub_blocks_b = np.diff(trac_sub_block_sum_b)
    sig_sub_blocks_b = np.diff(sig_sub_block_sum_b)
    sig_sub_blocks_b[sig_sub_blocks_b == 0] = np.nan

    # Announce if cross bonds overlap with block boundaries
    if 0 in trac_sub_blocks_a or 0 in trac_sub_blocks_b:
        print('crossbond overlaps with block boundary, this will generate errors')
    else:
        pass

    # Set sub block values for z0 and gamma
    block_idx = np.searchsorted(blocks_sum, np.cumsum(trac_sub_blocks_a))
    z0_trac_sub_block_a = z0_trac_block[block_idx]
    gamma_trac_sub_block_a = gamma_trac_block[block_idx]

    block_idx = np.searchsorted(blocks_sum, np.cumsum(sig_sub_blocks_a[~np.isnan(sig_sub_blocks_a)]))
    z0_sig_sub_block_a = []
    gamma_sig_sub_block_a = []
    prev_idx = None
    for i in block_idx:
        if prev_idx is not None and i != prev_idx:
            z0_sig_sub_block_a.append(np.nan)
            gamma_sig_sub_block_a.append(np.nan)
        z0_sig_sub_block_a.append(z0_sig_block[i])
        gamma_sig_sub_block_a.append(gamma_sig_block[i])
        prev_idx = i
    z0_sig_sub_block_a = np.array(z0_sig_sub_block_a)
    gamma_sig_sub_block_a = np.array(gamma_sig_sub_block_a)

    block_idx = np.searchsorted(blocks_sum, np.cumsum(trac_sub_blocks_b))
    z0_trac_sub_block_b = z0_trac_block[block_idx]
    gamma_trac_sub_block_b = gamma_trac_block[block_idx]

    block_idx = np.searchsorted(blocks_sum, np.cumsum(sig_sub_blocks_b[~np.isnan(sig_sub_blocks_b)]))
    z0_sig_sub_block_b = []
    gamma_sig_sub_block_b = []
    prev_idx = None
    for i in block_idx:
        if prev_idx is not None and i != prev_idx:
            z0_sig_sub_block_b.append(np.nan)
            gamma_sig_sub_block_b.append(np.nan)
        z0_sig_sub_block_b.append(z0_sig_block[i])
        gamma_sig_sub_block_b.append(gamma_sig_block[i])
        prev_idx = i
    z0_sig_sub_block_b = np.array(z0_sig_sub_block_b)
    gamma_sig_sub_block_b = np.array(gamma_sig_sub_block_b)

    # Set up equivalent-pi parameters
    ye_trac_a = 1 / (z0_trac_sub_block_a * np.sinh(gamma_trac_sub_block_a * trac_sub_blocks_a))  # Series admittance for traction rail
    ye_trac_b = 1 / (z0_trac_sub_block_b * np.sinh(gamma_trac_sub_block_b * trac_sub_blocks_b))
    yg_trac = (np.cosh(gamma_trac_block * blocks) - 1) * (1 / (
                z0_trac_block * np.sinh(gamma_trac_block * blocks)))  # Parallel admittance for traction rail

    ye_sig_a = 1 / (z0_sig_sub_block_a * np.sinh(gamma_sig_sub_block_a * sig_sub_blocks_a))  # Series admittance for Signal rail
    ye_sig_b = 1 / (z0_sig_sub_block_b * np.sinh(gamma_sig_sub_block_b * sig_sub_blocks_b))
    yg_sig = (np.cosh(gamma_sig_block * blocks) - 1) * (1 / (z0_sig_block * np.sinh(gamma_sig_block * blocks)))  # Parallel admittance for Signal rail
    yg_trac_comb = np.empty(len(yg_trac) + 1)
    yg_trac_comb[0] = yg_trac[0]
    yg_trac_comb[1:-1] = yg_trac[:-1] + yg_trac[1:]
    yg_trac_comb[-1] = yg_trac[-1]

    # Calculate numbers of nodes ready to use in indexing
    n_nodes_a = len(trac_sub_block_sum_a) + len(sig_sub_block_sum_a)  # Number of nodes in direction of travel a
    n_nodes_b = len(trac_sub_block_sum_b) + len(sig_sub_block_sum_b)
    n_nodes = n_nodes_a + n_nodes_b  # Number of nodes in the whole network
    n_nodes_trac_a = len(trac_sub_block_sum_a)  # Number of nodes in the traction rail
    n_nodes_trac_b = len(trac_sub_block_sum_b)
    # n_nodes_sig_a = len(sig_sub_block_sum_a)  # Number of nodes in the Signal rail
    # n_nodes_sig_b = len(sig_sub_block_sum_b)

    # Index of rail nodes in the rails
    node_locs_trac_a = np.arange(0, n_nodes_trac_a, 1).astype(int)
    node_locs_sig_a = np.arange(n_nodes_trac_a, n_nodes_a, 1).astype(int)
    node_locs_trac_b = np.arange(n_nodes_a, n_nodes_a + n_nodes_trac_b, 1).astype(int)
    node_locs_sig_b = np.arange(n_nodes_a + n_nodes_trac_b, n_nodes).astype(int)
    # Index of cross bond nodes
    node_locs_cb_a = node_locs_trac_a[np.where(np.isin(trac_sub_block_sum_a, pos_cb))[0]]
    node_locs_cb_b = node_locs_trac_b[np.where(np.isin(trac_sub_block_sum_b, pos_cb))[0]]
    # Index of axle nodes
    node_locs_axle_trac_a = node_locs_trac_a[np.where(np.isin(trac_sub_block_sum_a, axle_pos_a))[0]]
    node_locs_axle_sig_a = node_locs_sig_a[np.where(np.isin(sig_sub_block_sum_a, axle_pos_a))[0]]
    node_locs_axle_trac_b = node_locs_trac_b[np.where(np.isin(trac_sub_block_sum_b, axle_pos_b))[0]]
    node_locs_axle_sig_b = node_locs_sig_b[np.where(np.isin(sig_sub_block_sum_b, axle_pos_b))[0]]

    # Index of traction rail power supply and relay nodes
    # Note: 'a' begins with a relay and ends with a power supply, 'b' begins with a power supply and ends with a relay
    node_locs_no_cb_axle_trac_a = np.delete(node_locs_trac_a, np.where(np.isin(node_locs_trac_a, np.concatenate((node_locs_cb_a, node_locs_axle_trac_a))))[0])
    node_locs_power_trac_a = node_locs_no_cb_axle_trac_a[1:]
    node_locs_relay_trac_a = node_locs_no_cb_axle_trac_a[:-1]
    node_locs_no_cb_axle_trac_b = np.delete(node_locs_trac_b, np.where(np.isin(node_locs_trac_b, np.concatenate((node_locs_cb_b, node_locs_axle_trac_b))))[0])
    node_locs_power_trac_b = node_locs_no_cb_axle_trac_b[:-1]
    node_locs_relay_trac_b = node_locs_no_cb_axle_trac_b[1:]
    node_locs_no_cb_axle_sig_a = np.delete(node_locs_sig_a, np.where(np.isin(node_locs_sig_a, node_locs_axle_sig_a))[0])
    node_locs_power_sig_a = node_locs_no_cb_axle_sig_a[1::2]
    node_locs_relay_sig_a = node_locs_no_cb_axle_sig_a[0::2]
    node_locs_no_cb_axle_sig_b = np.delete(node_locs_sig_b, np.where(np.isin(node_locs_sig_b, node_locs_axle_sig_b))[0])
    node_locs_power_sig_b = node_locs_no_cb_axle_sig_b[0::2]
    node_locs_relay_sig_b = node_locs_no_cb_axle_sig_b[1::2]

    # Calculate nodal parallel admittances and sum of admittances into the node
    # Direction 'a' first
    # Traction rail
    y_sum = np.full(n_nodes, 69.420, dtype=float)  # Array of sum of admittances into the node
    # First node
    mask_first_trac_a = np.isin(node_locs_trac_a, node_locs_trac_a[0])
    first_trac_a = node_locs_trac_a[mask_first_trac_a]
    locs_first_trac_a = np.where(np.isin(node_locs_trac_a, first_trac_a))[0]
    y_sum[first_trac_a] = yg_trac_comb[0] + y_relay + ye_trac_a[locs_first_trac_a]
    # Axles
    locs_axle_trac_a = np.where(np.isin(node_locs_trac_a, node_locs_axle_trac_a))[0]
    y_sum[node_locs_axle_trac_a] = y_axle + ye_trac_a[locs_axle_trac_a - 1] + ye_trac_a[locs_axle_trac_a]
    # Cross bonds
    locs_cb_a = np.where(np.isin(node_locs_trac_a, node_locs_cb_a))[0]
    y_sum[node_locs_cb_a] = y_cb + ye_trac_a[locs_cb_a - 1] + ye_trac_a[locs_cb_a]
    # Middle nodes
    indices_other_node_trac_a = node_locs_trac_a[1:-1][~np.logical_or(np.isin(node_locs_trac_a[1:-1], node_locs_axle_trac_a), np.isin(node_locs_trac_a[1:-1], node_locs_cb_a))]
    mask_other_trac_a = np.isin(indices_other_node_trac_a, node_locs_trac_a)
    other_trac_a = indices_other_node_trac_a[mask_other_trac_a]
    locs_other_trac_a = np.where(np.isin(node_locs_trac_a, other_trac_a))[0]
    y_sum[other_trac_a] = yg_trac_comb[1:-1] + y_power + y_relay + ye_trac_a[locs_other_trac_a - 1] + ye_trac_a[locs_other_trac_a]
    # Last node
    mask_last_trac_a = np.isin(node_locs_trac_a, node_locs_trac_a[-1])
    last_trac_a = node_locs_trac_a[mask_last_trac_a]
    locs_last_trac_a = np.where(np.isin(node_locs_trac_a, last_trac_a))[0]
    y_sum[last_trac_a] = yg_trac_comb[-1] + y_power + ye_trac_a[locs_last_trac_a - 1]
    # Signal rail
    # Relay nodes
    locs_relay_sig_a = np.where(np.isin(node_locs_sig_a, node_locs_relay_sig_a))[0]
    y_sum[node_locs_relay_sig_a] = yg_sig + y_relay + ye_sig_a[locs_relay_sig_a]
    # Power nodes
    locs_power_sig_a = np.where(np.isin(node_locs_sig_a, node_locs_power_sig_a))[0]
    y_sum[node_locs_power_sig_a] = yg_sig + y_power + ye_sig_a[locs_power_sig_a - 1]
    # Axle nodes
    axle_locs = np.where(np.isin(node_locs_sig_a, node_locs_axle_sig_a))[0]
    y_sum[node_locs_axle_sig_a] = y_axle + ye_sig_a[axle_locs - 1] + ye_sig_a[axle_locs]
    # Direction 'b' second
    # Traction rail
    # First node
    mask_first_trac_b = np.isin(node_locs_trac_b, node_locs_trac_b[0])
    first_trac_b = node_locs_trac_b[mask_first_trac_b]
    locs_first_trac_b = np.where(np.isin(node_locs_trac_b, first_trac_b))[0]
    y_sum[first_trac_b] = yg_trac_comb[0] + y_relay + ye_trac_b[locs_first_trac_b]
    # Axles
    locs_axle_trac_b = np.where(np.isin(node_locs_trac_b, node_locs_axle_trac_b))[0]
    y_sum[node_locs_axle_trac_b] = y_axle + ye_trac_b[locs_axle_trac_b - 1] + ye_trac_b[locs_axle_trac_b]
    # Cross bonds
    locs_cb_b = np.where(np.isin(node_locs_trac_b, node_locs_cb_b))[0]
    y_sum[node_locs_cb_b] = y_cb + ye_trac_b[locs_cb_b - 1] + ye_trac_b[locs_cb_b]
    # Middle nodes
    indices_other_node_trac_b = node_locs_trac_b[1:-1][~np.logical_or(np.isin(node_locs_trac_b[1:-1], node_locs_axle_trac_b), np.isin(node_locs_trac_b[1:-1], node_locs_cb_b))]
    mask_other_trac_b = np.isin(indices_other_node_trac_b, node_locs_trac_b)
    other_trac_b = indices_other_node_trac_b[mask_other_trac_b]
    locs_other_trac_b = np.where(np.isin(node_locs_trac_b, other_trac_b))[0]
    y_sum[other_trac_b] = yg_trac_comb[1:-1] + y_power + y_relay + ye_trac_b[locs_other_trac_b - 1] + ye_trac_b[locs_other_trac_b]
    # Last node
    mask_last_trac_b = np.isin(node_locs_trac_b, node_locs_trac_b[-1])
    last_trac_b = node_locs_trac_b[mask_last_trac_b]
    locs_last_trac_b = np.where(np.isin(node_locs_trac_b, last_trac_b))[0]
    y_sum[last_trac_b] = yg_trac_comb[-1] + y_power + ye_trac_b[locs_last_trac_b - 1]
    # Signal rail
    # Relay nodes
    locs_relay_sig_b = np.where(np.isin(node_locs_sig_b, node_locs_relay_sig_b))[0]
    y_sum[node_locs_relay_sig_b] = yg_sig + y_relay + ye_sig_b[locs_relay_sig_b - 1]
    # Power nodes
    locs_power_sig_b = np.where(np.isin(node_locs_sig_b, node_locs_power_sig_b))[0]
    y_sum[node_locs_power_sig_b] = yg_sig + y_power + ye_sig_b[locs_power_sig_b]
    # Axle nodes
    axle_locs = np.where(np.isin(node_locs_sig_b, node_locs_axle_sig_b))[0]
    y_sum[node_locs_axle_sig_b] = y_axle + ye_sig_b[axle_locs - 1] + ye_sig_b[axle_locs]

    # Check that all values were filled correctly
    if 69.420 in y_sum:
        raise ValueError(f"y_sum configured incorrectly")

    # Build admittance matrix
    Y = np.zeros((n_nodes, n_nodes))
    # Diagonal values
    Y[range(0, n_nodes), range(0, n_nodes)] = y_sum
    # Series admittances between nodes
    Y[node_locs_trac_a[:-1], node_locs_trac_a[1:]] = -ye_trac_a
    Y[node_locs_trac_a[1:], node_locs_trac_a[:-1]] = -ye_trac_a
    Y[node_locs_sig_a[:-1], node_locs_sig_a[1:]] = -ye_sig_a
    Y[node_locs_sig_a[1:], node_locs_sig_a[:-1]] = -ye_sig_a
    Y[node_locs_trac_b[:-1], node_locs_trac_b[1:]] = -ye_trac_b
    Y[node_locs_trac_b[1:], node_locs_trac_b[:-1]] = -ye_trac_b
    Y[node_locs_sig_b[:-1], node_locs_sig_b[1:]] = -ye_sig_b
    Y[node_locs_sig_b[1:], node_locs_sig_b[:-1]] = -ye_sig_b
    # Relay admittances
    Y[node_locs_relay_trac_a, node_locs_relay_sig_a] = -y_relay
    Y[node_locs_relay_sig_a, node_locs_relay_trac_a] = -y_relay
    Y[node_locs_relay_trac_b, node_locs_relay_sig_b] = -y_relay
    Y[node_locs_relay_sig_b, node_locs_relay_trac_b] = -y_relay
    # Power admittances
    Y[node_locs_power_trac_a, node_locs_power_sig_a] = -y_power
    Y[node_locs_power_sig_a, node_locs_power_trac_a] = -y_power
    Y[node_locs_power_trac_b, node_locs_power_sig_b] = -y_power
    Y[node_locs_power_sig_b, node_locs_power_trac_b] = -y_power
    # Cross bond admittances
    Y[node_locs_cb_a, node_locs_cb_b] = -y_cb
    Y[node_locs_cb_b, node_locs_cb_a] = -y_cb
    # Axle admittances
    Y[node_locs_axle_trac_a, node_locs_axle_sig_a] = -y_axle
    Y[node_locs_axle_sig_a, node_locs_axle_trac_a] = -y_axle
    Y[node_locs_axle_trac_b, node_locs_axle_sig_b] = -y_axle
    Y[node_locs_axle_sig_b, node_locs_axle_trac_b] = -y_axle

    Y[np.isnan(Y)] = 0
    np.save('y_model.npy', Y)

    # Restructure angles array based on the new sub blocks
    bearings_a = bearings
    bearings_b = (bearings + np.pi) % (2*np.pi)
    block_indices_trac_a = np.searchsorted(blocks_sum, np.delete(trac_sub_block_sum_a, 0, 0))
    trac_sub_blocks_angles_a = bearings_a[block_indices_trac_a]
    trac_sub_blocks_ex_a = ex_blocks[block_indices_trac_a]
    trac_sub_blocks_ey_a = ey_blocks[block_indices_trac_a]
    block_indices_sig_a = np.searchsorted(blocks_sum, np.delete(sig_sub_block_sum_a, locs_relay_sig_a))
    sig_sub_block_angles_a = bearings_a[block_indices_sig_a]
    sig_sub_blocks_ex_a = ex_blocks[block_indices_sig_a]
    sig_sub_blocks_ey_a = ey_blocks[block_indices_sig_a]
    block_indices_trac_b = np.searchsorted(blocks_sum, np.delete(trac_sub_block_sum_b, 0, 0))
    trac_sub_blocks_angles_b = bearings_b[block_indices_trac_b]
    trac_sub_blocks_ex_b = ex_blocks[block_indices_trac_b]
    trac_sub_blocks_ey_b = ey_blocks[block_indices_trac_b]
    block_indices_sig_b = np.searchsorted(blocks_sum, np.delete(sig_sub_block_sum_b, locs_power_sig_b))
    sig_sub_block_angles_b = bearings_b[block_indices_sig_b]
    sig_sub_blocks_ex_b = ex_blocks[block_indices_sig_b]
    sig_sub_blocks_ey_b = ey_blocks[block_indices_sig_b]

    # Currents
    # Set up current matrix
    J = np.zeros([len(ex_blocks[0, :]), n_nodes])

    # 'a' first
    trac_sb_angles_a_broadcasted = trac_sub_blocks_angles_a[:, np.newaxis]
    e_x_par_trac_a = trac_sub_blocks_ex_a * np.cos(trac_sb_angles_a_broadcasted)
    e_x_par_trac_a = e_x_par_trac_a.T
    e_y_par_trac_a = trac_sub_blocks_ey_a * np.sin(trac_sb_angles_a_broadcasted)
    e_y_par_trac_a = e_y_par_trac_a.T
    e_par_trac_a = e_x_par_trac_a + e_y_par_trac_a
    sig_sb_angles_a_broadcasted = sig_sub_block_angles_a[:, np.newaxis]
    e_x_par_sig_a = sig_sub_blocks_ex_a * np.cos(sig_sb_angles_a_broadcasted)
    e_x_par_sig_a = e_x_par_sig_a.T
    e_y_par_sig_a = sig_sub_blocks_ey_a * np.sin(sig_sb_angles_a_broadcasted)
    e_y_par_sig_a = e_y_par_sig_a.T
    e_par_sig_a = e_x_par_sig_a + e_y_par_sig_a
    i_sig_a = e_par_sig_a / z_sig
    i_trac_a = e_par_trac_a / z_trac

    # 'b' second
    trac_sb_angles_b_broadcasted = trac_sub_blocks_angles_b[:, np.newaxis]
    e_x_par_trac_b = trac_sub_blocks_ex_b * np.cos(trac_sb_angles_b_broadcasted)
    e_x_par_trac_b = e_x_par_trac_b.T
    e_y_par_trac_b = trac_sub_blocks_ey_b * np.sin(trac_sb_angles_b_broadcasted)
    e_y_par_trac_b = e_y_par_trac_b.T
    e_par_trac_b = e_x_par_trac_b + e_y_par_trac_b
    sig_sb_angles_b_broadcasted = sig_sub_block_angles_b[:, np.newaxis]
    e_x_par_sig_b = sig_sub_blocks_ex_b * np.cos(sig_sb_angles_b_broadcasted)
    e_x_par_sig_b = e_x_par_sig_b.T
    e_y_par_sig_b = sig_sub_blocks_ey_b * np.sin(sig_sb_angles_b_broadcasted)
    e_y_par_sig_b = e_y_par_sig_b.T
    e_par_sig_b = e_x_par_sig_b + e_y_par_sig_b
    i_sig_b = e_par_sig_b / z_sig
    i_trac_b = e_par_trac_b / z_trac

    # 'a' first
    # Traction rail first node
    J[:, node_locs_trac_a[0]] = -i_trac_a[:, 0]
    # Traction rail centre nodes
    # Cross bond nodes
    mask = np.isin(node_locs_trac_a, node_locs_cb_a)
    indices = np.where(mask)[0]
    J[:, node_locs_cb_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices]
    # Axle nodes
    mask = np.isin(node_locs_trac_a, node_locs_axle_trac_a)
    indices = np.where(mask)[0]
    J[:, node_locs_axle_trac_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices]
    # Non-cross bond or axle nodes
    mask = np.isin(node_locs_trac_a, node_locs_cb_a) | np.isin(node_locs_trac_a, node_locs_axle_trac_a)
    indices = np.where(~mask)[0][1:-1]
    mask_del = ~np.isin(node_locs_trac_a, node_locs_cb_a) & ~np.isin(node_locs_trac_a, node_locs_axle_trac_a)
    non_cb_axle_node_locs_centre_a = node_locs_trac_a[mask_del][1:-1]
    J[:, non_cb_axle_node_locs_centre_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices] - i_power
    # Traction rail last node
    J[:, node_locs_trac_a[-1]] = i_trac_a[:, -1] - i_power

    # Signal rail nodes
    sig_relay_axle = node_locs_sig_a[np.where(~np.isin(node_locs_sig_a, node_locs_power_sig_a))[0]]
    split_blocks = np.unique(np.sort(np.append(np.where(np.isin(sig_relay_axle, node_locs_axle_sig_a))[0], np.where(np.isin(sig_relay_axle, node_locs_axle_sig_a))[0] - 1)))
    all_blocks = range(0, len(i_sig_a[0]))
    whole_blocks = np.where(~np.isin(all_blocks, split_blocks))[0]
    whole_blocks_start = sig_relay_axle[whole_blocks]
    whole_blocks_end = whole_blocks_start + 1
    split_blocks_start = sig_relay_axle[np.where(~np.isin(sig_relay_axle, node_locs_axle_sig_a) & ~np.isin(sig_relay_axle, whole_blocks_start))[0]]
    split_blocks_end = np.delete(node_locs_power_sig_a, np.where(np.isin(node_locs_power_sig_a, whole_blocks_end)))
    split_blocks_mid = sig_relay_axle[np.where(np.isin(sig_relay_axle, node_locs_axle_sig_a))[0]]
    J[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, whole_blocks_start))[0]]] = -i_sig_a[:, whole_blocks]
    J[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, whole_blocks_end))[0]]] = i_sig_a[:, whole_blocks] + i_power
    J[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, split_blocks_start))[0]]] = -i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_start))[0]]
    J[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, split_blocks_end))[0]]] = i_sig_a[:, split_blocks[np.where(~np.isin(split_blocks, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0] - 1))[0]]] + i_power
    J[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, split_blocks_mid))[0]]] = i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0] - 1] - i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0]]
    # 'b' second
    # Traction rail first node
    J[:, node_locs_trac_b[0]] = i_trac_b[:, 0] - i_power
    # Traction rail centre nodes
    # Cross bond nodes
    mask = np.isin(node_locs_trac_b, node_locs_cb_b)
    indices = np.where(mask)[0]
    J[:, node_locs_cb_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1]
    # Axle nodes
    mask = np.isin(node_locs_trac_b, node_locs_axle_trac_b)
    indices = np.where(mask)[0]
    J[:, node_locs_axle_trac_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1]
    # Non-cross bond or axle nodes
    mask = np.isin(node_locs_trac_b, node_locs_cb_b) | np.isin(node_locs_trac_b, node_locs_axle_trac_b)
    indices = np.where(~mask)[0][1:-1]
    mask_del = ~np.isin(node_locs_trac_b, node_locs_cb_b) & ~np.isin(node_locs_trac_b, node_locs_axle_trac_b)
    non_cb_axle_node_locs_centre_b = node_locs_trac_b[mask_del][1:-1]
    J[:, non_cb_axle_node_locs_centre_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1] - i_power
    # Traction rail last node
    J[:, node_locs_trac_b[-1]] = -i_trac_b[:, -1]

    # Signal rail nodes
    sig_power_axle = node_locs_sig_b[np.where(~np.isin(node_locs_sig_b, node_locs_relay_sig_b))[0]]
    split_blocks = np.unique(np.sort(np.append(np.where(np.isin(sig_power_axle, node_locs_axle_sig_b))[0], np.where(np.isin(sig_power_axle, node_locs_axle_sig_b))[0] - 1)))
    all_blocks = range(0, len(i_sig_b[0]))
    whole_blocks = np.where(~np.isin(all_blocks, split_blocks))[0]
    whole_blocks_start = sig_power_axle[whole_blocks]
    whole_blocks_end = whole_blocks_start + 1
    split_blocks_start = sig_power_axle[np.where(~np.isin(sig_power_axle, node_locs_axle_sig_b) & ~np.isin(sig_power_axle, whole_blocks_start))[0]]
    split_blocks_end = np.delete(node_locs_relay_sig_b, np.where(np.isin(node_locs_relay_sig_b, whole_blocks_end)))
    split_blocks_mid = sig_power_axle[np.where(np.isin(sig_power_axle, node_locs_axle_sig_b))[0]]
    J[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, whole_blocks_start))[0]]] = i_sig_b[:, whole_blocks] + i_power
    J[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, whole_blocks_end))[0]]] = -i_sig_b[:, whole_blocks]
    J[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, split_blocks_start))[0]]] = i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_start))[0]] + i_power
    J[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, split_blocks_end))[0]]] = -i_sig_b[:, split_blocks[np.where(~np.isin(split_blocks, np.where(np.isin(sig_power_axle, split_blocks_mid))[0] - 1))[0]]]
    J[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, split_blocks_mid))[0]]] = -i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_mid))[0] - 1] + i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_mid))[0]]

    # Electrically stagger the power supplies
    if electrical_staggering:
        node_locs_power_trac_a_down = node_locs_power_trac_a[1::2]
        node_locs_power_sig_a_down = node_locs_power_sig_a[1::2]
        node_locs_power_trac_b_down = np.flip(np.flip(node_locs_power_trac_b)[1::2])
        node_locs_power_sig_b_down = np.flip(np.flip(node_locs_power_sig_b)[1::2])
        J[:, node_locs_power_trac_a_down] = J[:, node_locs_power_trac_a_down] + (2 * i_power)
        J[:, node_locs_power_sig_a_down] = J[:, node_locs_power_sig_a_down] - (2 * i_power)
        J[:, node_locs_power_trac_b_down] = J[:, node_locs_power_trac_b_down] + (2 * i_power)
        J[:, node_locs_power_sig_b_down] = J[:, node_locs_power_sig_b_down] - (2 * i_power)
    else:
        pass
    J = J.T

    # Sparse matrix
    y_csc = csr_matrix(Y).tocsc()
    factor = cholesky(y_csc)
    V = factor(J)

    # Calculate relay voltages and currents
    # 'a' first
    v_relay_top_node_a = V[node_locs_relay_sig_a]
    v_relay_bottom_node_a = V[node_locs_relay_trac_a]
    v_relay_a = v_relay_top_node_a - v_relay_bottom_node_a

    # 'b' first
    v_relay_top_node_b = V[node_locs_relay_sig_b]
    v_relay_bottom_node_b = V[node_locs_relay_trac_b]
    v_relay_b = v_relay_top_node_b - v_relay_bottom_node_b

    i_relays_a = v_relay_a * y_relay
    i_relays_b = v_relay_b * y_relay

    return {
        'i_relays_a': i_relays_a,
        'i_relays_b': i_relays_b,
        'V': V,
        'node_locs_relay_trac_a': node_locs_relay_trac_a,
        'node_locs_relay_trac_b': node_locs_relay_trac_b,
        'node_locs_relay_sig_a': node_locs_relay_sig_a,
        'node_locs_relay_sig_b': node_locs_relay_sig_b,
    }


def model_refactor(section_name, axle_pos_a=None, axle_pos_b=None, ex_blocks=None, ey_blocks=None, ex_uniform=None, ey_uniform=None, electrical_staggering=True, **kwargs):
    # Define network parameters
    if axle_pos_a is None:  # Check if 'a' direction axles are specified, if not, default to empty list
        axle_pos_a = []
    if axle_pos_b is None:  # Check if 'b' direction axles are specified, if not, default to empty list
        axle_pos_b = []
    z_sig = kwargs.get('z_sig', 0.0289)  # Signal rail series impedance (ohms/km); optional kwarg
    z_trac = kwargs.get('z_trac', 0.0289)  # Traction rail series impedance (ohms/km); optional kwarg
    i_power = kwargs.get('i_power', 10/7.2)  # Track circuit power supply equivalent current source (amps); optional kwarg
    y_power = 1/7.2  # Track circuit power supply admittance (siemens)
    y_cb = 1/1e-3  # Cross bond admittance (siemens)
    y_axle = 1/251e-4  # Axle admittance (siemens)
    relay_type = kwargs.get('relay_type', 'BR939A')  # Check if relay type is specified, if not, default to BR939A; optional kwarg
    relays_admittances = {  # Track circuit relay admittances (siemens) depending on relay type
        'BR939A': 1 / 20,
        'BR966F2': 1 / 9,
        'BR966F9': 1 / 60
    }
    # Raise error if an unknown relay type is specified
    try:
        y_relay = relays_admittances[relay_type]
    except KeyError:
        raise ValueError(f'Relay type {relay_type} not recognised. Please choose from BR939A, BR966F2, or BR966F9.')

    # Load in the lengths and bearings of the track circuit blocks
    data = np.load(f'data/rail_data/{section_name}/{section_name}_distances_bearings.npz')
    blocks = data['distances']  # Block lengths (km)
    print('OVERRIDING BLOCKS AND BEARINGS')
    #blocks = np.full(3, 1.0)
    print(f'Number of blocks: {len(blocks)}')
    bearings = data['bearings']  # Block bearings (radians); Note: zero is directly northwards, with positive values increasing clockwise
   # bearings = np.array([np.deg2rad(90), np.deg2rad(85), np.deg2rad(95)])
    blocks_sum = np.cumsum(blocks)  # Cumulative sum of block lengths (km)

    # Identify the electric field components (ex and ey) inputs and setup accordingly
    # The model expects each electric field component to be provided as a range of values for each block and must be converted if not
    e_blocks_given = ex_blocks is not None and ey_blocks is not None  # Boolean for if both components of blocks values are input
    e_uni_given = ex_uniform is not None and ey_uniform is not None  # Boolean for if both components of uniform values are input
    if e_blocks_given and ex_uniform is None and ey_uniform is None:  # If only both blocks components are given
        pass  # Use as the electric field components inputs
    elif e_uni_given and ex_blocks is None and ey_blocks is None:  # If only both uniform components  are given
        ex_blocks = np.full((len(blocks), len(ex_uniform)), ex_uniform, dtype=float)  # Set the electric field component value for each block to the uniform value
        ey_blocks = np.full((len(blocks), len(ey_uniform)), ey_uniform, dtype=float)
        ey_blocks = np.full((len(blocks), len(ey_uniform)), ey_uniform, dtype=float)
    else:  # If inputs are mixed or incomplete
        raise ValueError("You must provide either (ex_blocks and ey_blocks) OR (ex_par and ey_par) exclusively.")  # Raise error

    # Identify the traction rail leakage inputs (if any) and setup accordingly
    y_trac = kwargs.get('y_trac', None)  # Check if traction rail leakage is specified, otherwise default to None; optional kwarg
    if y_trac is None:  # Default traction rail leakage setup from BGS resistivity model
        block_leakages = np.load(f'data/rail_data/{section_name}/{section_name}_block_leakage.npz')
        leakage_profile = kwargs.get('leakage_profile', 'a50')  # Check if leakage profile is specified, otherwise use median (a50) profile
        y_trac_block = block_leakages[leakage_profile]
    elif isinstance(y_trac, (int, float)):  # If input is a single int or float
        y_trac_block = np.full(len(blocks), y_trac, dtype=float)  # Set each block to that value and ensure float type
    elif isinstance(y_trac, (list, tuple, np.ndarray)) and len(y_trac) == 1:  # If input is a list, tuple, or numpy array but there is only a single value within
        y_trac_block = np.full(len(blocks), y_trac[0], dtype=float)  # Extract and set each block to that value and ensure float type
    elif isinstance(y_trac, (list, tuple, np.ndarray)) and len(y_trac) == len(blocks):  # If input is a list, tuple, or numpy array and the length matches the number of blocks
        y_trac_block = np.asarray(y_trac, dtype=float)  # Set to those values and ensure float type
    else:  # If traction rail leakage is incorrectly specified
        raise ValueError('Check traction rail leakage (y_trac) inputs.')  # Raise error

    # Identify the signal rail leakage inputs (if any) and setup accordingly
    y_sig = kwargs.get('y_sig', None)  # Check if signal rail leakage is specified, otherwise default to None; optional kwarg
    if y_sig is None:  # Default signal rail leakage values for moderate conditions
        y_sig_block = np.full(len(blocks), 0.1, dtype=float)
    elif isinstance(y_sig, (int, float)):  # If input is a single int or float
        y_sig_block = np.full(len(blocks), y_sig, dtype=float)  # Set each block to that value and ensure float type
    elif isinstance(y_sig, (list, tuple, np.ndarray)) and len(y_sig) == 1:  # If input is a list, tuple, or numpy array but there is only a single value within
        y_sig_block = np.full(len(blocks), y_sig[0], dtype=float)  # Extract and set each block to that value and ensure float type
    elif isinstance(y_sig, (list, tuple, np.ndarray)) and len(y_sig) == len(blocks):  # If input is a list, tuple, or numpy array and the length matches the number of blocks
        y_sig_block = np.asarray(y_trac, dtype=float)  # Set to those values and ensure float type
    else:  # If signal rail leakage is incorrectly specified
        raise ValueError('Check signal rail leakage (y_sig) inputs.')  # Raise error

    # Axles that overlap IRJs must be ignored as they are not part of the circuit
    if np.isin(axle_pos_a, blocks_sum).any():
        overlap_locs = np.where(np.isin(axle_pos_a, blocks_sum))[0]
        axle_pos_a = np.delete(axle_pos_a, overlap_locs)
        print("An axle overlapped with an IRJ and was ignored")  # Flag that this was done
    if np.isin(axle_pos_b, blocks_sum).any():
        overlap_locs = np.where(np.isin(axle_pos_b, blocks_sum))[0]
        axle_pos_b = np.delete(axle_pos_b, overlap_locs)
        print("An axle overlapped with an IRJ and was ignored")

    # Calculate the electrical characteristics of the rails
    gamma_trac_block = np.sqrt(z_trac * y_trac_block)  # Traction rail propagation constant
    gamma_sig_block = np.sqrt(z_sig * y_sig_block)  # Signal rail propagation constant
    z0_trac_block = np.sqrt(z_trac / y_trac_block)  # Traction rail characteristic impedance
    z0_sig_block = np.sqrt(z_sig / y_sig_block)   # Signal rail characteristic impedance

    # Add cross bonds and axles which split the blocks into sub blocks
    # Note: '_a' and '_b' are used to identify the opposite directions of travel in this network (two-track)
    pos_cb = np.arange(0.4, blocks_sum[-1], 0.4)  # Crossbond positions, spaced every 400m
    trac_sub_block_sum_a = np.unique(np.concatenate(([0.0], blocks_sum, pos_cb, axle_pos_a)))  # Total distance along the traction rail of block ends, crossbond connections, and axle connections
    trac_sub_blocks_a = np.diff(trac_sub_block_sum_a)  # Lengths of every traction rail sub block
    sig_sub_block_sum_a = np.sort(np.concatenate(([0.0], blocks_sum, blocks_sum[:-1], axle_pos_a)))  # Total distance along the signal rail of block ends and axle connections. The insulated rail joints (IRJs) are represented by the end of one block being the start of the next
    sig_sub_blocks_a = np.diff(sig_sub_block_sum_a)  # Lengths of every signal rail sub block
    sig_sub_blocks_a[sig_sub_blocks_a == 0] = np.nan  # Sets the zero lengths sub blocks as nan to indicate the IRJ gap
    # Repeat for opposite track
    trac_sub_block_sum_b = np.unique(np.concatenate(([0.0], blocks_sum, pos_cb, axle_pos_b)))
    trac_sub_blocks_b = np.diff(trac_sub_block_sum_b)
    sig_sub_block_sum_b = np.sort(np.concatenate(([0.0], blocks_sum, blocks_sum[:-1], axle_pos_b)))
    sig_sub_blocks_b = np.diff(sig_sub_block_sum_b)
    sig_sub_blocks_b[sig_sub_blocks_b == 0] = np.nan

    # Set sub block values for z0 and gamma
    block_idx = np.searchsorted(blocks_sum, np.cumsum(trac_sub_blocks_a))  # Indices of the blocks that the traction rail sub blocks are in
    z0_trac_sub_block_a = z0_trac_block[block_idx]  # Sets sub block traction rail characteristic impedance values based on block indices
    gamma_trac_sub_block_a = gamma_trac_block[block_idx]  # Sets sub block traction rail propagation constant values based on block indices
    block_idx = np.searchsorted(blocks_sum, np.nancumsum(sig_sub_blocks_a))  # Indices of the blocks that the signal rail sub blocks are in, ignoring nan values
    z0_sig_sub_block_a = z0_sig_block[block_idx]  # Sets sub block signal rail characteristic impedance values based on block indices
    z0_sig_sub_block_a[np.isnan(sig_sub_blocks_a)] = np.nan  # Replaces the nan values that indicate the IRJ gap
    gamma_sig_sub_block_a = gamma_sig_block[block_idx]  # Sets sub block signal rail propagation constant values based on block indices
    gamma_sig_sub_block_a[np.isnan(sig_sub_blocks_a)] = np.nan  # Replaces the nan values that indicate the IRJ gap
    # Repeat for opposite track
    block_idx = np.searchsorted(blocks_sum, np.cumsum(trac_sub_blocks_b))
    z0_trac_sub_block_b = z0_trac_block[block_idx]
    gamma_trac_sub_block_b = gamma_trac_block[block_idx]
    block_idx = np.searchsorted(blocks_sum, np.nancumsum(sig_sub_blocks_b))
    z0_sig_sub_block_b = z0_sig_block[block_idx]
    z0_sig_sub_block_b[np.isnan(sig_sub_blocks_b)] = np.nan
    gamma_sig_sub_block_b = gamma_sig_block[block_idx]
    gamma_sig_sub_block_b[np.isnan(sig_sub_blocks_b)] = np.nan

    # Set up equivalent-pi parameters
    ye_trac_a = 1 / (z0_trac_sub_block_a * np.sinh(gamma_trac_sub_block_a * trac_sub_blocks_a))  # Series admittance for traction rail
    half_yg_trac_a = (np.cosh(gamma_trac_sub_block_a * trac_sub_blocks_a) - 1) * (1 / (z0_trac_sub_block_a * np.sinh(gamma_trac_sub_block_a * trac_sub_blocks_a)))  # Half of the parallel admittance for traction rail. Keep halved as it is useful for later calculations
    ye_sig_a = 1 / (z0_sig_sub_block_a * np.sinh(gamma_sig_sub_block_a * sig_sub_blocks_a))  # Series admittance for Signal rail
    ye_sig_a[np.isnan(ye_sig_a)] = 0  # IRJs do not contribute to series admittance, so nan values can be set to zero as it is useful for later calculations
    half_yg_sig_a = (np.cosh(gamma_sig_sub_block_a * sig_sub_blocks_a) - 1) * (1 / (z0_sig_sub_block_a * np.sinh(gamma_sig_sub_block_a * sig_sub_blocks_a)))  # Half of the parallel admittance for Signal rail. Keep halved as it is useful for later calculations
    half_yg_sig_a[np.isnan(half_yg_sig_a)] = 0  # IRJs do not contribute to parallel admittance, so nan values can be set to zero as it is useful for later calculations
    # Repeat for opposite track
    ye_trac_b = 1 / (z0_trac_sub_block_b * np.sinh(gamma_trac_sub_block_b * trac_sub_blocks_b))
    half_yg_trac_b = (np.cosh(gamma_trac_sub_block_b * trac_sub_blocks_b) - 1) * (1 / (z0_trac_sub_block_b * np.sinh(gamma_trac_sub_block_b * trac_sub_blocks_b)))
    ye_sig_b = 1 / (z0_sig_sub_block_b * np.sinh(gamma_sig_sub_block_b * sig_sub_blocks_b))
    ye_sig_b[np.isnan(ye_sig_b)] = 0
    half_yg_sig_b = (np.cosh(gamma_sig_sub_block_b * sig_sub_blocks_b) - 1) * (1 / (z0_sig_sub_block_b * np.sinh(gamma_sig_sub_block_b * sig_sub_blocks_b)))
    half_yg_sig_b[np.isnan(half_yg_sig_b)] = 0

    yg_trac_nodal_a = np.full(len(half_yg_trac_a) + 1, np.nan)  # Parallel admittance needs to be converted to a nodal value. The traction rail has nodes equal to the number of sub blocks + 1
    yg_trac_nodal_a[1:-1] = half_yg_trac_a[:-1] + half_yg_trac_a[1:]  # Each middle traction rail node will have a contribution of half of the preceeding sub block's leakage and half of the next. We kept the values halved above, so now just add instead of halving again
    yg_trac_nodal_a[0] = half_yg_trac_a[0]  # The traction rail start node only gets leakage contributed from the first sub block
    yg_trac_nodal_a[-1] = half_yg_trac_a[-1]  # The traction rail end node only gets leakage contributed from the last sub block
    yg_sig_nodal_a = np.full(len(half_yg_sig_a) + 1, np.nan)
    yg_sig_nodal_a[1:-1] = half_yg_sig_a[:-1] + half_yg_sig_a[1:]  # Each signal rail node not at the start or end will have a contribution of half of the adjacent sub block's leakage and have nothing contributed by the IRJ. We kept the values halved above, so now just add instead of halving again
    yg_sig_nodal_a[0] = half_yg_sig_a[0]  # The signal rail start node only gets leakage contributed from the first sub block
    yg_sig_nodal_a[-1] = half_yg_sig_a[-1]  # The signal rail end node only gets leakage contributed from the last sub block
    # Repeat for opposite track
    yg_trac_nodal_b = np.full(len(half_yg_trac_b) + 1, np.nan)
    yg_trac_nodal_b[1:-1] = half_yg_trac_b[:-1] + half_yg_trac_b[1:]
    yg_trac_nodal_b[0] = half_yg_trac_b[0]
    yg_trac_nodal_b[-1] = half_yg_trac_b[-1]
    yg_sig_nodal_b = np.full(len(half_yg_sig_b) + 1, np.nan)
    yg_sig_nodal_b[1:-1] = half_yg_sig_b[:-1] + half_yg_sig_b[1:]
    yg_sig_nodal_b[0] = half_yg_sig_b[0]
    yg_sig_nodal_b[-1] = half_yg_sig_b[-1]
    # Flag errors if arrays are filled incorrectly
    if any(np.isnan(yg_trac_nodal_a)):
        raise ValueError('yg_trac_nodal_a contains nan values')
    elif any(np.isnan(yg_sig_nodal_a)):
        raise ValueError('yg_sig_nodal_a contains nan values')
    elif any(np.isnan(yg_trac_nodal_b)):
        raise ValueError('yg_trac_nodal_b contains nan values')
    elif any(np.isnan(yg_sig_nodal_b)):
        raise ValueError('yg_sig_nodal_b contains nan values')

    # Calculate numbers of nodes ready to use in indexing
    n_nodes_a = len(yg_trac_nodal_a) + len(yg_sig_nodal_a)  # Number of nodes in track a
    n_nodes_trac_a = len(yg_trac_nodal_a)  # Number of nodes in the traction rail of track a
    n_nodes_sig_a = len(yg_sig_nodal_a)  # Number of nodes in the signal rail
    # Repeat for opposite track
    n_nodes_b = len(yg_trac_nodal_b) + len(yg_sig_nodal_b)
    n_nodes_trac_b = len(yg_trac_nodal_b)
    n_nodes_sig_b = len(yg_sig_nodal_b)
    # Calculate number of nodes in the whole network
    n_nodes = n_nodes_a + n_nodes_b  # Total nodes in network

    # Set up node indices
    node_locs_trac_a = np.arange(0, n_nodes_trac_a, 1).astype(int)  # Traction rail nodes
    node_locs_sig_a = np.arange(n_nodes_trac_a, n_nodes_a, 1).astype(int)  # Signal rail nodes
    node_locs_cb_a = node_locs_trac_a[np.where(np.isin(trac_sub_block_sum_a, pos_cb))[0]]  # Cross bond nodes
    node_locs_axle_trac_a = node_locs_trac_a[np.where(np.isin(trac_sub_block_sum_a, axle_pos_a))[0]]  # Axle traction rail nodes
    node_locs_axle_sig_a = node_locs_sig_a[np.where(np.isin(sig_sub_block_sum_a, axle_pos_a))[0]]  # Axle signal rail nodes
    block_nodes_trac_a = node_locs_trac_a[np.isin(trac_sub_block_sum_a, np.concatenate(([0.0], blocks_sum)))]  # All nodes at block ends
    node_locs_power_trac_a = block_nodes_trac_a[1:]  # Traction rail power supply nodes. Traction rail middle nodes share a connection to both a power supply and a relay, so use [1:]
    node_locs_relay_trac_a = block_nodes_trac_a[:-1]  # Traction rail relay nodes. Traction rail middle nodes share a connection to both a power supply and a relay, so use [1:]
    block_nodes_sig_a = node_locs_sig_a[np.isin(sig_sub_block_sum_a, np.concatenate(([0.0], blocks_sum)))]  # All nodes at block ends
    node_locs_power_sig_a = block_nodes_sig_a[1::2]  # Signal rail power supply nodes. Signal rails are separated by IRJs, so use [1::2]
    node_locs_relay_sig_a = block_nodes_sig_a[::2]  # Signal rail relay nodes. Signal rails are separated by IRJs, so use [1::2]
    # Repeat for opposite track
    # Note: Relays are position on the side of the block the trains enter from, so the ordering is reversed for the opposite track
    node_locs_trac_b = np.arange(n_nodes_a, n_nodes_a + n_nodes_trac_b, 1).astype(int)
    node_locs_sig_b = np.arange(n_nodes_a + n_nodes_trac_b, n_nodes).astype(int)
    node_locs_cb_b = node_locs_trac_b[np.where(np.isin(trac_sub_block_sum_b, pos_cb))[0]]
    node_locs_axle_trac_b = node_locs_trac_b[np.where(np.isin(trac_sub_block_sum_b, axle_pos_b))[0]]
    node_locs_axle_sig_b = node_locs_sig_b[np.where(np.isin(sig_sub_block_sum_b, axle_pos_b))[0]]
    block_nodes_trac_b = node_locs_trac_b[np.isin(trac_sub_block_sum_b, np.concatenate(([0.0], blocks_sum)))]
    node_locs_power_trac_b = block_nodes_trac_b[:-1]
    node_locs_relay_trac_b = block_nodes_trac_b[1:]
    block_nodes_sig_b = node_locs_sig_b[np.isin(sig_sub_block_sum_b, np.concatenate(([0.0], blocks_sum)))]
    node_locs_power_sig_b = block_nodes_sig_b[::2]
    node_locs_relay_sig_b = block_nodes_sig_b[1::2]

    # NODAL ADMITTANCE MATRIX
    Y = np.zeros((n_nodes, n_nodes))  # Set default nodal admittance matrix to size, filled with zeroes
    # Use stamps to add to different indices of the nodal admittance matrix
    stamp_series_admittance(Y, node_locs_trac_a[:-1], node_locs_trac_a[1:], ye_trac_a)  # Traction rail series admittances
    stamp_series_admittance(Y, node_locs_sig_a[:-1], node_locs_sig_a[1:], ye_sig_a)  # Signal rail series admittances
    stamp_series_admittance(Y, node_locs_relay_trac_a, node_locs_relay_sig_a, y_relay)  # Relay series admittances
    stamp_series_admittance(Y, node_locs_power_trac_a, node_locs_power_sig_a, y_power)  # Power supply series admittances
    stamp_series_admittance(Y, node_locs_axle_trac_a, node_locs_axle_sig_a, y_relay)  # Axle series admittances
    stamp_parallel_admittance(Y, node_locs_trac_a, yg_trac_nodal_a)  # Traction rail parallel admittances
    stamp_parallel_admittance(Y, node_locs_sig_a, yg_sig_nodal_a)  # Signal rail parallel admittances
    # Repeat for opposite track
    stamp_series_admittance(Y, node_locs_trac_b[:-1], node_locs_trac_b[1:], ye_trac_b)
    stamp_series_admittance(Y, node_locs_sig_b[:-1], node_locs_sig_b[1:], ye_sig_b)
    stamp_series_admittance(Y, node_locs_relay_trac_b, node_locs_relay_sig_b, y_relay)
    stamp_series_admittance(Y, node_locs_power_trac_b, node_locs_power_sig_b, y_power)
    stamp_series_admittance(Y, node_locs_axle_trac_b, node_locs_axle_sig_b, y_relay)
    stamp_parallel_admittance(Y, node_locs_trac_b, yg_trac_nodal_b)
    stamp_parallel_admittance(Y, node_locs_sig_b, yg_sig_nodal_b)
    # Note: Cross bonds only need to be added once as they connect both tracks
    stamp_series_admittance(Y, node_locs_cb_a, node_locs_cb_b, y_cb)  # Cross bond admittances

    # ELECTRIC FIELD
    # Calculate electric field parallel to blocks
    bearings = bearings[:, np.newaxis]
    e_parallel = ex_blocks * np.cos(bearings) + ey_blocks * np.sin(bearings)
    # Restructure parallel electric field array to fit the sub blocks structure
    block_idx = np.searchsorted(blocks_sum, np.cumsum(trac_sub_blocks_a))  # Indices of the blocks that the traction rail sub blocks are in
    e_parallel_trac_sub_block_a = e_parallel[block_idx, :]  # Maps block values onto traction rail sub blocks for parallel electric field
    block_idx = np.searchsorted(blocks_sum, np.nancumsum(sig_sub_blocks_a))
    e_parallel_sig_sub_block_a = e_parallel[block_idx, :]  # Maps block values onto signal rail sub blocks for parallel electric field
    e_parallel_sig_sub_block_a[np.isnan(sig_sub_blocks_a)] = 0  # Set electric field at IRJs to zero as it is useful for later calculations
    # Repeat for opposite track
    block_idx = np.searchsorted(blocks_sum, np.cumsum(trac_sub_blocks_b))
    e_parallel_trac_sub_block_b = e_parallel[block_idx, :]
    block_idx = np.searchsorted(blocks_sum, np.nancumsum(sig_sub_blocks_b))
    e_parallel_sig_sub_block_b = e_parallel[block_idx, :]  # Maps block values onto signal rail sub blocks for parallel electric field
    e_parallel_sig_sub_block_b[np.isnan(sig_sub_blocks_b)] = 0  # Set electric field at IRJs to zero as it is useful for later calculations

    # CURRENTS
    # Calculate current sources due to electric field
    i_trac_sub_block_a = e_parallel_trac_sub_block_a / z_trac  # Current sources along traction rail
    i_sig_sub_block_a = e_parallel_sig_sub_block_a / z_sig  # Current sources along signal rail
    # Repeat for opposite track
    i_trac_sub_block_b = e_parallel_trac_sub_block_b / z_trac
    i_sig_sub_block_b = e_parallel_sig_sub_block_b / z_sig

    # Set up current matrix
    J = np.zeros([n_nodes, len(ex_blocks[0, :])])  # Set default current matrix to size, filled with zeroes

    # Use stamps to add to different indices of the current matrix
    stamp_current_source(J, node_locs_trac_a[:-1], node_locs_trac_a[1:], i_trac_sub_block_a)  # Fill current sources along traction rail
    stamp_current_source(J, node_locs_sig_a[:-1], node_locs_sig_a[1:], i_sig_sub_block_a)  # Fill current sources along signal rail
    # Note: For electrical staggering the polarity of power supply current sources every other block must be reversed
    if electrical_staggering:
        stamp_current_source(J, node_locs_power_trac_a[0::2], node_locs_power_sig_a[0::2], i_power)  # Fill half of power supply current sources
        stamp_current_source(J, node_locs_power_trac_a[1::2], node_locs_power_sig_a[1::2], -i_power)  # Fill other half of power supply current sources
    else:
        stamp_current_source(J, node_locs_power_trac_a, node_locs_power_sig_a, i_power)
    # Repeat for opposite track
    stamp_current_source(J, node_locs_trac_b[:-1], node_locs_trac_b[1:], i_trac_sub_block_b)
    stamp_current_source(J, node_locs_sig_b[:-1], node_locs_sig_b[1:], i_sig_sub_block_b)
    if electrical_staggering:
        stamp_current_source(J, node_locs_power_trac_b[0::2], node_locs_power_sig_b[0::2], i_power)
        stamp_current_source(J, node_locs_power_trac_b[1::2], node_locs_power_sig_b[1::2], -i_power)
    else:
        stamp_current_source(J, node_locs_power_trac_b, node_locs_power_sig_b, i_power)

    # Sparse matrix
    y_csc = csr_matrix(Y).tocsc()
    factor = cholesky(y_csc)
    V = factor(J)

    # Calculate relay voltages and currents
    # 'a' first
    v_relay_top_node_a = V[node_locs_relay_sig_a]
    v_relay_bottom_node_a = V[node_locs_relay_trac_a]
    v_relay_a = v_relay_top_node_a - v_relay_bottom_node_a

    # 'b' first
    v_relay_top_node_b = V[node_locs_relay_sig_b]
    v_relay_bottom_node_b = V[node_locs_relay_trac_b]
    v_relay_b = v_relay_top_node_b - v_relay_bottom_node_b

    i_relays_a = v_relay_a * y_relay
    i_relays_b = v_relay_b * y_relay

    return {
        'i_relays_a': i_relays_a,
        'i_relays_b': i_relays_b,
        'V': V,
        'node_locs_relay_trac_a': node_locs_relay_trac_a,
        'node_locs_relay_trac_b': node_locs_relay_trac_b,
        'node_locs_relay_sig_a': node_locs_relay_sig_a,
        'node_locs_relay_sig_b': node_locs_relay_sig_b,
    }


# Stamp functions
def stamp_series_admittance(Matrix, i, j, admittance):
    Matrix[i, i] += admittance
    Matrix[j, j] += admittance
    Matrix[i, j] -= admittance
    Matrix[j, i] -= admittance


def stamp_parallel_admittance(Matrix, i, admittance):
    Matrix[i, i] += admittance


def stamp_current_source(Matrix, i, j, current_source):
    Matrix[i] -= current_source
    Matrix[j] += current_source


def test_model():
    bearings = np.deg2rad(np.arange(0, 360, 5))
    e_values = np.linspace(0, 20, 2001)
    ex_all = np.full((len(bearings), len(e_values)), np.nan)
    ey_all = np.full((len(bearings), len(e_values)), np.nan)
    for i in range(0, len(bearings)):
        ex_all[i] = e_values * np.cos(bearings[i])
        ey_all[i] = e_values * np.sin(bearings[i])
    #outputs = model_refactor(section_name='glasgow_edinburgh_falkirk', ex_blocks=np.array([[0, 1], [2, 3], [4, 5]]), ey_blocks=np.array([[0, 1], [2, 3], [4, 5]]), axle_pos_a=np.array([2.1]), axle_pos_b=([0.8]), y_trac=[1.6, 1.5, 1.4], y_sig=[0.1, 0.15, 0.075])
    outputs = model_refactor(section_name='east_coast_main_line', ex_uniform=np.array([0]), ey_uniform=np.array([0]), axle_pos_a=np.array([]), axle_pos_b=([]), y_trac=1.6, electrical_staggering=False)
    outputs2 = model_refactor(section_name='east_coast_main_line', ex_uniform=np.array([0]), ey_uniform=np.array([0]), axle_pos_a=np.array([]), axle_pos_b=([]), electrical_staggering=False)
    i = outputs['i_relays_a']
    i2 = outputs2['i_relays_a']
    plt.plot(i, 'x')
    plt.plot(i2, '.')
    plt.axhline(0.055, color='red')
    plt.axhline(-0.055, color='red')
    plt.show()


test_model()
