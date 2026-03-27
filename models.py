import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sksparse.cholmod import cholesky


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
    print(f'Number of blocks: {len(blocks)}')
    bearings = data['bearings']  # Block bearings (radians); Note: zero is directly northwards, with positive values increasing clockwise
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
    stamp_series_admittance(Y, node_locs_axle_trac_a, node_locs_axle_sig_a, y_axle)  # Axle series admittances
    stamp_parallel_admittance(Y, node_locs_trac_a, yg_trac_nodal_a)  # Traction rail parallel admittances
    stamp_parallel_admittance(Y, node_locs_sig_a, yg_sig_nodal_a)  # Signal rail parallel admittances
    # Repeat for opposite track
    stamp_series_admittance(Y, node_locs_trac_b[:-1], node_locs_trac_b[1:], ye_trac_b)
    stamp_series_admittance(Y, node_locs_sig_b[:-1], node_locs_sig_b[1:], ye_sig_b)
    stamp_series_admittance(Y, node_locs_relay_trac_b, node_locs_relay_sig_b, y_relay)
    stamp_series_admittance(Y, node_locs_power_trac_b, node_locs_power_sig_b, y_power)
    stamp_series_admittance(Y, node_locs_axle_trac_b, node_locs_axle_sig_b, y_axle)
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
        'J': J,
        'node_locs_relay_trac_a': node_locs_relay_trac_a,
        'node_locs_relay_trac_b': node_locs_relay_trac_b,
        'node_locs_relay_sig_a': node_locs_relay_sig_a,
        'node_locs_relay_sig_b': node_locs_relay_sig_b,
    }


def test_model(section_name, axle_pos_a=None, axle_pos_b=None, ex_blocks=None, ey_blocks=None, ex_uniform=None, ey_uniform=None, electrical_staggering=True, **kwargs):
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
    print(f'Number of blocks: {len(blocks)}')
    bearings = data['bearings']  # Block bearings (radians); Note: zero is directly northwards, with positive values increasing clockwise
    print("TEST MODEL")
    bearings = np.full(len(bearings), np.deg2rad(90))
    curve1_angles = np.deg2rad(np.linspace(90, 0, 16))
    curve2_angles = np.deg2rad(np.linspace(0, 90, 16))
    bearings[40:56] = curve1_angles
    bearings[56:60] = 0
    bearings[60:76] = curve2_angles
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
    stamp_series_admittance(Y, node_locs_axle_trac_a, node_locs_axle_sig_a, y_axle)  # Axle series admittances
    stamp_parallel_admittance(Y, node_locs_trac_a, yg_trac_nodal_a)  # Traction rail parallel admittances
    stamp_parallel_admittance(Y, node_locs_sig_a, yg_sig_nodal_a)  # Signal rail parallel admittances
    # Repeat for opposite track
    stamp_series_admittance(Y, node_locs_trac_b[:-1], node_locs_trac_b[1:], ye_trac_b)
    stamp_series_admittance(Y, node_locs_sig_b[:-1], node_locs_sig_b[1:], ye_sig_b)
    stamp_series_admittance(Y, node_locs_relay_trac_b, node_locs_relay_sig_b, y_relay)
    stamp_series_admittance(Y, node_locs_power_trac_b, node_locs_power_sig_b, y_power)
    stamp_series_admittance(Y, node_locs_axle_trac_b, node_locs_axle_sig_b, y_axle)
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
        'J': J,
        'node_locs_relay_trac_a': node_locs_relay_trac_a,
        'node_locs_relay_trac_b': node_locs_relay_trac_b,
        'node_locs_relay_sig_a': node_locs_relay_sig_a,
        'node_locs_relay_sig_b': node_locs_relay_sig_b,
    }


# Stamp functions
def stamp_series_admittance(matrix, i, j, admittance):
    matrix[i, i] += admittance
    matrix[j, j] += admittance
    matrix[i, j] -= admittance
    matrix[j, i] -= admittance


def stamp_parallel_admittance(matrix, i, admittance):
    matrix[i, i] += admittance


def stamp_current_source(matrix, i, j, current_source):
    matrix[i] -= current_source
    matrix[j] += current_source

