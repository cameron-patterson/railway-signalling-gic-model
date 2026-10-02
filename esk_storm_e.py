import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from models import model

plt.rcParams['font.size'] = '12'
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(4, 1)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])
ax3 = fig.add_subplot(gs[3])

e_data = np.load(f'data/storm_e_fields/Mar1989/glasgow_edinburgh_falkirk_Mar1989_e_blocks_v1.npz')
ex = e_data['ex_blocks']
ey = e_data['ey_blocks']

axle_data = np.load(f'data/axle_positions/glasgow_edinburgh_falkirk_train_end_axles_midpoint_a.npy')
axles = axle_data[75]

i_data = model(section_name='glasgow_edinburgh_falkirk', axle_pos_a=axles, axle_pos_b=[], ex_blocks=ex/100, ey_blocks=ey/100)
ia = i_data['i_relays_a']
ia1 = ia[75, 1280:1340]
ia2 = ia[76, 1280:1340]
t1 = np.arange(0, 60, 1)
t2 = np.arange(0, 60, 1)

ia1 = np.append(ia1, (-0.081, -0.055, 0.081, 0.055, 0.081, 0.055, -0.081, -0.055, -0.081, -0.055, 0.081, 0.055))
t1 = np.append(t1, (23.34, 27.90, 28.72, 38.46, 42.72, 45.30, 48.24, 50.63, 52.28, 55.61, 57.82, 58.53))
order1 = np.argsort(t1)
ia1 = ia1[order1]
t1 = t1[order1]

ia2 = np.append(ia2, (0.055, 0.081, 0.055, 0.081))
t2 = np.append(t2, (23.78, 27.08, 48.99, 49.84))
order2 = np.argsort(t2)
ia2 = ia2[order2]
t2 = t2[order2]

ax1.plot(ey[75, 1280:1340].T/100)
ax2.plot(t1[:25], ia1[:25], color='lightgrey', linestyle='-', label='Occupied')
ax2.plot(t1[24:30], ia1[24:30], color='red', linestyle='-', label='Misoperation')
ax2.plot(t1[29:32], ia1[29:32], color='lightgrey', linestyle='-')
ax2.plot(t1[31:43], ia1[31:43], color='red', linestyle='-')
ax2.plot(t1[42:48], ia1[42:48], color='lightgrey', linestyle='-')
ax2.plot(t1[47:52], ia1[47:52], color='red', linestyle='-')
ax2.plot(t1[51:56], ia1[51:56], color='lightgrey', linestyle='-')
ax2.plot(t1[55:59], ia1[55:59], color='red', linestyle='-')
ax2.plot(t1[58:62], ia1[58:62], color='lightgrey', linestyle='-')
ax2.plot(t1[61:66], ia1[61:66], color='red', linestyle='-')
ax2.plot(t1[65:69], ia1[65:69], color='lightgrey', linestyle='-')
ax2.plot(t1[68:71], ia1[68:71], color='red', linestyle='-')
ax2.plot(t1[70:], ia1[70:], color='lightgrey', linestyle='-')

ax3.plot(t2[:25], ia2[:25], color='lightgrey', linestyle='-', label='Unoccupied')
ax3.plot(t2[24:30], ia2[24:30], color='red', linestyle='-', label='Misoperation')
ax3.plot(t2[29:52], ia2[29:52], color='lightgrey', linestyle='-')
ax3.plot(t2[51:54], ia2[51:54], color='red', linestyle='-')
ax3.plot(t2[53:], ia2[53:], color='lightgrey', linestyle='-')

ax2.axhline(0.055, color='tomato', linestyle='--')
ax2.axhline(-0.055, color='tomato', linestyle='--')
ax2.axhline(0.081, color='limegreen', linestyle='--')
ax2.axhline(-0.081, color='limegreen', linestyle='--')
ax3.axhline(0.055, color='tomato', linestyle='--')
ax3.axhline(-0.055, color='tomato', linestyle='--')
ax3.axhline(0.081, color='limegreen', linestyle='--')
ax3.axhline(-0.081, color='limegreen', linestyle='--')

ax2.set_ylabel('Relay Current (A)')
ax3.set_xlabel('Minutes')
ax3.set_ylabel('Relay Current (A)')
ax2.legend(loc='lower left')
ax3.legend(loc='lower left')
#ax2.set_ylim(-0.5, 0.5)
#ax3.set_ylim(-0.5, 0.5)
ax1.set_xlim(0, 59)
ax2.set_xlim(0, 59)
ax3.set_xlim(0, 59)

plt.show()
