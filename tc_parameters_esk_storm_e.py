from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator, FuncFormatter

from models import model

plt.rcParams['font.size'] = '12'
plt.rcParams['mathtext.default'] = 'regular'
fig = plt.figure(figsize=(12, 12))
gs = GridSpec(4, 1, hspace=0)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])
ax3 = fig.add_subplot(gs[3])
axes = [ax0, ax1, ax2, ax3]

df_base = pd.read_csv(f'data/storm_b_fields/baseline-mar1989-esk.csv', skiprows=6)
data_array_base = df_base.to_numpy()
base_x = np.average(data_array_base[:, 0])
base_y = np.average(data_array_base[:, 1])

df = pd.read_csv(f'data/storm_b_fields/mar1989-esk.csv', skiprows=6)
data_array = df.to_numpy()
bx = data_array[:, 0] - base_x
by = data_array[:, 1] - base_y

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

ax0.plot(bx[1280:1340], color='cornflowerblue', label=r'$B_{x}$ (north-south)', zorder=5)

ax1.plot(ey[75, 1280:1340].T/100, color='goldenrod', label=r'$10 \times E_{y}$ (east-west)', zorder=5)

ax2.plot(t1[:25], ia1[:25], color='lightgrey', linestyle='-', label='No misoperation', zorder=5)
ax2.plot(t1[24:30], ia1[24:30], color='limegreen', linestyle='-', label='Wrong-side failure', zorder=5)
ax2.plot(t1[29:32], ia1[29:32], color='lightgrey', linestyle='-', zorder=5)
ax2.plot(t1[31:43], ia1[31:43], color='limegreen', linestyle='-', zorder=5)
ax2.plot(t1[42:48], ia1[42:48], color='lightgrey', linestyle='-', zorder=5)
ax2.plot(t1[47:52], ia1[47:52], color='limegreen', linestyle='-', zorder=5)
ax2.plot(t1[51:56], ia1[51:56], color='lightgrey', linestyle='-', zorder=5)
ax2.plot(t1[55:59], ia1[55:59], color='limegreen', linestyle='-', zorder=5)
ax2.plot(t1[58:62], ia1[58:62], color='lightgrey', linestyle='-', zorder=5)
ax2.plot(t1[61:66], ia1[61:66], color='limegreen', linestyle='-', zorder=5)
ax2.plot(t1[65:69], ia1[65:69], color='lightgrey', linestyle='-', zorder=5)
ax2.plot(t1[68:71], ia1[68:71], color='limegreen', linestyle='-', zorder=5)
ax2.plot(t1[70:], ia1[70:], color='lightgrey', linestyle='-', zorder=5)

ax3.plot(t2[:25], ia2[:25], color='lightgrey', linestyle='-', label='No misoperation', zorder=5)
ax3.plot(t2[24:30], ia2[24:30], color='red', linestyle='-', label='Right-side failure', zorder=5)
ax3.plot(t2[29:52], ia2[29:52], color='lightgrey', linestyle='-', zorder=5)
ax3.plot(t2[51:54], ia2[51:54], color='red', linestyle='-', zorder=5)
ax3.plot(t2[53:], ia2[53:], color='lightgrey', linestyle='-', zorder=5)

style2 = '--'
ax2.axhline(0.055, color='tomato', linestyle=style2, label='Drop-out current', zorder=4)
ax2.axhline(-0.055, color='tomato', linestyle=style2, zorder=4)
ax2.axhline(0.081, color='green', linestyle=style2, label='Pick-up current', zorder=4)
ax2.axhline(-0.081, color='green', linestyle=style2, zorder=4)
ax3.axhline(0.055, color='tomato', linestyle=style2, label='Drop-out current', zorder=4)
ax3.axhline(-0.055, color='tomato', linestyle=style2, zorder=4)
ax3.axhline(0.081, color='green', linestyle=style2, label='Pick-up current', zorder=4)
ax3.axhline(-0.081, color='green', linestyle=style2, zorder=4)

ax0.set_xlim(0, 59)
ax1.set_xlim(0, 59)
ax2.set_xlim(0, 59)
ax3.set_xlim(0, 59)

ax0.set_ylim(-3000, 0)
ax1.set_ylim(-6, 12)
ax2.set_ylim(-0.45, 0.45)
ax3.set_ylim(-0.45, 0.45)

ax0.set_yticks([-2500, -2000, -1500, -1000, -500, 0])
ax2.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
ax3.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])

ax0.tick_params(labelbottom=False)
ax1.tick_params(labelbottom=False)
ax2.tick_params(labelbottom=False)
ax0.sharex(ax3)
ax1.sharex(ax3)
ax2.sharex(ax3)

start = pd.to_datetime("13/03/1989 21:21", format="%d/%m/%Y %H:%M")
ax3.xaxis.set_major_locator(MultipleLocator(10))        # major tick every 10 min
ax3.xaxis.set_minor_locator(MultipleLocator(1))         # minor tick every 2 min
ax3.xaxis.set_major_formatter(FuncFormatter(lambda m, pos: (start + pd.Timedelta(minutes=m)).strftime("%H:%M")))

ax3.set_xlabel("Time on 13 March 1989")

ax0.set_ylabel('Magnetic field\nstrength\n(nT)')
ax1.set_ylabel('Electric field\nstrength\n(V/km)')
ax2.set_ylabel('Occupied\nrelay current\n(A)')
ax3.set_ylabel('Unoccupied\nrelay current\n(A)')

ax0.legend(loc='lower left')
ax1.legend(loc='upper left')
ax2.legend(loc='lower center', ncols=4)
ax3.legend(loc='lower center', ncols=4)

fig.align_ylabels(axes)

plt.savefig('plots/parameters_paper/BEI.pdf')
plt.show()
