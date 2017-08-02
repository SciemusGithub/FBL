## Generate the dataset, save the data and a plot of it as well

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from skfuzzy import control as ctrl
import scipy.io

# Set for reproducibility
np.random.seed(seed=123)

loc_risk = ctrl.Antecedent(np.arange(0, 11, 1), 'loc_risk')
maintenance = ctrl.Antecedent(np.arange(0, 11, 1), 'maintenance')
downtime = ctrl.Consequent(np.arange(0, 101, 1), 'downtime')

# Membership functions with parameters pre-specified. 
# Data would be generated from this fuzzy system
# MCMC would be used later to identify whether the parameters of the fuzzy system can be recovered from the data.  
loc_risk['low'] = fuzz.trimf(loc_risk.universe, [0, 0, 5])
loc_risk['medium'] = fuzz.trimf(loc_risk.universe, [0, 5, 10])
loc_risk['high'] = fuzz.trimf(loc_risk.universe, [5, 10, 10])

maintenance['poor'] = fuzz.trimf(maintenance.universe, [0, 0, 5])
maintenance['average'] = fuzz.trimf(maintenance.universe, [0, 5, 10])
maintenance['good'] = fuzz.trimf(maintenance.universe, [5, 10, 10])

downtime['low'] = fuzz.trimf(downtime.universe, [0, 0, 50])
downtime['medium'] = fuzz.trimf(downtime.universe, [0, 50, 100])
downtime['high'] = fuzz.trimf(downtime.universe, [50, 100, 100])

# Rule objects connect one or more antecedent membership functions with
# one or more consequent membership functions, using 'or' or 'and' to combine the antecedents.
rule1 = ctrl.Rule(loc_risk['high'] | maintenance['poor'], downtime['high'])
rule2 = ctrl.Rule(maintenance['average'] | loc_risk['medium'], downtime['medium'])
rule3 = ctrl.Rule(maintenance['good'] & loc_risk['low'], downtime['low'])

outputs=[[],[],[]]

loc_inp=np.random.random(500)
maint_inp=np.random.random(500)

for x,y in zip(loc_inp,maint_inp):
		tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
		tipping = ctrl.ControlSystemSimulation(tipping_ctrl)
		tipping.input['loc_risk'] = 10.0*x
		tipping.input['maintenance'] = 10.0*y
		tipping.compute()
		outputs[0].append(x)
		outputs[1].append(y)
		outputs[2].append(tipping.output['downtime'])


## Plot the outputs on a graph for visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=27, azim=139)
ax.plot_trisurf(outputs[0],outputs[1],outputs[2], cmap=cm.jet, linewidth=0.2)
ax.set_xlabel('Location risk')
ax.set_ylabel('Maintenance')
ax.set_zlabel('System downtime')
plt.savefig('data_plot.png')

## Save the data
scipy.io.savemat('data_gen.mat',{'outputs': outputs})

