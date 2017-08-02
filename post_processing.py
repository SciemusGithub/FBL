## This will generate Krushke type graphs for checking the MCMC convergence. 
## Also this would plot how the final membership function looks like and the probabilistic output
## due to this. 
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
plt.style.use('bmh')
from skfuzzy import control as ctrl


def fuzzy_mem_func_plot(value1_list,value2_list,value3_list,value4_list,value5_list,value6_list,value7_list,value8_list,value9_list):
	## Plot for the two inputs and the output
	fig = plt.figure()
	plt.hold(True)
	x_tip  = np.arange(0, 11, .1)
	## plot for location risk membership funcs
	for value1,value2, value3 in zip(value1_list,value2_list,value3_list):
		tip_lo = fuzz.trimf(x_tip, [0, 0, float(value1)])
		tip_med = fuzz.trimf(x_tip, [0, float(value2), 10])
		tip_hi = fuzz.trimf(x_tip, [float(value3), 10, 10])

		plt.plot(x_tip, tip_lo, 'b', linewidth=.5, label='Low',alpha=0.1)
		plt.plot(x_tip, tip_med, 'g', linewidth=.5, label='Medium',alpha=0.1)
		plt.plot(x_tip, tip_hi, 'r', linewidth=.5, label='High',alpha=0.1)
	
	plt.title('Location risk Membership function')
	plt.savefig('location_mem_func_posterior.png')

	fig = plt.figure()
	plt.hold(True)
	x_tip  = np.arange(0, 11, .1)

	## plot for maintenance membership funcs
	for value4,value5, value6 in zip(value4_list,value5_list,value6_list):
		tip_lo = fuzz.trimf(x_tip, [0, 0, float(value4)])
		tip_med = fuzz.trimf(x_tip, [0, float(value5), 10])
		tip_hi = fuzz.trimf(x_tip, [float(value6), 10, 10])

		plt.plot(x_tip, tip_lo, 'b', linewidth=.5, label='Low',alpha=0.1)
		plt.plot(x_tip, tip_med, 'g', linewidth=.5, label='Medium',alpha=0.1)
		plt.plot(x_tip, tip_hi, 'r', linewidth=.5, label='High',alpha=0.1)
	
	plt.title('Maintenance Membership function')
	plt.savefig('maintenance_mem_func_posterior.png')


	fig = plt.figure()
	plt.hold(True)
	x_tip  = np.arange(0, 101, .1)

	## plot for downtime membership funcs
	for value7,value8, value9 in zip(value7_list,value8_list,value9_list):
		tip_lo = fuzz.trimf(x_tip, [0, 0, float(value7)])
		tip_med = fuzz.trimf(x_tip, [0, float(value8), 100])
		tip_hi = fuzz.trimf(x_tip, [float(value9), 100, 100])

		plt.plot(x_tip, tip_lo, 'b', linewidth=.5, label='Low',alpha=0.1)
		plt.plot(x_tip, tip_med, 'g', linewidth=.5, label='Medium',alpha=0.1)
		plt.plot(x_tip, tip_hi, 'r', linewidth=.5, label='High',alpha=0.1)
	
	plt.title('Downtime Membership function')
	plt.savefig('downtime_mem_func_posterior.png')





## Read all mcmc samples from csv
df=pd.read_csv('./run_1/chain-0.csv')
burnin=int(.2*df.shape[0])
## get few samples after burnin, for plotting the posterior of membership funcitons
df=df[burnin:burnin+300]

### Plot the super-imposition of the membership functions
fuzzy_mem_func_plot(df['par1'].as_matrix(),df['par2'].as_matrix(),df['par3'].as_matrix(),df['par4'].as_matrix(),df['par5'].as_matrix(),df['par6'].as_matrix(),df['par7'].as_matrix(),df['par8'].as_matrix(),df['par9'].as_matrix())





