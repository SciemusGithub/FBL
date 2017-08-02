## MCMC parameter estimates for the parameters of the fuzzy model
## TODO: rename the params qual_obs etc to more meaningful ones

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from pymc3 import *
import scipy.io
import skfuzzy as fuzz
import pandas as pd
from skfuzzy import control as ctrl
import random

import theano.tensor as T 
from theano.compile.ops import as_op

np.random.seed(123) ## Sets seed for numpy only, not the python random gen, otherwise folder names are same
## on multiple runs

## Load the .mat file containing the data
mat = scipy.io.loadmat('data_gen.mat')
## The observed data
qual_obs=10*mat['outputs'][0]
serv_obs=10*mat['outputs'][1]
tip_obs=mat['outputs'][2]
## Use only 15 data points to estimate the parameters
qual_obs=qual_obs[0:15]
serv_obs=serv_obs[0:15]
tip_obs=tip_obs[0:15]

maxsamples=5000
mcmc_list=[]

## Keeping an archive of previously sampled values to speed up simulations
archive=[[],[]] # first list would contain the input vector and the second would contain the output

@as_op(itypes=[T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar], otypes=[T.dvector])  ## have to replace the dscalar's with dvector
def fuzzy_func(value1,value2,value3,value4,value5,value6,value7,value8,value9):
	#print value1,value2,value3,value4,value5,value6,value7,value8,value9
	if [value1,value2,value3,value4,value5,value6,value7,value8,value9] in archive[0]:
		arch_ind=archive[0].index([value1,value2,value3,value4,value5,value6,value7,value8,value9])
		# print 'taking from archive index: ', arch_ind
		return np.array(archive[1][archive[0].index([value1,value2,value3,value4,value5,value6,value7,value8,value9])])
	else:
		# print 'inside fuzzy func'
		## Define the whole fuzzy inference within the evaluation function
		loc_risk = ctrl.Antecedent(np.arange(0, 11, 1), 'loc_risk')
		maintenance = ctrl.Antecedent(np.arange(0, 11, 1), 'maintenance')
		downtime = ctrl.Consequent(np.arange(0, 101, 1), 'downtime')

		loc_risk['low'] = fuzz.trimf(loc_risk.universe, [0, 0, float(value1)])
		loc_risk['medium'] = fuzz.trimf(loc_risk.universe, [0, float(value2), 10])
		loc_risk['high'] = fuzz.trimf(loc_risk.universe, [float(value3), 10, 10])

		maintenance['poor'] = fuzz.trimf(maintenance.universe, [0, 0, float(value4)])
		maintenance['average'] = fuzz.trimf(maintenance.universe, [0, float(value5), 10])
		maintenance['good'] = fuzz.trimf(maintenance.universe, [float(value6), 10, 10])

		## We want to estimate the membership function of just this one, both value=value1=13
		# Custom membership functions can be built interactively with a familiar, Pythonic API
		
		downtime['low'] = fuzz.trimf(downtime.universe, [0, 0, float(value7)])
		downtime['medium'] = fuzz.trimf(downtime.universe, [0, float(value8), 100])
		downtime['high'] = fuzz.trimf(downtime.universe, [float(value9), 100, 100])


		rule1 = ctrl.Rule(loc_risk['high'] | maintenance['poor'], downtime['high'])
		rule2 = ctrl.Rule(maintenance['average'] | loc_risk['medium'], downtime['medium'])
		rule3 = ctrl.Rule(maintenance['good'] & loc_risk['low'], downtime['low'])

		outputs=[]

		rule=[rule1, rule2, rule3]
		tipping_ctrl = ctrl.ControlSystem(rule)
		tipping = ctrl.ControlSystemSimulation(tipping_ctrl,flush_after_run=len(qual_obs)+1)

		for x,y in zip(qual_obs, serv_obs):
			tipping.input['loc_risk'] = x
			tipping.input['maintenance'] = y
			tipping.compute()
			outputs.append(tipping.output['downtime'])

		## append to archive
		archive[0].append([value1,value2,value3,value4,value5,value6,value7,value8,value9])
		archive[1].append(outputs)
		return np.array(outputs)


with Model() as model_ip:
	## all original vals are 5 or 50
	## location risk params 
    par1 = Uniform('par1',0,10.0) # 
    par2 = Uniform('par2',0,10.0) #
    par3 = Uniform('par3',0,10.0) #
	## maintenance params 
    par4 = Uniform('par4',0,10.0) # 
    par5 = Uniform('par5',0,10.0) #
    par6 = Uniform('par6',0,10.0) #
	## downtime params 
    par7 = Uniform('par7',0,100.0) # 
    par8 = Uniform('par8',0,100.0) #
    par9 = Uniform('par9',0,100.0) #

    b = fuzzy_func(par1, par2, par3, par4, par5, par6, par7, par8, par9)
    
    # Likelihood (sampling distribution) of observations
    Y_obs = Normal('Y_obs', mu=b, sd=1 , observed=tip_obs) 

with model_ip:
    step = Metropolis() 
    foldername='run_1'#+str(random.randint(0,10000))
    db = backends.Text(foldername)
    trace = sample(maxsamples, step, trace=db)

burnin=int(.2*maxsamples)
traceplot(trace[burnin:])
plt.savefig('posterior_dists_of_fuzzy_params.png')
summary(trace[burnin:])

