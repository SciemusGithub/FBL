#### Setup python virtual environment and activate it
This assumes the python 2.7 Anaconda distribution is installed, pip package manager is available and you are on a linux machine

```
conda create -n fuzzy_bayes python=2.7
source activate fuzzy_bayes
```


#### Install the required libraries in the new environment

```
pip install -r requirements.txt
```


#### Run the following code to generate the data from a fuzzy inference system
This would also save a surface plot of the generated data, along with a .mat file for the data

```
python data_gen.py
```

#### Run MCMC code to estimate the parameters of the fuzzy system which generated the data
This would take a few minutes to finish. 

```
python param_estimate.py
```
It would save the MCMC samples as a csv in a folder called 'run_1'
This is just a quick demo with 5000 iterations of the MCMC chain, which is not enough for convergence.
Change the maxsamples to 10,000 for a converged solution. The time taken would be a lot more. 

#### Plot the posterior distribution of the membership functions and visualise

```
python post_processing.py
```









