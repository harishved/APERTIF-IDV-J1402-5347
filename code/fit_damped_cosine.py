# Fit a damped cosine to the Apr 09 datatset to find the analytic function that described the
# underlying ACF of scintillation
# Author: Harish Vedantham
# ASTRON (Apr 2020)
# USAGE: python fit_damped_cosine.py <data_file_name>
# Assume that the data file is an ascii file with two columns: time in seconds and flux in Jy
#
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import celerite
from celerite import terms
from scipy.optimize import minimize
import sys
import emcee
import corner
#
# 
#
# NUMERICAL ACF COMPUTATION
#
def compute_acf(ut,s): # Code to numerically compute the ACF
   # x = tome vector
   # y = flux density vector (data)
   noise = np.var(np.diff(s))/2 # Noise variance from time differences    
   s-=np.mean(s)    # Mean subtract flux values
   npts = len(s)    # Num of points in light curve
   nbins = npts     # Number of bins for auto-corr computation (take = npts for now)
   lag = np.arange(0.0,nbins,1.0)*(ut[1]-ut[0]) # Range of temporal lags for ac computation
   acf = np.zeros(len(lag))     # Initialize autocorrelation vector to all zeros
   wt = np.zeros(len(lag))      # weight vector: many data points land into given bin
   #
   # COMPUTE ACF VECTOR USING ALL AVAILABLE PAIRS
   #
   for i in range(npts):        # For each flux measurement
      for j in range(i,npts):   # For each pair of flux measurements
         acf[j-i]+=s[i]*s[j]    # Accumulate acf at right lag
         wt[j-i]+=1             # Add 1 to weight at right lag
   acf/=wt                      # Normalize acf by weights
   acf[0]-=noise                # Subtract noise bias
   acf/=acf[0]                  # Normalize acf vector to unity at zero lag   
   return lag,acf
#
#
# NEGATIVE LOG LIKELIHOOD (FUNCTION TO OPTIMIZE)
def neg_log_like(params, y, gp):
    # params = Model parameters
    # y = data vector
    # gp = Gaussian process object
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)
#
def log_like(params,y,gp):
   gp.set_parameter_vector(params)
   return gp.log_likelihood(y) + lnprior(params)
#
def lnprior(theta):
    log_c, log_d = theta   
    if -3.75-2.0 < log_c < -3.75+2.0 and -1.8-2.0 < log_d < -1.8+2.0: # THESE VALUES ARE HARDCODED
        return 0.0
    return -np.inf
#
# read in the data, mean subtract and compute variance
tp = np.loadtxt(sys.argv[1])
x = tp[:,0]/60.0 # In minutes
x-=x[0] 
y = tp[:,1]*1e3 # in mJy
print ("mean = %.3f"%np.mean(y))
y-=np.mean(y)
#y-=6.06
yerr = np.std(np.diff(y))/2**0.5
print ("yerr = %f"%yerr)
K0 = (np.mean(y**2)-yerr**2)
print ("K0 = %f"%K0)
#
#
log_a = np.log(K0)
log_c = np.log(0.1)	# Initial value --> 10 min timescale
log_d = np.log(0.1/5.0) # Initial value ---> 50 min timescale
#
# Select reasonable bounds on the parameter value
#
bounds = dict(log_a = (np.log(0.01*np.var(y)),np.log(100*np.var(y))),\
              log_c = (np.log(1e-10),np.log(100.0)),\
              log_d = (np.log(1e-10),np.log(100.0)))\
#
kernel = terms.ComplexTerm(log_a = np.log(K0), log_c = log_c, log_d = log_d, bounds = bounds)
kernel.freeze_parameter("log_a")
#
gp = celerite.GP(kernel, mean = 0.0) # mean = 0 because we have mean subtracted already
gp.compute(x, yerr)  # You always need to call compute once. (what the docs say!)
print("Initial log likelihood: {0}".format(gp.log_likelihood(y)))
#
# Use scipy minimize
initial_params = gp.get_parameter_vector()
bounds = gp.get_parameter_bounds()
#
#
r = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds, args=(y, gp))
gp.set_parameter_vector(r.x)
print ("Final log likelihood: {0}".format(gp.log_likelihood(y)))
#print ("Best fitting parameters: %.3e, %.3e, %.3e"%(np.exp(r.x[0]),np.exp(r.x[1]),np.exp(r.x[2])))
print ("Best fitting parameters: %.3e, %.3e"%(np.exp(r.x[0]),np.exp(r.x[1])))
#
# COMPUTE THE 1/E TIMESCALE
tvec = np.arange(0.0,300.0,300.0/500.0)
ker = K0* np.exp(-tvec*np.exp(r.x[0]))*np.cos(tvec*np.exp(r.x[1]))
ker/=ker[0]
#Now find the first 1/e crossing (there may be more at larger lags!)
I_cross = np.min(np.where(ker<1/np.exp(1.)))
tau =  np.polyval( np.polyfit( \
       ker[I_cross-2:I_cross+2],tvec[I_cross-2:I_cross+2],1),\
       [1/np.exp(1)])   #Compute 1/e timescale using linear interpolation 
print ("1/e timescale = %.3e mins"%(tau))
#
# Plot fitted kernal
plt.figure(figsize=(10/1.2,5/1.2))
plt.plot(tvec,ker,label="Fitted damped cosine")
lag,acf = compute_acf(x,y)
plt.step(lag,acf,label="Numericql ACF")
plt.legend(frameon=False)
plt.xlabel("Time offset [minutes]")
plt.ylabel("Auto-correlation function [normalize]")
plt.xlim([0,50])
plt.ylim([-1,1])
plt.plot([tau,tau],[-1,1],'k--')
plt.tight_layout()
plt.savefig("compare_kernel.pdf")
plt.close()
#
#
# PERFORM MCMC SIMULATION TO GET FORMAL ERROR ON THE FIT PARAMETERS
#
#
ndim = 2
nwalkers = 200
nsteps = 5000
#
pos = [[np.random.rand()*0.2 + r.x[0], np.random.rand()*0.1 + r.x[1]]  for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers,ndim,log_like,args=(y,gp),threads=16)
sampler.run_mcmc(pos,nsteps)
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
#np.savez("samples_med.npz",d=samples)
fig = corner.corner(samples,labels=[r"${\rm log}(c)$", r"${\rm log}(d)$"],quantiles=[0.16,0.5,0.84],show_titles=True,title_kwargs={"fontsize": 12})
fig.savefig("triangle.png")
