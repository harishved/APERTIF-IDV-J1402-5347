# PROGRAM TO FIND THE SCALE PARAMETER IN THE AUTOCOVARIANCE KERNEL
# THE ACF IS FULLY HARDCODED AND ONLY A SINGLE TIME-SCALE-PARAMETER IS FIT
# THE ACF IS FIXED VIA THE CO-EFFICIENTS IN THE FUNCTION GET_COMPLEX_COEFFS
# AUTHOR: HARISH VEDANTHAM
# ASTRON  (APRIL 2020)
#
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import celerite
from celerite import terms
from scipy.optimize import minimize
import sys
#
# USAGE: python fit_scale_parameter.py <data_file_name>
# Assume that the data file is an ascii file with two columns: time in seconds and flux in Jy
# The a,b,c coefficients are hardcoded based on 09-Apr-2019 dataset fits
#
# CUSTOM KERNEL THAT ONLY TAKES A SCALE PARAMETER AND VARIANCE
#
class CustomTerm(terms.Term):
    parameter_names = ("log_R",)
    #def get_real_coefficients(self, params):
    #    log_R, log_A = params
    #    return (
    #        np.exp(log_A), np.exp(1.568e-01+log_R)
    #    )
    # 2.219e-02, 1.567e-01
    def get_complex_coefficients(self, params):
        log_R = params
        return (
            3.477105, 0.0, 2.219e-02*np.exp(log_R), np.exp(log_R)*1.567e-01 # HARDCODED PARAMETERS
        )
# NEGATIVE LOG LIKELIHOOD (FUNCTION TO OPTIMIZE)
def neg_log_like(params, y, gp):
    # params = Model parameters
    # y = data vector
    # gp = Gaussian process object
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)
#
# read in the data, mean subtract and compute variance
tp = np.loadtxt(sys.argv[1])
x = tp[:,0]/60.0 # In minutes
x-=x[0] 
y = tp[:,1]*1e3 # in mJy
y-=np.mean(y)
yerr = np.std(np.diff(y))/2**0.5
#
log_R = 0.0	        # Initial value --> same as 09-Apr timescale
#
# Select reasonable bounds on the parameter value
#
bounds = dict(log_R = (np.log(1e-5),np.log(1e5)))
#
kernel = CustomTerm(log_R = log_R, bounds = bounds)
#
gp = celerite.GP(kernel, mean = 0.0) # mean = 0 because we have mean subtracted already
gp.compute(x, yerr)  # You always need to call compute once. (what the docs say!)
print("Initial log likelihood: {0}".format(gp.log_likelihood(y)))
#
# Use scipy minimize to find the best fit scale parameter
initial_params = gp.get_parameter_vector()
bounds = gp.get_parameter_bounds()
#
r = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds, args=(y, gp))
gp.set_parameter_vector(r.x)
print ("Final log likelihood: {0}".format(gp.log_likelihood(y)))
#
print ("Best fitting scale parameters: %.3e"%(np.exp(r.x[0])))
print ("Best fitting scint rate (1/min) =  %.3e"%(np.exp(r.x[0])/7.171)) # THE RATE FOR THE APRIL 09 DATASET IS HARCODED
#
# Brute force compute the likelihood function on a 1-d grid to find the formal errors on the fit
#
log_Rvec = np.arange(r.x[0]-0.2,r.x[0]+0.2,(3)/1000) # This is the search range 
#						     # You may have to vary this to make sure the 1sigma point is in it
ll = np.zeros(log_Rvec.shape) # Likelihood vector
for i in range(len(ll)):
   gp.set_parameter_vector([log_Rvec[i]])
   ll[i] = gp.log_likelihood(y)
ll-=np.amax(ll)
#
n = len(ll)
I = np.where(ll[:int(n/2)]+0.5<0)[0][-1]
low = np.polyval( np.polyfit(ll[I-2:I+2],log_Rvec[I-2:I+2],1),[-0.5])  
#
I = np.where(ll[int(n/2):]+0.5<0)[0][0]+int(n/2)
high = np.polyval( np.polyfit(ll[I-2:I+2],log_Rvec[I-2:I+2],1),[-0.5])   
print ("1-sigma bounds on timescale [min]: best, low, high")
#print (7.171*low[0],7.171*r.x[0],7.171*high[0])
print(7.171*np.exp(-r.x[0]), 7.171*(-np.exp(-high[0])+np.exp(-r.x[0])), 7.171*(-np.exp(-r.x[0])+np.exp(-low[0])))
print("1-sigma bounds on rate [1/min]: best, low, high")
print(np.exp(r.x[0])/7.171,(np.exp(high[0])-np.exp(r.x[0]))/7.171, (np.exp(r.x[0])-np.exp(low[0]))/7.171)
#
