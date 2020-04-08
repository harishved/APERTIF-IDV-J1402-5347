import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import emcee
import corner

def lnprior(theta):
    d_pc, vs_perp, theta_R = theta
    if 0.1 < d_pc < 0.7 and 15 < vs_perp < 35.0 and 0.5 < theta_R < 1.5:
        return 0.0
    return -np.inf
#
# Cost function for minimization
#
def lnprob(theta,phase,om,om_err):
   prior = lnprior(theta)
   if prior<-1e6:
      return prior
   # theta ihe parameter vector
   # theta[0] is the screen distance in parsec
   # vs_perp is the screen velocity perp to the major axis of scintels
   # theta_R is the PA of the scintels in the plane of the sky (measures from RA towards DEC)
   #
   d_pc,vs_perp,theta_R, = theta
   v_earth_kms = 29.78 # Earth orbital speed
   theta_sc = 0.36/2 # Scatter braodenines source size  # From NE2001 model for this sightline (hardcoded)
   D = d_pc*3e18 # Dstance to the screen
   rf = (D*20./2/3.14)**0.5	# Fresnel length (hardcoded wavelength)
   theta_f = rf/D	# Fresnel angle
   if theta_f<theta_sc*1e-3/3600*np.pi/180.0: # If fresnel angle smaller than source size
      s0 = theta_sc*1e-3/3600*np.pi/180.0*D   # Scintel length-scale is set by the source size
   else:				# Else it is set by the Fresnel length
      s0 = rf
   #
#
   ra_vec = np.array([0.5102431533, -0.7890506980, 0.3421270532])	# RA vector in ecliptic co-ordinates (hardcoded for this sightline)
   dec_vec = np.array([0.6938392383, 0.6127266876, 0.3783558083])	# DEC vector in ecliptic co-ordinates (hardcoded for this sightline)
   los = np.array([-0.5081722951, 0.0443277195, 0.8601139295])		# LOS vector in ecliptic co-ordinates (hardcoded for this sightline)
   ve_x = v_earth_kms*np.sin(phase)					# x and y ecliptic components of Earths velocity
   ve_y = -v_earth_kms*np.cos(phase)					# 
   ve_ra = ve_x*ra_vec[0] + ve_y*ra_vec[1]				# RA and DEC componenets of Earths velocity
   ve_dec = ve_x*dec_vec[0] + ve_y*dec_vec[1]				#
   ve_perp = -ve_ra*np.sin(theta_R) + ve_dec*np.cos(theta_R)		# Earth's velocity in sky-plane perp to scintel major axis 
   #
   vperp = np.absolute(vs_perp -ve_perp) 				# Relative velocity perpendicular to the scinte long axis
   #  
   model_om = vperp*1e5*60/s0	# Model predicted sicntillation rate in min^-1
   ln_prob = -0.5*np.sum((model_om-om)**2 / om_err**2) # Least squared error normalised by variance ( = chi-squared)
   return ln_prob+lnprior(theta)
#
#
#
#
tp = np.load("rates.npz")
phase = tp['phase']
om = tp['rate']
om_err =tp['rate_err'] 
#

ndim = 3
nwalkers = 200
nsteps = 5000
#
pos = [[np.random.rand()*0.1+0.5, np.random.rand()*5+20, np.random.rand()*0.4+1]  for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(phase,om,om_err),threads=30)
sampler.run_mcmc(pos,nsteps)
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
np.savez("samples_med.npz",d=samples)
#fig = corner.corner(samples, labels=["Screen dist.", "RA speed [km/s]", "DEC speed [km/s]"])
fig = corner.corner(samples,labels=[r"$d\,[{\rm pc}]$", r"$v_{\perp,\,{\rm R}}\,{\rm [km/s]}$", r"$\theta_R\, {\rm [rad]}$"],quantiles=[0.16,0.5,0.84],show_titles=True,title_kwargs={"fontsize": 12})
fig.savefig("triangle.png")

