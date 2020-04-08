import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
#
# Cost function for minimization
#
def cost_func(theta):
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
   tp = np.loadtxt("rates.txt",comments="#",delimiter=",") # File with annual modulation rate estimates
   # col1 = phase angle of the Earth in its orbit, col2 = rate in min^-1, col3 = err on rate in min^-1
   phase = tp[:,0]
   om = tp[:,1]
   om_err = ( (np.amax(tp[:,2:],axis=1))**2 + om**2/100)**0.5
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
   lse = np.sum((model_om-om)**2 / om_err**2) # Least squared error normalised by variance ( = chi-squared)
   return lse
#

def model_fit_plot(theta):
   d_pc,vs_perp,theta_R, = theta
   v_earth_kms = 29.78 # Earth orbital speed
   theta_sc = 0.36/2 # Scatter braodenines source size  # From NE2001 model for this sightline (hardcoded)
   D = d_pc*3e18 # Dstance to the screen
   rf = (D*20./2/3.14)**0.5     # Fresnel length (hardcoded wavelength)
   theta_f = rf/D       # Fresnel angle
   if theta_f<theta_sc*1e-3/3600*np.pi/180.0: # If fresnel angle smaller than source size
      s0 = theta_sc*1e-3/3600*np.pi/180.0*D   # Scintel length-scale is set by the source size
   else:                                # Else it is set by the Fresnel length
      s0 = rf
   #
   tp = np.loadtxt("rates.txt",comments="#",delimiter=",") # File with annual modulation rate estimates
   # col1 = phase angle of the Earth in its orbit, col2 = rate in min^-1, col3 = err on rate in min^-1
   phase = tp[:,0]
   om = tp[:,1]
   om_err = ( (np.amax(tp[:,2:],axis=1))**2 + om**2/100)**0.5 
#
   phase1 = np.arange(0.0,2*np.pi+0.1,0.05)	# Phase vector for plot of best model
   ra_vec = np.array([0.5102431533, -0.7890506980, 0.3421270532])       # RA vector in ecliptic co-ordinates (hardcoded for this sightline)
   dec_vec = np.array([0.6938392383, 0.6127266876, 0.3783558083])       # DEC vector in ecliptic co-ordinates (hardcoded for this sightline)
   los = np.array([-0.5081722951, 0.0443277195, 0.8601139295])          # LOS vector in ecliptic co-ordinates (hardcoded for this sightline)
   ve_x = v_earth_kms*np.sin(phase1)                                     # x and y ecliptic components of Earths velocity
   ve_y = -v_earth_kms*np.cos(phase1)                                    # 
   ve_ra = ve_x*ra_vec[0] + ve_y*ra_vec[1]                              # RA and DEC componenets of Earths velocity
   ve_dec = ve_x*dec_vec[0] + ve_y*dec_vec[1]                           #
   ve_perp = -ve_ra*np.sin(theta_R) + ve_dec*np.cos(theta_R)            # Earth's velocity in sky-plane perp to scintel major axis 
   #
   vperp = np.absolute(vs_perp -ve_perp)                                # Relative velocity perpendicular to the scinte long axis
   #  
   model_om = vperp*1e5*60/s0   # Model predicted sicntillation rate in min^-1
   lse = cost_func(theta) 
   #
   #
   plt.figure(figsize=(3.5,3.5))
   plt.plot(phase1,model_om,'k')
   plt.errorbar(x=phase,y=om,yerr=om_err,fmt='ko',markersize=2.5)
   #plt.yscale("log") 
   #plt.ylim([1,1000])
   plt.xlabel("Orbital phase [rad]")
   plt.ylabel(r"Scintillation rate [min$^{-1}$]")
   plt.title(r"$d = %.2f\,{\rm pc},\,\, v_s^\perp = %.1f\,{\rm km/s},\,\,\theta_{\rm s} = %.1f\,{\rm rad} $"%(d_pc,vs_perp,theta_R),fontsize=12)
   plt.text(x=2.1,y=0.14,s=r"$\chi^2_{dof} = %.1f$"%(lse/((len(phase)-3))))
   plt.tight_layout()
   plt.savefig("aniso.pdf")
   plt.close()
#
##
#
dvec = np.arange(0.1,10.0,0.1)
vvec = np.arange(0,100.0,1.0)
thvec = np.arange(0.0,2*np.pi,0.1)
ll = np.zeros((len(dvec),len(vvec),len(thvec)))
for i in range(len(dvec)):
   for j in range(len(vvec)):
      for k in range(len(thvec)):
         ll[i,j,k] = cost_func([dvec[i],vvec[j],thvec[k]])

a,b,c = np.unravel_index(np.argmin(ll),ll.shape)
print (dvec[a],vvec[b],thvec[c])

