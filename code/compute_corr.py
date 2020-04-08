import numpy as np
import matplotlib.pyplot as plt
# Compute the autocorrelation function and find the 1/e timescale
# while taking into account normnalizations and noise bias
#
# Author: Harish Vedantham, ASTRON, 2019-2020
#
# List of input light curve files (ascii) in folder ../data/
#


flist = ["09-Apr-2019.dat",\
         "11-May-2019.dat",\
         "08-Jun-2019.dat",\
         "12-Jul-2019.dat",\
         "07-Aug-2019.dat",\
         "13-Sep-2019.dat",\
         "10-Oct-2019.dat",\
         "04-Nov-2019.dat",\
         "07-Dec-2019.dat",\
         "06-Jan-2020.dat",\
         "28-Jan-2020.dat",\
         "02-Mar-2020.dat"]

# List of dates corresponding to files
#
#
for fname in flist:
   #
   # READ IN DATA
   #
   tp = np.loadtxt("../data/%s"%fname) # Load light curve data
   ut = tp[:,0]	# UT time
   s = tp[:,1]*1e3*1.76  # Flux density in mJy
   #
   # SET UP VECTORS AND VARIABLES FOR ACF COMPUTATION
   #
   start_ut = ut[0] # Starting time in UT (seconds)
   ut-=ut[0]        # shift origin to start at 0 time 
   noise = np.var(np.diff(s))/2 # Noise variance from time differences   
   mind = (np.var(s)-noise)**0.5/np.mean(s) # Modulation index
   s-=np.mean(s)    # Mean subtract flux values
   npts = len(s)    # Num of points in light curve
   nbins = npts     # Number of bins for auto-corr computation (take = npts for now)
   lag = np.arange(0.0,nbins,1.0)*(ut[1]-ut[0]) # Range of temporal lags for ac computation
   acf = np.zeros(len(lag))     # Initialize autocorrelation vector to all zeros
   wt = np.zeros(len(lag))      # weight vector: many data points land into given bin

   #
   # COMPUTE ACF VECTOR USING ALL AVAILABLE PAIRS
   #
   for i in range(npts):	# For each flux measurement
      for j in range(i,npts):   # For each pair of flux measurements
         acf[j-i]+=s[i]*s[j]    # Accumulate acf at right lag
         wt[j-i]+=1		# Add 1 to weight at right lag
   acf/=wt			# Normalize acf by weights
   acf[0]-=noise		# Subtract noise bias
   acf/=acf[0]			# Normalize acf vector to unity at zero lag
   #
   # PLOT THE LIGHT CURVE ACF VECTOR AND COMPUTE 1/E TIMESACLE
   #
   plt.figure(figsize=(8/1.2,8/1.2))
   plt.subplot(211)
   plt.step(ut/60,s,'k')
   plt.xlabel("Time in minutes")
   plt.ylabel("Mean-subtracted flux [mJy]")
   y_lim = max(np.amax(np.absolute(s)),7)
   plt.ylim([-y_lim,y_lim])
   plt.title(fname[:-4])
   plt.subplot(212) 
   plt.step(lag/60,acf,'k') # x-axis is in units of minutes
   # Now find the first 1/e crossing (there may be more at larger lags!)
   I_cross = np.min(np.where(acf<1/np.exp(1.)))
   tau =  np.polyval( np.polyfit( \
          acf[I_cross-2:I_cross+2],lag[I_cross-2:I_cross+2]/60,1),\
          [1/np.exp(1)])   #Compute 1/e timescale using linear interpolation 
   N = (ut[-1]-ut[0])/60.0 / tau
   tau_err = tau * N**-0.5
   print("%s: 1/e timescale = %.2f mins, err = %.2f, mod. ind. = %.3f"%(fname,tau,tau_err,mind))	
   # 							Print the 1/e timesacle in mins
   #
   # ADD BELLS AND WHISTLES TO PLOT
   #
   plt.plot([-1e3,1e3],[1/np.exp(1),1/np.exp(1)],'k--') # 1/e line
   plt.ylim([-1.25,1.25])
   plt.xlim([-1,max(120,min(5*tau,5*60))])
   plt.plot([tau,tau],[-1.25,1.25],'k--') # 1/e timescale  line
   plt.xlabel("Time lag [minutes]")
   plt.ylabel("Autocorrelation function (normalised)")
   plt.tight_layout()
   plt.savefig("acfplots/%s.pdf"%fname[:-4]) # Make sure this folder exists in the current path
   plt.close()
   #
   # END
   #
