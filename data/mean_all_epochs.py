# Program to read light curves for all epochs and print the mean flux, thermal noise and modulation index
import numpy as np

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

print ("DATE, MEAN, NOISE, MOD. IND")
for f in flist:
   tp = np.loadtxt(f)[:,1]
   tp*=1e3
   mean = np.mean(tp)
   tp-=mean
   noise = np.std(np.diff(tp))/2**0.5
   rms = (np.var(tp)-noise**2)**0.5
   print ("%s, %.2f, %.2f, %.2f"%(f,mean*1.76,noise*1.76,rms/mean))
