# Program to reach light curves for all epochs and plot them
import matplotlib.pyplot as plt
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

fig,ax = plt.subplots(6,2,sharex=True,sharey=True,squeeze=True,gridspec_kw = {'wspace':0.03,'hspace':0.05},figsize=(8,8))
counter=0
for row in range(6):
   for col in range(2):
      tp = np.loadtxt(flist[counter])
      x = tp[:,0]
      y = tp[:,1]*1.76
      x-=x[0]
      x/=2600.0
      y*=1e3
      ax[row,col].plot(x,y) 
      ax[row,col].set_xlim([-0.5,16.5])
      ax[row,col].set_ylim([0,37.5])
      ax[row,col].text(x=2,y=30,s="%s"%flist[counter][:-4])
      counter+=1

ax[5,0].set_xlabel("Time [hours]")
ax[5,1].set_xlabel("Time [hours]")
ax[2,0].set_ylabel("Flux-density [mJy]")
plt.tight_layout()
plt.savefig("lc_gallery.pdf")

plt.close()
