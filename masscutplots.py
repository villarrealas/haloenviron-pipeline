# required imports
import matplotlib
matplotlib.use('Agg') #make script not cry
import numpy as np
from numpy.lib.recfunctions import append_fields
import matplotlib.pyplot as plt
import scipy.stats as stats
import halotools.sim_manager as sm
import astropy.coordinates as coord

# we are going to read in a lot of data now.
# for now, let us do this one box size at a time.
# we start with l0250
gnewton = 4.302e-6

rs_dict = {'halo_id':(0,'i8'), 'halo_mass':(2,'f8'), 'halo_vmax':(3,'f8'), 'halo_rvir':(5,'f8'),
           'halo_rs':(6,'f8'), 'halo_x':(8,'f8'), 'halo_y':(9,'f8'), 'halo_z':(10,'f8'),
           'halo_spin':(17,'f8'), 'halo_ctoa':(28, 'f8'), 'halo_pid':(33,'i8')}

broadmassthresh = 1e11
vhost_min = 240.0
vrat_frac_min = 0.03
vsub_min = vhost_min * vrat_frac_min

fname_d200 = 'l0250_d200b.catalog'
fname_d100 = 'l0250_d100b.catalog'
fname_d75  = 'l0250_d75b.catalog'
fname_d50  = 'l0250_d50b.catalog'

reader = sm.TabularAsciiReader(fname_d200, rs_dict, row_cut_min_dict={'halo_mass':broadmassthresh}, row_cut_eq_dict={'halo_pid':-1})
hosts_data_d200 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d100, rs_dict, row_cut_min_dict={'halo_mass':broadmassthresh}, row_cut_eq_dict={'halo_pid':-1})
hosts_data_d100 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d75, rs_dict, row_cut_min_dict={'halo_mass':broadmassthresh}, row_cut_eq_dict={'halo_pid':-1})
hosts_data_d75 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d50, rs_dict, row_cut_min_dict={'halo_mass':broadmassthresh}, row_cut_eq_dict={'halo_pid':-1})
hosts_data_d50 = reader.read_ascii()

# here we will read in all the sub catalogs for making our satellite
# completeness test later.

reader = sm.TabularAsciiReader(fname_d200, rs_dict, row_cut_min_dict={'halo_vmax':vsub_min}, row_cut_neq_dict={'halo_pid':-1})
subs_data_d200 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d100, rs_dict, row_cut_min_dict={'halo_vmax':vsub_min}, row_cut_neq_dict={'halo_pid':-1})
subs_data_d100 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d75, rs_dict, row_cut_min_dict={'halo_vmax':vsub_min}, row_cut_neq_dict={'halo_pid':-1})
subs_data_d75 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d50, rs_dict, row_cut_min_dict={'halo_vmax':vsub_min}, row_cut_neq_dict={'halo_pid':-1})
subs_data_d50 = reader.read_ascii()

# now we can generate the data needed to make the halo occupation mark plot

nflagtemp = np.zeros(len(hosts_data_d200))
subcount = np.zeros( (len(hosts_data_d200), 10) )
fracrange = np.linspace(0.03,0.4,num=10)

for idx in range(0, len(fracrange)):
   for i in range(0, len(hosts_data_d200)):
      if hosts_data_d200[i]['halo_vmax'] > vhost_min:
         nflagtemp[i] = 1
         subs_cut_d200 = subs_data_d200[np.where(subs_data_d200['halo_pid'] == hosts_data_d200[i]['halo_id'])]
         for j in range(0, len(subs_cut_d200)):
            xdist = subs_cut_d200[j]['halo_x'] - hosts_data_d200[i]['halo_x']
            ydist = subs_cut_d200[j]['halo_y'] - hosts_data_d200[i]['halo_y']
            zdist = subs_cut_d200[j]['halo_z'] - hosts_data_d200[i]['halo_z']
            totdist = (np.sqrt(xdist**2 + ydist**2 + zdist**2))*0.001
            if totdist < hosts_data_d200[i]['halo_rvir']:
               ratio = subs_cut_d200[j]['halo_vmax']/hosts_data_d200[i]['halo_vmax']
               if ratio > fracrange[idx]:
                  subcount[i,idx] = subcount[i,idx]+1

meansubcount_l0250 = np.zeros( len(fracrange) )
for idx in range(0, len(fracrange)):
    meansubcount_l0250[idx] = np.mean(subcount[:,idx])

# now we are going to calculate our two additional marks for each one.
vratio_d200 = hosts_data_d200['halo_vmax']/(np.sqrt(gnewton*hosts_data_d200['halo_mass']/hosts_data_d200['halo_rvir']))
cnfw_d200 = hosts_data_d200['halo_rvir']/hosts_data_d200['halo_rs']

vratio_d100 = hosts_data_d100['halo_vmax']/(np.sqrt(gnewton*hosts_data_d100['halo_mass']/hosts_data_d100['halo_rvir']))
cnfw_d100 = hosts_data_d100['halo_rvir']/hosts_data_d100['halo_rs']

vratio_d75 = hosts_data_d75['halo_vmax']/(np.sqrt(gnewton*hosts_data_d75['halo_mass']/hosts_data_d75['halo_rvir']))
cnfw_d75 = hosts_data_d75['halo_rvir']/hosts_data_d75['halo_rs']

vratio_d50 = hosts_data_d50['halo_vmax']/(np.sqrt(gnewton*hosts_data_d50['halo_mass']/hosts_data_d50['halo_rvir']))
cnfw_d50 = hosts_data_d50['halo_rvir']/hosts_data_d50['halo_rs']

#vratio_d10 = hosts_data_d10['halo_vmax']/(np.sqrt(gnewton*hosts_data_d10['halo_mass']/hosts_data_d10['halo_rvir']))
#cnfw_d10 = hosts_data_d10['halo_rvir']/hosts_data_d10['halo_rs']

hosts_dataf_d200 = append_fields(hosts_data_d200, ('halo_cV', 'halo_cNFW'), (vratio_d200, cnfw_d200))
hosts_dataf_d100 = append_fields(hosts_data_d100, ('halo_cV', 'halo_cNFW'), (vratio_d100, cnfw_d100))
hosts_dataf_d75 = append_fields(hosts_data_d75, ('halo_cV', 'halo_cNFW'), (vratio_d75, cnfw_d75))
hosts_dataf_d50 = append_fields(hosts_data_d50, ('halo_cV', 'halo_cNFW'), (vratio_d50, cnfw_d50))

# all the data is now entered into our new arrays, so now it should be
# straightforward to make binned medians of all the data for purposes of
# determining best fit mass cuts.

total_bins = 30
X1_d200 = np.log10(hosts_dataf_d200['halo_mass'])
Y1_d200 = hosts_dataf_d200['halo_cNFW']
Y2_d200 = hosts_dataf_d200['halo_cV']
Y3_d200 = hosts_dataf_d200['halo_ctoa']
Y4_d200 = hosts_dataf_d200['halo_spin']

X1_d100 = np.log10(hosts_dataf_d100['halo_mass'])
Y1_d100 = hosts_dataf_d100['halo_cNFW']
Y2_d100 = hosts_dataf_d100['halo_cV']
Y3_d100 = hosts_dataf_d100['halo_ctoa']
Y4_d100 = hosts_dataf_d100['halo_spin']

X1_d75 = np.log10(hosts_dataf_d75['halo_mass'])
Y1_d75 = hosts_dataf_d75['halo_cNFW']
Y2_d75 = hosts_dataf_d75['halo_cV']
Y3_d75 = hosts_dataf_d75['halo_ctoa']
Y4_d75 = hosts_dataf_d75['halo_spin']

X1_d50 = np.log10(hosts_dataf_d50['halo_mass'])
Y1_d50 = hosts_dataf_d50['halo_cNFW']
Y2_d50 = hosts_dataf_d50['halo_cV']
Y3_d50 = hosts_dataf_d50['halo_ctoa']
Y4_d50 = hosts_dataf_d50['halo_spin']

#X1_d10 = np.log10(hosts_dataf_d10['halo_mass'])
#Y1_d10 = hosts_dataf_d10['halo_cNFW']
#Y2_d10 = hosts_dataf_d10['halo_cV']
#Y3_d10 = hosts_dataf_d10['halo_ctoa']
#Y4_d10 = hosts_dataf_d10['halo_spin']

cutoff_d200 = 7e11
cutoff_d100 = 8e11
cutoff_d75  = 9e11
cutoff_d50  = 1.5e12

bins_d200 = np.linspace(X1_d200.min(), X1_d200.max(), total_bins)
delta_d200 = bins_d200[1]-bins_d200[0]
idx_d200 = np.digitize(X1_d200, bins_d200)
running_median_d200 = [np.median(Y1_d200[idx_d200==k]) for k in range(total_bins)]
running_std_d200 = [Y1_d200[idx_d200==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d200-delta_d200/2), running_median_d200, running_std_d200, color='b', label=r'$\Delta=200$', lw=2)

bins_d100 = np.linspace(X1_d100.min(), X1_d100.max(), total_bins)
delta_d100 = bins_d100[1]-bins_d100[0]
idx_d100 = np.digitize(X1_d100, bins_d100)
running_median_d100 = [np.median(Y1_d100[idx_d100==k]) for k in range(total_bins)]
running_std_d100 = [Y1_d100[idx_d100==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d100-delta_d100/2), running_median_d100, running_std_d100, color='r', label=r'$\Delta=100$', lw=2)

bins_d75 = np.linspace(X1_d75.min(), X1_d75.max(), total_bins)
delta_d75 = bins_d75[1]-bins_d75[0]
idx_d75 = np.digitize(X1_d75, bins_d75)
running_median_d75 = [np.median(Y1_d75[idx_d75==k]) for k in range(total_bins)]
running_std_d75 = [Y1_d75[idx_d75==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d75-delta_d75/2), running_median_d75, running_std_d75, color='g', label=r'$\Delta=75$', lw=2)

bins_d50 = np.linspace(X1_d50.min(), X1_d50.max(), total_bins)
delta_d50 = bins_d50[1]-bins_d50[0]
idx_d50 = np.digitize(X1_d50, bins_d50)
running_median_d50 = [np.median(Y1_d50[idx_d50==k]) for k in range(total_bins)]
running_std_d50 = [Y1_d50[idx_d50==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d50-delta_d50/2), running_median_d50, running_std_d50, color='c', label=r'$\Delta=50$', lw=2)

#bins_d10 = np.linspace(X1_d10.min(), X1_d10.max(), total_bins)
#delta_d10 = bins_d10[1]-bins_d10[0]
#idx_d10 = np.digitize(X1_d10, bins_d10)
#running_median_d10 = [np.median(Y1_d10[idx_d10==k]) for k in range(total_bins)]
#running_std_d10 = [Y1_d10[idx_d10==k].std() for k in range(total_bins)]
#plt.errorbar(10**(bins_d10-delta_d10/2), running_median_d10, running_std_d10, color='darkgoldenrod', label=r'$\Delta=10$', lw=2)

plt.axvline(x=cutoff_d200, color='b', lw=3)
plt.axvline(x=cutoff_d100, color='r', lw=3)
plt.axvline(x=cutoff_d75, color='g', lw=3)
plt.axvline(x=cutoff_d50, color='c', lw=3)

plt.xlim(2e11,1e15)
plt.ylim(0,50)
plt.xscale('log')
plt.savefig('cnfwcut_l0250.png')
plt.clf()

bins_d200 = np.linspace(X1_d200.min(), X1_d200.max(), total_bins)
delta_d200 = bins_d200[1]-bins_d200[0]
idx_d200 = np.digitize(X1_d200, bins_d200)
running_median_d200 = [np.median(Y2_d200[idx_d200==k]) for k in range(total_bins)]
running_std_d200 = [Y2_d200[idx_d200==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d200-delta_d200/2), running_median_d200, running_std_d200, color='b', label=r'$\Delta=200$', lw=2)

bins_d100 = np.linspace(X1_d100.min(), X1_d100.max(), total_bins)
delta_d100 = bins_d100[1]-bins_d100[0]
idx_d100 = np.digitize(X1_d100, bins_d100)
running_median_d100 = [np.median(Y2_d100[idx_d100==k]) for k in range(total_bins)]
running_std_d100 = [Y2_d100[idx_d100==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d100-delta_d100/2), running_median_d100, running_std_d100, color='r', label=r'$\Delta=100$', lw=2)

bins_d75 = np.linspace(X1_d75.min(), X1_d75.max(), total_bins)
delta_d75 = bins_d75[1]-bins_d75[0]
idx_d75 = np.digitize(X1_d75, bins_d75)
running_median_d75 = [np.median(Y2_d75[idx_d75==k]) for k in range(total_bins)]
running_std_d75 = [Y2_d75[idx_d75==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d75-delta_d75/2), running_median_d75, running_std_d75, color='g', label=r'$\Delta=75$', lw=2)

bins_d50 = np.linspace(X1_d50.min(), X1_d50.max(), total_bins)
delta_d50 = bins_d50[1]-bins_d50[0]
idx_d50 = np.digitize(X1_d50, bins_d50)
running_median_d50 = [np.median(Y2_d50[idx_d50==k]) for k in range(total_bins)]
running_std_d50 = [Y2_d50[idx_d50==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d50-delta_d50/2), running_median_d50, running_std_d50, color='c', label=r'$\Delta=50$', lw=2)

#bins_d10 = np.linspace(X1_d10.min(), X1_d10.max(), total_bins)
#delta_d10 = bins_d10[1]-bins_d10[0]
#idx_d10 = np.digitize(X1_d10, bins_d10)
#running_median_d10 = [np.median(Y2_d10[idx_d10==k]) for k in range(total_bins)]
#running_std_d10 = [Y2_d10[idx_d10==k].std() for k in range(total_bins)]
#plt.errorbar(10**(bins_d10-delta_d10/2), running_median_d10, running_std_d10, color='darkgoldenrod', label=r'$\Delta=10$', lw=2)

plt.axvline(x=cutoff_d200, color='b', lw=3)
plt.axvline(x=cutoff_d100, color='r', lw=3)
plt.axvline(x=cutoff_d75, color='g', lw=3)
plt.axvline(x=cutoff_d50, color='c', lw=3)

plt.xlim(2e11,1e15)
#plt.ylim(0,50)
plt.xscale('log')
plt.savefig('cvcut_l0250.png')
plt.clf()

bins_d200 = np.linspace(X1_d200.min(), X1_d200.max(), total_bins)
delta_d200 = bins_d200[1]-bins_d200[0]
idx_d200 = np.digitize(X1_d200, bins_d200)
running_median_d200 = [np.median(Y3_d200[idx_d200==k]) for k in range(total_bins)]
running_std_d200 = [Y3_d200[idx_d200==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d200-delta_d200/2), running_median_d200, running_std_d200, color='b', label=r'$\Delta=200$', lw=2)

bins_d100 = np.linspace(X1_d100.min(), X1_d100.max(), total_bins)
delta_d100 = bins_d100[1]-bins_d100[0]
idx_d100 = np.digitize(X1_d100, bins_d100)
running_median_d100 = [np.median(Y3_d100[idx_d100==k]) for k in range(total_bins)]
running_std_d100 = [Y3_d100[idx_d100==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d100-delta_d100/2), running_median_d100, running_std_d100, color='r', label=r'$\Delta=100$', lw=2)

bins_d75 = np.linspace(X1_d75.min(), X1_d75.max(), total_bins)
delta_d75 = bins_d75[1]-bins_d75[0]
idx_d75 = np.digitize(X1_d75, bins_d75)
running_median_d75 = [np.median(Y3_d75[idx_d75==k]) for k in range(total_bins)]
running_std_d75 = [Y3_d75[idx_d75==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d75-delta_d75/2), running_median_d75, running_std_d75, color='g', label=r'$\Delta=75$', lw=2)

bins_d50 = np.linspace(X1_d50.min(), X1_d50.max(), total_bins)
delta_d50 = bins_d50[1]-bins_d50[0]
idx_d50 = np.digitize(X1_d50, bins_d50)
running_median_d50 = [np.median(Y3_d50[idx_d50==k]) for k in range(total_bins)]
running_std_d50 = [Y3_d50[idx_d50==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d50-delta_d50/2), running_median_d50, running_std_d50, color='c', label=r'$\Delta=50$', lw=2)

#bins_d10 = np.linspace(X1_d10.min(), X1_d10.max(), total_bins)
#delta_d10 = bins_d10[1]-bins_d10[0]
#idx_d10 = np.digitize(X1_d10, bins_d10)
#running_median_d10 = [np.median(Y3_d10[idx_d10==k]) for k in range(total_bins)]
#running_std_d10 = [Y3_d10[idx_d10==k].std() for k in range(total_bins)]
#plt.errorbar(10**(bins_d10-delta_d10/2), running_median_d10, running_std_d10, color='darkgoldenrod', label=r'$\Delta=10$', lw=2)

plt.axvline(x=cutoff_d200, color='b', lw=3)
plt.axvline(x=cutoff_d100, color='r', lw=3)
plt.axvline(x=cutoff_d75, color='g', lw=3)
plt.axvline(x=cutoff_d50, color='c', lw=3)

plt.xlim(2e11,1e15)
#plt.ylim(0,50)
plt.xscale('log')
plt.savefig('shapecut_l0250.png')
plt.clf()

bins_d200 = np.linspace(X1_d200.min(), X1_d200.max(), total_bins)
delta_d200 = bins_d200[1]-bins_d200[0]
idx_d200 = np.digitize(X1_d200, bins_d200)
running_median_d200 = [np.median(Y4_d200[idx_d200==k]) for k in range(total_bins)]
running_std_d200 = [Y4_d200[idx_d200==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d200-delta_d200/2), running_median_d200, running_std_d200, color='b', label=r'$\Delta=200$', lw=2)

bins_d100 = np.linspace(X1_d100.min(), X1_d100.max(), total_bins)
delta_d100 = bins_d100[1]-bins_d100[0]
idx_d100 = np.digitize(X1_d100, bins_d100)
running_median_d100 = [np.median(Y4_d100[idx_d100==k]) for k in range(total_bins)]
running_std_d100 = [Y4_d100[idx_d100==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d100-delta_d100/2), running_median_d100, running_std_d100, color='r', label=r'$\Delta=100$', lw=2)

bins_d75 = np.linspace(X1_d75.min(), X1_d75.max(), total_bins)
delta_d75 = bins_d75[1]-bins_d75[0]
idx_d75 = np.digitize(X1_d75, bins_d75)
running_median_d75 = [np.median(Y4_d75[idx_d75==k]) for k in range(total_bins)]
running_std_d75 = [Y4_d75[idx_d75==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d75-delta_d75/2), running_median_d75, running_std_d75, color='g', label=r'$\Delta=75$', lw=2)

bins_d50 = np.linspace(X1_d50.min(), X1_d50.max(), total_bins)
delta_d50 = bins_d50[1]-bins_d50[0]
idx_d50 = np.digitize(X1_d50, bins_d50)
running_median_d50 = [np.median(Y4_d50[idx_d50==k]) for k in range(total_bins)]
running_std_d50 = [Y4_d50[idx_d50==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d50-delta_d50/2), running_median_d50, running_std_d50, color='c', label=r'$\Delta=50$', lw=2)

#bins_d10 = np.linspace(X1_d10.min(), X1_d10.max(), total_bins)
#delta_d10 = bins_d10[1]-bins_d10[0]
#idx_d10 = np.digitize(X1_d10, bins_d10)
#running_median_d10 = [np.median(Y4_d10[idx_d10==k]) for k in range(total_bins)]
#running_std_d10 = [Y4_d10[idx_d10==k].std() for k in range(total_bins)]
#plt.errorbar(10**(bins_d10-delta_d10/2), running_median_d10, running_std_d10, color='darkgoldenrod', label=r'$\Delta=10$', lw=2)

plt.axvline(x=cutoff_d200, color='b', lw=3)
plt.axvline(x=cutoff_d100, color='r', lw=3)
plt.axvline(x=cutoff_d75, color='g', lw=3)
plt.axvline(x=cutoff_d50, color='c', lw=3)

plt.xlim(2e11,1e15)
#plt.ylim(0,50)
plt.xscale('log')
plt.savefig('spincut_l0250.png')
plt.clf()

## copy it all
## repeat for L0125
## profit

fname_d200 = 'l0125_d200b.catalog'
fname_d100 = 'l0125_d100b.catalog'
fname_d75  = 'l0125_d75b.catalog'
fname_d50  = 'l0125_d50b.catalog'
fname_d10  = 'l0125_d10b.catalog'
broadmassthresh = 1e10

reader = sm.TabularAsciiReader(fname_d200, rs_dict, row_cut_min_dict={'halo_mass':broadmassthresh}, row_cut_eq_dict={'halo_pid':-1})
hosts_data_d200 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d100, rs_dict, row_cut_min_dict={'halo_mass':broadmassthresh}, row_cut_eq_dict={'halo_pid':-1})
hosts_data_d100 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d75, rs_dict, row_cut_min_dict={'halo_mass':broadmassthresh}, row_cut_eq_dict={'halo_pid':-1})
hosts_data_d75 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d50, rs_dict, row_cut_min_dict={'halo_mass':broadmassthresh}, row_cut_eq_dict={'halo_pid':-1})
hosts_data_d50 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d10, rs_dict, row_cut_min_dict={'halo_mass':broadmassthresh}, row_cut_eq_dict={'halo_pid':-1})
hosts_data_d10 = reader.read_ascii()

# here we will read in all the sub catalogs for making our satellite
# completeness test later.

reader = sm.TabularAsciiReader(fname_d200, rs_dict, row_cut_min_dict={'halo_vmax':vsub_min}, row_cut_neq_dict={'halo_pid':-1})
subs_data_d200 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d100, rs_dict, row_cut_min_dict={'halo_vmax':vsub_min}, row_cut_neq_dict={'halo_pid':-1})
subs_data_d100 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d75, rs_dict, row_cut_min_dict={'halo_vmax':vsub_min}, row_cut_neq_dict={'halo_pid':-1})
subs_data_d75 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d50, rs_dict, row_cut_min_dict={'halo_vmax':vsub_min}, row_cut_neq_dict={'halo_pid':-1})
subs_data_d50 = reader.read_ascii()

# now we can generate the data needed to make the halo occupation mark plot

nflagtemp = np.zeros(len(hosts_data_d200))
subcount = np.zeros( (len(hosts_data_d200), 10) )
fracrange = np.linspace(0.03,0.4,num=10)

for idx in range(0, len(fracrange)):
   for i in range(0, len(hosts_data_d200)):
      if hosts_data_d200[i]['halo_vmax'] > vhost_min:
         nflagtemp[i] = 1
         subs_cut_d200 = subs_data_d200[np.where(subs_data_d200['halo_pid'] == hosts_data_d200[i]['halo_id'])]
         for j in range(0, len(subs_cut_d200)):
            xdist = subs_cut_d200[j]['halo_x'] - hosts_data_d200[i]['halo_x']
            ydist = subs_cut_d200[j]['halo_y'] - hosts_data_d200[i]['halo_y']
            zdist = subs_cut_d200[j]['halo_z'] - hosts_data_d200[i]['halo_z']
            totdist = (np.sqrt(xdist**2 + ydist**2 + zdist**2))*0.001
            if totdist < hosts_data_d200[i]['halo_rvir']:
               ratio = subs_cut_d200[j]['halo_vmax']/hosts_data_d200[i]['halo_vmax']
               if ratio > fracrange[idx]:
                  subcount[i,idx] = subcount[i,idx]+1

meansubcount_l0125 = np.zeros( len(fracrange) )
for idx in range(0, len(fracrange)):
    meansubcount_l0125[idx] = np.mean(subcount[:,idx])
# now we are going to calculate our two additional marks for each one.
vratio_d200 = hosts_data_d200['halo_vmax']/(np.sqrt(gnewton*hosts_data_d200['halo_mass']/hosts_data_d200['halo_rvir']))
cnfw_d200 = hosts_data_d200['halo_rvir']/hosts_data_d200['halo_rs']

vratio_d100 = hosts_data_d100['halo_vmax']/(np.sqrt(gnewton*hosts_data_d100['halo_mass']/hosts_data_d100['halo_rvir']))
cnfw_d100 = hosts_data_d100['halo_rvir']/hosts_data_d100['halo_rs']

vratio_d75 = hosts_data_d75['halo_vmax']/(np.sqrt(gnewton*hosts_data_d75['halo_mass']/hosts_data_d75['halo_rvir']))
cnfw_d75 = hosts_data_d75['halo_rvir']/hosts_data_d75['halo_rs']

vratio_d50 = hosts_data_d50['halo_vmax']/(np.sqrt(gnewton*hosts_data_d50['halo_mass']/hosts_data_d50['halo_rvir']))
cnfw_d50 = hosts_data_d50['halo_rvir']/hosts_data_d50['halo_rs']

vratio_d10 = hosts_data_d10['halo_vmax']/(np.sqrt(gnewton*hosts_data_d10['halo_mass']/hosts_data_d10['halo_rvir']))
cnfw_d10 = hosts_data_d10['halo_rvir']/hosts_data_d10['halo_rs']

hosts_dataf_d200 = append_fields(hosts_data_d200, ('halo_cV', 'halo_cNFW'), (vratio_d200, cnfw_d200))
hosts_dataf_d100 = append_fields(hosts_data_d100, ('halo_cV', 'halo_cNFW'), (vratio_d100, cnfw_d100))
hosts_dataf_d75 = append_fields(hosts_data_d75, ('halo_cV', 'halo_cNFW'), (vratio_d75, cnfw_d75))
hosts_dataf_d50 = append_fields(hosts_data_d50, ('halo_cV', 'halo_cNFW'), (vratio_d50, cnfw_d50))
hosts_dataf_d10 = append_fields(hosts_data_d10, ('halo_cV', 'halo_cNFW'), (vratio_d10, cnfw_d10))

# all the data is now entered into our new arrays, so now it should be
# straightforward to make binned medians of all the data for purposes of
# determining best fit mass cuts.

total_bins = 30
X1_d200 = np.log10(hosts_dataf_d200['halo_mass'])
Y1_d200 = hosts_dataf_d200['halo_cNFW']
Y2_d200 = hosts_dataf_d200['halo_cV']
Y3_d200 = hosts_dataf_d200['halo_ctoa']
Y4_d200 = hosts_dataf_d200['halo_spin']

X1_d100 = np.log10(hosts_dataf_d100['halo_mass'])
Y1_d100 = hosts_dataf_d100['halo_cNFW']
Y2_d100 = hosts_dataf_d100['halo_cV']
Y3_d100 = hosts_dataf_d100['halo_ctoa']
Y4_d100 = hosts_dataf_d100['halo_spin']

X1_d75 = np.log10(hosts_dataf_d75['halo_mass'])
Y1_d75 = hosts_dataf_d75['halo_cNFW']
Y2_d75 = hosts_dataf_d75['halo_cV']
Y3_d75 = hosts_dataf_d75['halo_ctoa']
Y4_d75 = hosts_dataf_d75['halo_spin']

X1_d50 = np.log10(hosts_dataf_d50['halo_mass'])
Y1_d50 = hosts_dataf_d50['halo_cNFW']
Y2_d50 = hosts_dataf_d50['halo_cV']
Y3_d50 = hosts_dataf_d50['halo_ctoa']
Y4_d50 = hosts_dataf_d50['halo_spin']

X1_d10 = np.log10(hosts_dataf_d10['halo_mass'])
Y1_d10 = hosts_dataf_d10['halo_cNFW']
Y2_d10 = hosts_dataf_d10['halo_cV']
Y3_d10 = hosts_dataf_d10['halo_ctoa']
Y4_d10 = hosts_dataf_d10['halo_spin']

cutoff_d200 = 7e10
cutoff_d100 = 8e10
cutoff_d75  = 9e10
cutoff_d50  = 1e11

bins_d200 = np.linspace(X1_d200.min(), X1_d200.max(), total_bins)
delta_d200 = bins_d200[1]-bins_d200[0]
idx_d200 = np.digitize(X1_d200, bins_d200)
running_median_d200 = [np.median(Y1_d200[idx_d200==k]) for k in range(total_bins)]
running_std_d200 = [Y1_d200[idx_d200==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d200-delta_d200/2), running_median_d200, running_std_d200, color='b', label=r'$\Delta=200$', lw=2)

bins_d100 = np.linspace(X1_d100.min(), X1_d100.max(), total_bins)
delta_d100 = bins_d100[1]-bins_d100[0]
idx_d100 = np.digitize(X1_d100, bins_d100)
running_median_d100 = [np.median(Y1_d100[idx_d100==k]) for k in range(total_bins)]
running_std_d100 = [Y1_d100[idx_d100==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d100-delta_d100/2), running_median_d100, running_std_d100, color='r', label=r'$\Delta=100$', lw=2)

bins_d75 = np.linspace(X1_d75.min(), X1_d75.max(), total_bins)
delta_d75 = bins_d75[1]-bins_d75[0]
idx_d75 = np.digitize(X1_d75, bins_d75)
running_median_d75 = [np.median(Y1_d75[idx_d75==k]) for k in range(total_bins)]
running_std_d75 = [Y1_d75[idx_d75==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d75-delta_d75/2), running_median_d75, running_std_d75, color='g', label=r'$\Delta=75$', lw=2)

bins_d50 = np.linspace(X1_d50.min(), X1_d50.max(), total_bins)
delta_d50 = bins_d50[1]-bins_d50[0]
idx_d50 = np.digitize(X1_d50, bins_d50)
running_median_d50 = [np.median(Y1_d50[idx_d50==k]) for k in range(total_bins)]
running_std_d50 = [Y1_d50[idx_d50==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d50-delta_d50/2), running_median_d50, running_std_d50, color='c', label=r'$\Delta=50$', lw=2)

bins_d10 = np.linspace(X1_d10.min(), X1_d10.max(), total_bins)
delta_d10 = bins_d10[1]-bins_d10[0]
idx_d10 = np.digitize(X1_d10, bins_d10)
running_median_d10 = [np.median(Y1_d10[idx_d10==k]) for k in range(total_bins)]
running_std_d10 = [Y1_d10[idx_d10==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d10-delta_d10/2), running_median_d10, running_std_d10, color='darkgoldenrod', label=r'$\Delta=10$', lw=2)

plt.axvline(x=cutoff_d200, color='b', lw=3)
plt.axvline(x=cutoff_d100, color='r', lw=3)
plt.axvline(x=cutoff_d75, color='g', lw=3)
plt.axvline(x=cutoff_d50, color='c', lw=3)

plt.xlim(2e10,1e15)
plt.ylim(0,100)
plt.xscale('log')
plt.savefig('cnfwcut_l0125.png')
plt.clf()

bins_d200 = np.linspace(X1_d200.min(), X1_d200.max(), total_bins)
delta_d200 = bins_d200[1]-bins_d200[0]
idx_d200 = np.digitize(X1_d200, bins_d200)
running_median_d200 = [np.median(Y2_d200[idx_d200==k]) for k in range(total_bins)]
running_std_d200 = [Y2_d200[idx_d200==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d200-delta_d200/2), running_median_d200, running_std_d200, color='b', label=r'$\Delta=200$', lw=2)

bins_d100 = np.linspace(X1_d100.min(), X1_d100.max(), total_bins)
delta_d100 = bins_d100[1]-bins_d100[0]
idx_d100 = np.digitize(X1_d100, bins_d100)
running_median_d100 = [np.median(Y2_d100[idx_d100==k]) for k in range(total_bins)]
running_std_d100 = [Y2_d100[idx_d100==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d100-delta_d100/2), running_median_d100, running_std_d100, color='r', label=r'$\Delta=100$', lw=2)

bins_d75 = np.linspace(X1_d75.min(), X1_d75.max(), total_bins)
delta_d75 = bins_d75[1]-bins_d75[0]
idx_d75 = np.digitize(X1_d75, bins_d75)
running_median_d75 = [np.median(Y2_d75[idx_d75==k]) for k in range(total_bins)]
running_std_d75 = [Y2_d75[idx_d75==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d75-delta_d75/2), running_median_d75, running_std_d75, color='g', label=r'$\Delta=75$', lw=2)

bins_d50 = np.linspace(X1_d50.min(), X1_d50.max(), total_bins)
delta_d50 = bins_d50[1]-bins_d50[0]
idx_d50 = np.digitize(X1_d50, bins_d50)
running_median_d50 = [np.median(Y2_d50[idx_d50==k]) for k in range(total_bins)]
running_std_d50 = [Y2_d50[idx_d50==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d50-delta_d50/2), running_median_d50, running_std_d50, color='c', label=r'$\Delta=50$', lw=2)

bins_d10 = np.linspace(X1_d10.min(), X1_d10.max(), total_bins)
delta_d10 = bins_d10[1]-bins_d10[0]
idx_d10 = np.digitize(X1_d10, bins_d10)
running_median_d10 = [np.median(Y2_d10[idx_d10==k]) for k in range(total_bins)]
running_std_d10 = [Y2_d10[idx_d10==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d10-delta_d10/2), running_median_d10, running_std_d10, color='darkgoldenrod', label=r'$\Delta=10$', lw=2)

plt.axvline(x=cutoff_d200, color='b', lw=3)
plt.axvline(x=cutoff_d100, color='r', lw=3)
plt.axvline(x=cutoff_d75, color='g', lw=3)
plt.axvline(x=cutoff_d50, color='c', lw=3)

plt.xlim(2e10,1e15)
#plt.ylim(0,50)
plt.xscale('log')
plt.savefig('cvcut_l0125.png')
plt.clf()

bins_d200 = np.linspace(X1_d200.min(), X1_d200.max(), total_bins)
delta_d200 = bins_d200[1]-bins_d200[0]
idx_d200 = np.digitize(X1_d200, bins_d200)
running_median_d200 = [np.median(Y3_d200[idx_d200==k]) for k in range(total_bins)]
running_std_d200 = [Y3_d200[idx_d200==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d200-delta_d200/2), running_median_d200, running_std_d200, color='b', label=r'$\Delta=200$', lw=2)

bins_d100 = np.linspace(X1_d100.min(), X1_d100.max(), total_bins)
delta_d100 = bins_d100[1]-bins_d100[0]
idx_d100 = np.digitize(X1_d100, bins_d100)
running_median_d100 = [np.median(Y3_d100[idx_d100==k]) for k in range(total_bins)]
running_std_d100 = [Y3_d100[idx_d100==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d100-delta_d100/2), running_median_d100, running_std_d100, color='r', label=r'$\Delta=100$', lw=2)

bins_d75 = np.linspace(X1_d75.min(), X1_d75.max(), total_bins)
delta_d75 = bins_d75[1]-bins_d75[0]
idx_d75 = np.digitize(X1_d75, bins_d75)
running_median_d75 = [np.median(Y3_d75[idx_d75==k]) for k in range(total_bins)]
running_std_d75 = [Y3_d75[idx_d75==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d75-delta_d75/2), running_median_d75, running_std_d75, color='g', label=r'$\Delta=75$', lw=2)

bins_d50 = np.linspace(X1_d50.min(), X1_d50.max(), total_bins)
delta_d50 = bins_d50[1]-bins_d50[0]
idx_d50 = np.digitize(X1_d50, bins_d50)
running_median_d50 = [np.median(Y3_d50[idx_d50==k]) for k in range(total_bins)]
running_std_d50 = [Y3_d50[idx_d50==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d50-delta_d50/2), running_median_d50, running_std_d50, color='c', label=r'$\Delta=50$', lw=2)

bins_d10 = np.linspace(X1_d10.min(), X1_d10.max(), total_bins)
delta_d10 = bins_d10[1]-bins_d10[0]
idx_d10 = np.digitize(X1_d10, bins_d10)
running_median_d10 = [np.median(Y3_d10[idx_d10==k]) for k in range(total_bins)]
running_std_d10 = [Y3_d10[idx_d10==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d10-delta_d10/2), running_median_d10, running_std_d10, color='darkgoldenrod', label=r'$\Delta=10$', lw=2)

plt.axvline(x=cutoff_d200, color='b', lw=3)
plt.axvline(x=cutoff_d100, color='r', lw=3)
plt.axvline(x=cutoff_d75, color='g', lw=3)
plt.axvline(x=cutoff_d50, color='c', lw=3)

plt.xlim(2e10,1e15)
#plt.ylim(0,50)
plt.xscale('log')
plt.savefig('shapecut_l0125.png')
plt.clf()

bins_d200 = np.linspace(X1_d200.min(), X1_d200.max(), total_bins)
delta_d200 = bins_d200[1]-bins_d200[0]
idx_d200 = np.digitize(X1_d200, bins_d200)
running_median_d200 = [np.median(Y4_d200[idx_d200==k]) for k in range(total_bins)]
running_std_d200 = [Y4_d200[idx_d200==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d200-delta_d200/2), running_median_d200, running_std_d200, color='b', label=r'$\Delta=200$', lw=2)

bins_d100 = np.linspace(X1_d100.min(), X1_d100.max(), total_bins)
delta_d100 = bins_d100[1]-bins_d100[0]
idx_d100 = np.digitize(X1_d100, bins_d100)
running_median_d100 = [np.median(Y4_d100[idx_d100==k]) for k in range(total_bins)]
running_std_d100 = [Y4_d100[idx_d100==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d100-delta_d100/2), running_median_d100, running_std_d100, color='r', label=r'$\Delta=100$', lw=2)

bins_d75 = np.linspace(X1_d75.min(), X1_d75.max(), total_bins)
delta_d75 = bins_d75[1]-bins_d75[0]
idx_d75 = np.digitize(X1_d75, bins_d75)
running_median_d75 = [np.median(Y4_d75[idx_d75==k]) for k in range(total_bins)]
running_std_d75 = [Y4_d75[idx_d75==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d75-delta_d75/2), running_median_d75, running_std_d75, color='g', label=r'$\Delta=75$', lw=2)

bins_d50 = np.linspace(X1_d50.min(), X1_d50.max(), total_bins)
delta_d50 = bins_d50[1]-bins_d50[0]
idx_d50 = np.digitize(X1_d50, bins_d50)
running_median_d50 = [np.median(Y4_d50[idx_d50==k]) for k in range(total_bins)]
running_std_d50 = [Y4_d50[idx_d50==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d50-delta_d50/2), running_median_d50, running_std_d50, color='c', label=r'$\Delta=50$', lw=2)

bins_d10 = np.linspace(X1_d10.min(), X1_d10.max(), total_bins)
delta_d10 = bins_d10[1]-bins_d10[0]
idx_d10 = np.digitize(X1_d10, bins_d10)
running_median_d10 = [np.median(Y4_d10[idx_d10==k]) for k in range(total_bins)]
running_std_d10 = [Y4_d10[idx_d10==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d10-delta_d10/2), running_median_d10, running_std_d10, color='darkgoldenrod', label=r'$\Delta=10$', lw=2)

plt.axvline(x=cutoff_d200, color='b', lw=3)
plt.axvline(x=cutoff_d100, color='r', lw=3)
plt.axvline(x=cutoff_d75, color='g', lw=3)
plt.axvline(x=cutoff_d50, color='c', lw=3)

plt.xlim(2e10,1e15)
#plt.ylim(0,50)
plt.xscale('log')
plt.savefig('spincut_l0125.png')
plt.clf()

## one more cut
## time to do l0500
## and swap delta = 10 for delta = 340!

fname_d200 = 'l0500_d200b.catalog'
fname_d100 = 'l0500_d100b.catalog'
fname_d75  = 'l0500_d75b.catalog'
fname_d50  = 'l0500_d50b.catalog'
fname_d340  = 'l0500_d340b.catalog'
broadmassthresh = 1e12

reader = sm.TabularAsciiReader(fname_d200, rs_dict, row_cut_min_dict={'halo_mass':broadmassthresh}, row_cut_eq_dict={'halo_pid':-1})
hosts_data_d200 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d100, rs_dict, row_cut_min_dict={'halo_mass':broadmassthresh}, row_cut_eq_dict={'halo_pid':-1})
hosts_data_d100 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d75, rs_dict, row_cut_min_dict={'halo_mass':broadmassthresh}, row_cut_eq_dict={'halo_pid':-1})
hosts_data_d75 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d50, rs_dict, row_cut_min_dict={'halo_mass':broadmassthresh}, row_cut_eq_dict={'halo_pid':-1})
hosts_data_d50 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d340, rs_dict, row_cut_min_dict={'halo_mass':broadmassthresh}, row_cut_eq_dict={'halo_pid':-1})
hosts_data_d340 = reader.read_ascii()

# here we will read in all the sub catalogs for making our satellite
# completeness test later.

reader = sm.TabularAsciiReader(fname_d200, rs_dict, row_cut_min_dict={'halo_vmax':vsub_min}, row_cut_neq_dict={'halo_pid':-1})
subs_data_d200 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d100, rs_dict, row_cut_min_dict={'halo_vmax':vsub_min}, row_cut_neq_dict={'halo_pid':-1})
subs_data_d100 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d75, rs_dict, row_cut_min_dict={'halo_vmax':vsub_min}, row_cut_neq_dict={'halo_pid':-1})
subs_data_d75 = reader.read_ascii()

reader = sm.TabularAsciiReader(fname_d50, rs_dict, row_cut_min_dict={'halo_vmax':vsub_min}, row_cut_neq_dict={'halo_pid':-1})
subs_data_d50 = reader.read_ascii()

# now we can generate the data needed to make the halo occupation mark plot

nflagtemp = np.zeros(len(hosts_data_d200))
subcount = np.zeros( (len(hosts_data_d200), 10) )
fracrange = np.linspace(0.03,0.4,num=10)

for idx in range(0, len(fracrange)):
   for i in range(0, len(hosts_data_d200)):
      if hosts_data_d200[i]['halo_vmax'] > vhost_min:
         nflagtemp[i] = 1
         subs_cut_d200 = subs_data_d200[np.where(subs_data_d200['halo_pid'] == hosts_data_d200[i]['halo_id'])]
         for j in range(0, len(subs_cut_d200)):
            xdist = subs_cut_d200[j]['halo_x'] - hosts_data_d200[i]['halo_x']
            ydist = subs_cut_d200[j]['halo_y'] - hosts_data_d200[i]['halo_y']
            zdist = subs_cut_d200[j]['halo_z'] - hosts_data_d200[i]['halo_z']
            totdist = (np.sqrt(xdist**2 + ydist**2 + zdist**2))*0.001
            if totdist < hosts_data_d200[i]['halo_rvir']:
               ratio = subs_cut_d200[j]['halo_vmax']/hosts_data_d200[i]['halo_vmax']
               if ratio > fracrange[idx]:
                  subcount[i,idx] = subcount[i,idx]+1

meansubcount_l0500 = np.zeros( len(fracrange) )
for idx in range(0, len(fracrange)):
    meansubcount_l0500[idx] = np.mean(subcount[:,idx])

plt.plot(fracrange, meansubcount_l0125/(125.**3), 'b.')
plt.plot(fracrange, meansubcount_l0250/(250.**3), 'r.')
plt.plot(fracrange, meansubcount_l0500/(500.**3), 'g.')
plt.savefig('testnsat.png')
plt.clf()

# now we are going to calculate our two additional marks for each one.
vratio_d200 = hosts_data_d200['halo_vmax']/(np.sqrt(gnewton*hosts_data_d200['halo_mass']/hosts_data_d200['halo_rvir']))
cnfw_d200 = hosts_data_d200['halo_rvir']/hosts_data_d200['halo_rs']

vratio_d100 = hosts_data_d100['halo_vmax']/(np.sqrt(gnewton*hosts_data_d100['halo_mass']/hosts_data_d100['halo_rvir']))
cnfw_d100 = hosts_data_d100['halo_rvir']/hosts_data_d100['halo_rs']

vratio_d75 = hosts_data_d75['halo_vmax']/(np.sqrt(gnewton*hosts_data_d75['halo_mass']/hosts_data_d75['halo_rvir']))
cnfw_d75 = hosts_data_d75['halo_rvir']/hosts_data_d75['halo_rs']

vratio_d50 = hosts_data_d50['halo_vmax']/(np.sqrt(gnewton*hosts_data_d50['halo_mass']/hosts_data_d50['halo_rvir']))
cnfw_d50 = hosts_data_d50['halo_rvir']/hosts_data_d50['halo_rs']

vratio_d340 = hosts_data_d340['halo_vmax']/(np.sqrt(gnewton*hosts_data_d340['halo_mass']/hosts_data_d340['halo_rvir']))
cnfw_d340 = hosts_data_d340['halo_rvir']/hosts_data_d340['halo_rs']

hosts_dataf_d200 = append_fields(hosts_data_d200, ('halo_cV', 'halo_cNFW'), (vratio_d200, cnfw_d200))
hosts_dataf_d100 = append_fields(hosts_data_d100, ('halo_cV', 'halo_cNFW'), (vratio_d100, cnfw_d100))
hosts_dataf_d75 = append_fields(hosts_data_d75, ('halo_cV', 'halo_cNFW'), (vratio_d75, cnfw_d75))
hosts_dataf_d50 = append_fields(hosts_data_d50, ('halo_cV', 'halo_cNFW'), (vratio_d50, cnfw_d50))
hosts_dataf_d340 = append_fields(hosts_data_d340, ('halo_cV', 'halo_cNFW'), (vratio_d340, cnfw_d340))

# all the data is now entered into our new arrays, so now it should be
# straightforward to make binned medians of all the data for purposes of
# determining best fit mass cuts.

total_bins = 30
X1_d200 = np.log10(hosts_dataf_d200['halo_mass'])
Y1_d200 = hosts_dataf_d200['halo_cNFW']
Y2_d200 = hosts_dataf_d200['halo_cV']
Y3_d200 = hosts_dataf_d200['halo_ctoa']
Y4_d200 = hosts_dataf_d200['halo_spin']

X1_d100 = np.log10(hosts_dataf_d100['halo_mass'])
Y1_d100 = hosts_dataf_d100['halo_cNFW']
Y2_d100 = hosts_dataf_d100['halo_cV']
Y3_d100 = hosts_dataf_d100['halo_ctoa']
Y4_d100 = hosts_dataf_d100['halo_spin']

X1_d75 = np.log10(hosts_dataf_d75['halo_mass'])
Y1_d75 = hosts_dataf_d75['halo_cNFW']
Y2_d75 = hosts_dataf_d75['halo_cV']
Y3_d75 = hosts_dataf_d75['halo_ctoa']
Y4_d75 = hosts_dataf_d75['halo_spin']

X1_d50 = np.log10(hosts_dataf_d50['halo_mass'])
Y1_d50 = hosts_dataf_d50['halo_cNFW']
Y2_d50 = hosts_dataf_d50['halo_cV']
Y3_d50 = hosts_dataf_d50['halo_ctoa']
Y4_d50 = hosts_dataf_d50['halo_spin']

X1_d340 = np.log10(hosts_dataf_d340['halo_mass'])
Y1_d340 = hosts_dataf_d340['halo_cNFW']
Y2_d340 = hosts_dataf_d340['halo_cV']
Y3_d340 = hosts_dataf_d340['halo_ctoa']
Y4_d340 = hosts_dataf_d340['halo_spin']

cutoff_d200 = 4e12
cutoff_d100 = 5e12
cutoff_d75  = 6e12
cutoff_d50  = 7e12

bins_d200 = np.linspace(X1_d200.min(), X1_d200.max(), total_bins)
delta_d200 = bins_d200[1]-bins_d200[0]
idx_d200 = np.digitize(X1_d200, bins_d200)
running_median_d200 = [np.median(Y1_d200[idx_d200==k]) for k in range(total_bins)]
running_std_d200 = [Y1_d200[idx_d200==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d200-delta_d200/2), running_median_d200, running_std_d200, color='b', label=r'$\Delta=200$', lw=2)

bins_d100 = np.linspace(X1_d100.min(), X1_d100.max(), total_bins)
delta_d100 = bins_d100[1]-bins_d100[0]
idx_d100 = np.digitize(X1_d100, bins_d100)
running_median_d100 = [np.median(Y1_d100[idx_d100==k]) for k in range(total_bins)]
running_std_d100 = [Y1_d100[idx_d100==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d100-delta_d100/2), running_median_d100, running_std_d100, color='r', label=r'$\Delta=100$', lw=2)

bins_d75 = np.linspace(X1_d75.min(), X1_d75.max(), total_bins)
delta_d75 = bins_d75[1]-bins_d75[0]
idx_d75 = np.digitize(X1_d75, bins_d75)
running_median_d75 = [np.median(Y1_d75[idx_d75==k]) for k in range(total_bins)]
running_std_d75 = [Y1_d75[idx_d75==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d75-delta_d75/2), running_median_d75, running_std_d75, color='g', label=r'$\Delta=75$', lw=2)

bins_d50 = np.linspace(X1_d50.min(), X1_d50.max(), total_bins)
delta_d50 = bins_d50[1]-bins_d50[0]
idx_d50 = np.digitize(X1_d50, bins_d50)
running_median_d50 = [np.median(Y1_d50[idx_d50==k]) for k in range(total_bins)]
running_std_d50 = [Y1_d50[idx_d50==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d50-delta_d50/2), running_median_d50, running_std_d50, color='c', label=r'$\Delta=50$', lw=2)

bins_d340 = np.linspace(X1_d340.min(), X1_d340.max(), total_bins)
delta_d340 = bins_d340[1]-bins_d340[0]
idx_d340 = np.digitize(X1_d340, bins_d340)
running_median_d340 = [np.median(Y1_d340[idx_d340==k]) for k in range(total_bins)]
running_std_d340 = [Y1_d340[idx_d340==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d340-delta_d340/2), running_median_d340, running_std_d340, color='m', label=r'$\Delta=340$', lw=2)

plt.axvline(x=cutoff_d200, color='b', lw=3)
plt.axvline(x=cutoff_d100, color='r', lw=3)
plt.axvline(x=cutoff_d75, color='g', lw=3)
plt.axvline(x=cutoff_d50, color='c', lw=3)

plt.xlim(2e12,1e15)
plt.ylim(0,40)
plt.xscale('log')
plt.savefig('cnfwcut_l0500.png')
plt.clf()

bins_d200 = np.linspace(X1_d200.min(), X1_d200.max(), total_bins)
delta_d200 = bins_d200[1]-bins_d200[0]
idx_d200 = np.digitize(X1_d200, bins_d200)
running_median_d200 = [np.median(Y2_d200[idx_d200==k]) for k in range(total_bins)]
running_std_d200 = [Y2_d200[idx_d200==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d200-delta_d200/2), running_median_d200, running_std_d200, color='b', label=r'$\Delta=200$', lw=2)

bins_d100 = np.linspace(X1_d100.min(), X1_d100.max(), total_bins)
delta_d100 = bins_d100[1]-bins_d100[0]
idx_d100 = np.digitize(X1_d100, bins_d100)
running_median_d100 = [np.median(Y2_d100[idx_d100==k]) for k in range(total_bins)]
running_std_d100 = [Y2_d100[idx_d100==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d100-delta_d100/2), running_median_d100, running_std_d100, color='r', label=r'$\Delta=100$', lw=2)

bins_d75 = np.linspace(X1_d75.min(), X1_d75.max(), total_bins)
delta_d75 = bins_d75[1]-bins_d75[0]
idx_d75 = np.digitize(X1_d75, bins_d75)
running_median_d75 = [np.median(Y2_d75[idx_d75==k]) for k in range(total_bins)]
running_std_d75 = [Y2_d75[idx_d75==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d75-delta_d75/2), running_median_d75, running_std_d75, color='g', label=r'$\Delta=75$', lw=2)

bins_d50 = np.linspace(X1_d50.min(), X1_d50.max(), total_bins)
delta_d50 = bins_d50[1]-bins_d50[0]
idx_d50 = np.digitize(X1_d50, bins_d50)
running_median_d50 = [np.median(Y2_d50[idx_d50==k]) for k in range(total_bins)]
running_std_d50 = [Y2_d50[idx_d50==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d50-delta_d50/2), running_median_d50, running_std_d50, color='c', label=r'$\Delta=50$', lw=2)

bins_d340 = np.linspace(X1_d340.min(), X1_d340.max(), total_bins)
delta_d340 = bins_d340[1]-bins_d340[0]
idx_d340 = np.digitize(X1_d340, bins_d340)
running_median_d340 = [np.median(Y2_d340[idx_d340==k]) for k in range(total_bins)]
running_std_d340 = [Y2_d340[idx_d340==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d340-delta_d340/2), running_median_d340, running_std_d340, color='m', label=r'$\Delta=340$', lw=2)

plt.axvline(x=cutoff_d200, color='b', lw=3)
plt.axvline(x=cutoff_d100, color='r', lw=3)
plt.axvline(x=cutoff_d75, color='g', lw=3)
plt.axvline(x=cutoff_d50, color='c', lw=3)

plt.xlim(2e12,1e15)
#plt.ylim(0,50)
plt.xscale('log')
plt.savefig('cvcut_l0500.png')
plt.clf()

bins_d200 = np.linspace(X1_d200.min(), X1_d200.max(), total_bins)
delta_d200 = bins_d200[1]-bins_d200[0]
idx_d200 = np.digitize(X1_d200, bins_d200)
running_median_d200 = [np.median(Y3_d200[idx_d200==k]) for k in range(total_bins)]
running_std_d200 = [Y3_d200[idx_d200==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d200-delta_d200/2), running_median_d200, running_std_d200, color='b', label=r'$\Delta=200$', lw=2)

bins_d100 = np.linspace(X1_d100.min(), X1_d100.max(), total_bins)
delta_d100 = bins_d100[1]-bins_d100[0]
idx_d100 = np.digitize(X1_d100, bins_d100)
running_median_d100 = [np.median(Y3_d100[idx_d100==k]) for k in range(total_bins)]
running_std_d100 = [Y3_d100[idx_d100==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d100-delta_d100/2), running_median_d100, running_std_d100, color='r', label=r'$\Delta=100$', lw=2)

bins_d75 = np.linspace(X1_d75.min(), X1_d75.max(), total_bins)
delta_d75 = bins_d75[1]-bins_d75[0]
idx_d75 = np.digitize(X1_d75, bins_d75)
running_median_d75 = [np.median(Y3_d75[idx_d75==k]) for k in range(total_bins)]
running_std_d75 = [Y3_d75[idx_d75==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d75-delta_d75/2), running_median_d75, running_std_d75, color='g', label=r'$\Delta=75$', lw=2)

bins_d50 = np.linspace(X1_d50.min(), X1_d50.max(), total_bins)
delta_d50 = bins_d50[1]-bins_d50[0]
idx_d50 = np.digitize(X1_d50, bins_d50)
running_median_d50 = [np.median(Y3_d50[idx_d50==k]) for k in range(total_bins)]
running_std_d50 = [Y3_d50[idx_d50==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d50-delta_d50/2), running_median_d50, running_std_d50, color='c', label=r'$\Delta=50$', lw=2)

bins_d340 = np.linspace(X1_d340.min(), X1_d340.max(), total_bins)
delta_d340 = bins_d340[1]-bins_d340[0]
idx_d340 = np.digitize(X1_d340, bins_d340)
running_median_d340 = [np.median(Y3_d340[idx_d340==k]) for k in range(total_bins)]
running_std_d340 = [Y3_d340[idx_d340==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d340-delta_d340/2), running_median_d340, running_std_d340, color='m', label=r'$\Delta=340$', lw=2)

plt.axvline(x=cutoff_d200, color='b', lw=3)
plt.axvline(x=cutoff_d100, color='r', lw=3)
plt.axvline(x=cutoff_d75, color='g', lw=3)
plt.axvline(x=cutoff_d50, color='c', lw=3)

plt.xlim(2e12,1e15)
#plt.ylim(0,50)
plt.xscale('log')
plt.savefig('shapecut_l0500.png')
plt.clf()

bins_d200 = np.linspace(X1_d200.min(), X1_d200.max(), total_bins)
delta_d200 = bins_d200[1]-bins_d200[0]
idx_d200 = np.digitize(X1_d200, bins_d200)
running_median_d200 = [np.median(Y4_d200[idx_d200==k]) for k in range(total_bins)]
running_std_d200 = [Y4_d200[idx_d200==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d200-delta_d200/2), running_median_d200, running_std_d200, color='b', label=r'$\Delta=200$', lw=2)

bins_d100 = np.linspace(X1_d100.min(), X1_d100.max(), total_bins)
delta_d100 = bins_d100[1]-bins_d100[0]
idx_d100 = np.digitize(X1_d100, bins_d100)
running_median_d100 = [np.median(Y4_d100[idx_d100==k]) for k in range(total_bins)]
running_std_d100 = [Y4_d100[idx_d100==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d100-delta_d100/2), running_median_d100, running_std_d100, color='r', label=r'$\Delta=100$', lw=2)

bins_d75 = np.linspace(X1_d75.min(), X1_d75.max(), total_bins)
delta_d75 = bins_d75[1]-bins_d75[0]
idx_d75 = np.digitize(X1_d75, bins_d75)
running_median_d75 = [np.median(Y4_d75[idx_d75==k]) for k in range(total_bins)]
running_std_d75 = [Y4_d75[idx_d75==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d75-delta_d75/2), running_median_d75, running_std_d75, color='g', label=r'$\Delta=75$', lw=2)

bins_d50 = np.linspace(X1_d50.min(), X1_d50.max(), total_bins)
delta_d50 = bins_d50[1]-bins_d50[0]
idx_d50 = np.digitize(X1_d50, bins_d50)
running_median_d50 = [np.median(Y4_d50[idx_d50==k]) for k in range(total_bins)]
running_std_d50 = [Y4_d50[idx_d50==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d50-delta_d50/2), running_median_d50, running_std_d50, color='c', label=r'$\Delta=50$', lw=2)

bins_d340 = np.linspace(X1_d340.min(), X1_d340.max(), total_bins)
delta_d340 = bins_d340[1]-bins_d340[0]
idx_d340 = np.digitize(X1_d340, bins_d340)
running_median_d340 = [np.median(Y4_d340[idx_d340==k]) for k in range(total_bins)]
running_std_d340 = [Y4_d340[idx_d340==k].std() for k in range(total_bins)]
plt.errorbar(10**(bins_d340-delta_d340/2), running_median_d340, running_std_d340, color='m', label=r'$\Delta=340$', lw=2)

plt.axvline(x=cutoff_d200, color='b', lw=3)
plt.axvline(x=cutoff_d100, color='r', lw=3)
plt.axvline(x=cutoff_d75, color='g', lw=3)
plt.axvline(x=cutoff_d50, color='c', lw=3)

plt.xlim(2e12,1e15)
#plt.ylim(0,50)
plt.xscale('log')
plt.savefig('spincut_l0500.png')
