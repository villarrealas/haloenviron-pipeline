# required imports for the function here
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import halotools.mock_observables as mo
import halotools.sim_manager as sm

fname = './l0125_d50b.catalog' # file name to z0.0.catalog
mthresh = 1e11  # threshold mass in Msun/h
gnewton = 4.302e-6
lbox = 125

# define dict pointing to all marks of interest
rs_dict = {'halo_id':(0,'i8'), 'halo_mass':(2,'f8'), 'halo_vmax':(3,'f8'), 'halo_rvir':(5,'f8'),
           'halo_rs':(6,'f8'), 'halo_x':(8,'f8'), 'halo_y':(9,'f8'), 'halo_z':(10,'f8'),
           'halo_spin':(17,'f8'), 'halo_ctoa':(28, 'f8'), 'halo_pid':(33,'i8'),
           'halo_cnfw':(11,'f8'), 'halo_vratio':(12,'f8')}

reader = sm.TabularAsciiReader(fname, rs_dict, row_cut_min_dict={'halo_mass':mthresh},
                               row_cut_eq_dict={'halo_pid':-1})
hosts_data = reader.read_ascii()

# add vratio mark
vratio_temp = hosts_data['halo_vmax']/(np.sqrt(gnewton * hosts_data['halo_mass']/hosts_data['halo_rvir']))
hosts_data['halo_vratio'] = vratio_temp
cnfw_temp = hosts_data['halo_rvir']/hosts_data['halo_rs']
hosts_data['halo_cnfw'] = cnfw_temp

# now that we have data, we need to mass correct all of our marks
mass_sort = np.sort(hosts_data, order='halo_mass')

vratio_binned = stats.binned_statistic(np.log10(mass_sort['halo_mass']), 
                                       np.log10(mass_sort['halo_vratio']),
                                       statistic='mean', bins=10)
vratio_fix = np.log10(mass_sort['halo_vratio']) - vratio_binned.statistic[vratio_binned.binnumber-1]
mass_sort['halo_vratio'] = vratio_fix

cnfw_binned = stats.binned_statistic(np.log10(mass_sort['halo_mass']), 
                                     np.log10(mass_sort['halo_cnfw']),
                                     statistic = 'mean', bins=10)
cnfw_fix = np.log10(mass_sort['halo_cnfw']) - cnfw_binned.statistic[cnfw_binned.binnumber-1]
mass_sort['halo_cnfw'] = cnfw_fix

shape_binned = stats.binned_statistic(np.log10(mass_sort['halo_mass']), 
                                      np.log10(mass_sort['halo_ctoa']),
                                      statistic='mean', bins=10)
shape_fix = np.log10(mass_sort['halo_ctoa']) - shape_binned.statistic[shape_binned.binnumber-1]
mass_sort['halo_ctoa'] = shape_fix

spin_binned = stats.binned_statistic(np.log10(mass_sort['halo_mass']), 
                                     (mass_sort['halo_spin']),
                                     statistic='mean', bins=10)
spin_fix = (mass_sort['halo_spin']) - spin_binned.statistic[spin_binned.binnumber-1]
mass_sort['halo_spin'] = spin_fix

# now all our marks have been fixed. First let's run through the marked correlation functions
# then we can take various correlation function comparisons

pos = np.vstack((mass_sort['halo_x'], mass_sort['halo_y'], mass_sort['halo_z'])).T

minlog = np.log10(3.0)
maxlog = np.log10(30.0)
nstep = 30
steplog = (maxlog - minlog) / (nstep - 1.)
logbins = np.arange(minlog, maxlog+steplog, steplog)
binmids = np.zeros(30)
for i in range(0,nstep):
    binmids[i] = (logbins[i]+logbins[i+1])/2.

xi = mo.tpcf(pos, 10**logbins, period=lbox)

mcf_vratio = mo.marked_tpcf(pos, 10**logbins, marks1=mass_sort['halo_vratio'], period=lbox,
                            normalize_by='number_counts', wfunc=1)
mcfn_vratio = (mcf_vratio - np.mean(mass_sort['halo_vratio'])**2)/(np.var(mass_sort['halo_vratio']))

mcf_cnfw = mo.marked_tpcf(pos, 10**logbins, marks1=mass_sort['halo_cnfw'], period=lbox,
                          normalize_by='number_counts', wfunc=1)
mcfn_cnfw = (mcf_cnfw - np.mean(mass_sort['halo_cnfw'])**2)/(np.var(mass_sort['halo_cnfw']))

mcf_ctoa = mo.marked_tpcf(pos, 10**logbins, marks1=mass_sort['halo_ctoa'], period=lbox,
                          normalize_by='number_counts', wfunc=1)
mcfn_ctoa = (mcf_ctoa - np.mean(mass_sort['halo_ctoa'])**2)/(np.var(mass_sort['halo_ctoa']))

mcf_spin = mo.marked_tpcf(pos, 10**logbins, marks1=mass_sort['halo_spin'], period=lbox,
                          normalize_by='number_counts', wfunc=1)
mcfn_spin = (mcf_spin - np.mean(mass_sort['halo_spin'])**2)/(np.var(mass_sort['halo_spin']))

# how we'll need to shuffle the marks N times, run the calculation N times,
# and determine the min and max range of the mark calculation. So:
nrand = 50
mcf_vratio_rand = np.zeros((nstep, nrand))
mcfn_vratio_rand = np.zeros((nstep, nrand))
mcf_cnfw_rand = np.zeros((nstep, nrand))
mcfn_cnfw_rand = np.zeros((nstep, nrand))
mcf_ctoa_rand = np.zeros((nstep, nrand))
mcfn_ctoa_rand = np.zeros((nstep, nrand))
mcf_spin_rand = np.zeros((nstep, nrand))
mcfn_spin_rand = np.zeros((nstep, nrand))
for i in range(0, nrand):
    randm_vratio = np.random.permutation(mass_sort['halo_vratio'])
    randm_cnfw = np.random.permutation(mass_sort['halo_cnfw'])
    randm_ctoa = np.random.permutation(mass_sort['halo_ctoa'])
    randm_spin = np.random.permutation(mass_sort['halo_spin'])
    mcf_vratio_rand[:,i] = mo.marked_tpcf(pos, 10**logbins, marks1=randm_vratio, period=lbox,
                                      normalize_by='number_counts', wfunc=1)
    mcfn_vratio_rand[:,i] = (mcf_vratio_rand[:,i] - np.mean(randm_vratio)**2)/(np.var(randm_vratio))
    mcf_cnfw_rand[:,i] = mo.marked_tpcf(pos, 10**logbins, marks1=randm_cnfw, period=lbox,
                                        normalize_by='number_counts', wfunc=1)
    mcfn_cnfw_rand[:,i] = (mcf_cnfw_rand[:,i] - np.mean(randm_cnfw)**2)/(np.var(randm_cnfw))
    mcf_ctoa_rand[:,i] = mo.marked_tpcf(pos, 10**logbins, marks1=randm_ctoa, period=lbox,
                                        normalize_by='number_counts', wfunc=1)
    mcfn_ctoa_rand[:,i] = (mcf_ctoa_rand[:,i] - np.mean(randm_ctoa)**2)/(np.var(randm_ctoa))
    mcf_spin_rand[:,i] = mo.marked_tpcf(pos, 10**logbins, marks1=randm_spin, period=lbox,
                                        normalize_by='number_counts', wfunc=1)
    mcfn_spin_rand[:,i] = (mcf_spin_rand[:,i] - np.mean(randm_spin)**2)/(np.var(randm_spin))

mcfn_vratio_min = np.zeros(nstep)
mcfn_vratio_max = np.zeros(nstep)
mcfn_cnfw_min = np.zeros(nstep)
mcfn_cnfw_max = np.zeros(nstep)
mcfn_ctoa_min = np.zeros(nstep)
mcfn_ctoa_max = np.zeros(nstep)
mcfn_spin_min = np.zeros(nstep)
mcfn_spin_max = np.zeros(nstep)
for i in range(0, nstep):
    mcfn_vratio_min[i] = np.percentile(mcfn_vratio_rand[i,:],2, interpolation='nearest')
    mcfn_vratio_max[i] = np.percentile(mcfn_vratio_rand[i,:],98,interpolation='nearest')
    mcfn_cnfw_min[i] = np.percentile(mcfn_cnfw_rand[i,:],2, interpolation='nearest')
    mcfn_cnfw_max[i] = np.percentile(mcfn_cnfw_rand[i,:],98, interpolation='nearest')
    mcfn_ctoa_min[i] = np.percentile(mcfn_ctoa_rand[i,:],2, interpolation='nearest')
    mcfn_ctoa_max[i] = np.percentile(mcfn_ctoa_rand[i,:],98, interpolation='nearest')
    mcfn_spin_min[i] = np.percentile(mcfn_spin_rand[i,:],2, interpolation='nearest')
    mcfn_spin_max[i] = np.percentile(mcfn_spin_rand[i,:],98, interpolation='nearest')


# In[ ]:

plt.semilogx(10**binmids, mcfn_vratio, 'r-')
plt.fill_between(10**binmids, mcfn_vratio_min, mcfn_vratio_max, facecolor='red', alpha=0.3)


# In[ ]:

plt.semilogx(10**binmids, mcfn_cnfw, 'r-')
plt.fill_between(10**binmids, mcfn_cnfw_min, mcfn_cnfw_max, facecolor='red', alpha=0.3)


# In[ ]:

plt.semilogx(10**binmids, mcfn_ctoa, 'r-')
plt.fill_between(10**binmids, mcfn_ctoa_min, mcfn_ctoa_max, facecolor='red', alpha=0.3)


# In[ ]:

plt.semilogx(10**binmids, mcfn_spin, 'r-')
plt.fill_between(10**binmids, mcfn_spin_min, mcfn_spin_max, facecolor='red', alpha=0.3)


# In[ ]:

cnfw_sort = np.sort(hosts_data, order='halo_cnfw')
print len(cnfw_sort)


# In[ ]:

lowlim=int(np.floor(.2*len(cnfw_sort)))
highlim=int(np.floor(.2*len(cnfw_sort)))
low_cnfw_sort = cnfw_sort[0:lowlim]
high_cnfw_sort = cnfw_sort[highlim:-1]
lowpos = np.vstack((low_cnfw_sort['halo_x'], low_cnfw_sort['halo_y'], low_cnfw_sort['halo_z'])).T
highpos = np.vstack((high_cnfw_sort['halo_x'], high_cnfw_sort['halo_y'], high_cnfw_sort['halo_z'])).T
x_rand = lbox * np.random.random(len(lowpos)*3)
y_rand = lbox * np.random.random(len(lowpos)*3)
z_rand = lbox * np.random.random(len(lowpos)*3)
randpos = np.vstack((x_rand,y_rand,z_rand)).T
Nsub = np.array([2,2,2])
xi_low, cov_low = mo.tpcf_jackknife(lowpos, randoms=randpos, rbins=10**logbins, Nsub=Nsub, period=lbox)
xi_high, cov_high = mo.tpcf_jackknife(highpos, randoms=randpos, rbins=10**logbins, Nsub=Nsub, period=lbox)
error_low = np.sqrt(1./np.diag(cov_low))
error_high = np.sqrt(1./np.diag(cov_high))
yerr_low_low = np.maximum(1e-2, xi_low - error_low)
yerr_high_low = xi_low + error_low
yerr_low_high = np.maximum(1e-2, xi_high - error_high)
yerr_high_high = xi_high + error_high

ax = plt.subplot(111)
plt.errorbar(10**binmids, xi_low, yerr=[yerr_low_low, yerr_high_low], fmt='b.')
plt.errorbar(10**binmids, xi_high, yerr=[yerr_low_high, yerr_high_high], fmt='r.')
ax.set_xscale('log')
ax.set_yscale('log')


# In[ ]:

np.savetxt('l0125_m1e11_d50b.dat', np.transpose([10**binmids, xi, xi_low, xi_high, mcfn_vratio, mcfn_vratio_min, 
                                                  mcfn_vratio_max, mcfn_cnfw, mcfn_cnfw_min, mcfn_cnfw_max, mcfn_ctoa,
                                                  mcfn_ctoa_min, mcfn_ctoa_max, mcfn_spin, mcfn_spin_min, mcfn_spin_max]), 
header='r xi xi_low xi_high mcfn_vratio +low +high mcfn_cnfw +low +high mcf_ctoa +low +high mcf_spin +low +high')
