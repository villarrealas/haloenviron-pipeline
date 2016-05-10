# required imports for the function here
import numpy as np
from numpy.lib.recfunctions import append_fields
import matplotlib.pyplot as plt
import scipy.stats as stats
import halotools.mock_observables as mo
import halotools.sim_manager as sm
import astropy.coordinates as coord

lbox=125       # size of simulation box
delta=10      # overdensity parameter
mthresh= 1e12  # mass threshold
matchdelta=10  # best fit delta to remove environmental effects
matchthresh= 1e12 # best fit delta mass cut

# reading in the best match file to serve as the master catalog for creating a matched catalog that will
# be analyzed down the road.
fname_match = './l0'+str(lbox)+'_d'+str(matchdelta)+'b.catalog'
outname_match = './l0'+str(lbox)+'_m'+str(mthresh).replace("+","") + '_d'+str(delta)+'b_nolog_match.dat'
gnewton = 4.302e-6   # for quick conversion
vhost_min = 240.0    # minimum vmax for host in satellite counting
vrat_frac = 0.3      # minimum vsub/vhost for satellite counting
vsub_min = vhost_min * vrat_frac # resulting minimum vsub for catalog pruning

# file name settings based on above
fname = './l0'+str(lbox)+'_d'+str(delta)+'b.catalog'
outname = './l0'+str(lbox)+'_m'+str(mthresh).replace("+","") +'_d'+str(delta)+'b_nolog.dat'
nrand = 200          # number of randomizations for error bars
nstep = 10           # number of bins for correlation functions
Nsub = np.array([2,2,2]) # number of octants for jackknife errors

# define dict pointing to all marks of interest
rs_dict = {'halo_id':(0,'i8'), 'halo_mass':(2,'f8'), 'halo_vmax':(3,'f8'), 'halo_rvir':(5,'f8'),
           'halo_rs':(6,'f8'), 'halo_x':(8,'f8'), 'halo_y':(9,'f8'), 'halo_z':(10,'f8'),
           'halo_spin':(17,'f8'), 'halo_ctoa':(28, 'f8'), 'halo_pid':(33,'i8')}

reader = sm.TabularAsciiReader(fname_match, rs_dict, row_cut_min_dict={'halo_mass':matchthresh},
				row_cut_eq_dict={'halo_pid':-1})
hosts_data_master = reader.read_ascii()

print 'Matched Catalog read: ', len(hosts_data_master)

reader = sm.TabularAsciiReader(fname, rs_dict, row_cut_min_dict={'halo_mass':mthresh},
                               row_cut_eq_dict={'halo_pid':-1})
hosts_data = reader.read_ascii()

print 'Hosts read: ', len(hosts_data)

reader = sm.TabularAsciiReader(fname, rs_dict, row_cut_min_dict={'halo_vmax':vsub_min}, row_cut_neq_dict={'halo_pid':-1})
subs_data = reader.read_ascii()

print 'Subs read: ', len(subs_data)

# take a moment to calculate what the matched catalog contains
c_master = coord.SkyCoord(x=hosts_data_master['halo_x'], y=hosts_data_master['halo_y'], z=hosts_data_master['halo_z'],
                         unit='Mpc', frame='icrs', representation='cartesian')
c_matching = coord.SkyCoord(x=hosts_data['halo_x'], y=hosts_data['halo_y'], z=hosts_data['halo_z'],
                            unit='Mpc', frame='icrs', representation='cartesian')

_, _, sep3d = coord.match_coordinates_3d(c_matching, c_master)
mask = (sep3d.value <= hosts_data['halo_rvir']*.001*.1)

print 'Matches found: ', np.sum(mask)

# let the calculation be done to add additional marks to the main catalog (and satellite number) and then
# make a cut with only the matched galaxies. We won't do satellite numbers because it will likely
# not work terribly well due to statistics.

# add calculated marks
vratio_temp = hosts_data['halo_vmax']/(np.sqrt(gnewton * hosts_data['halo_mass']/hosts_data['halo_rvir']))
cnfw_temp = hosts_data['halo_rvir']/hosts_data['halo_rs']

# now we want to go host halo by host halo, match pid to subhalos, and
# count the number that meet our requirements and then add this on
# as both a number and a flag.
nflagtemp = np.zeros(len(hosts_data))
subcount = np.zeros(len(hosts_data))

for i in range(0, len(hosts_data)):
   if hosts_data[i]['halo_vmax'] > vhost_min:
      nflagtemp[i] = 1
      subs_cut = subs_data[np.where(subs_data['halo_pid'] == hosts_data[i]['halo_id'])]
      for j in range(0, len(subs_cut)):
         xdist = subs_cut[j]['halo_x'] - hosts_data[i]['halo_x']
         ydist = subs_cut[j]['halo_y'] - hosts_data[i]['halo_y']
         zdist = subs_cut[j]['halo_z'] - hosts_data[i]['halo_z']
         totdist = (np.sqrt(xdist**2 + ydist**2 + zdist**2))*0.001
         if totdist < hosts_data[i]['halo_rvir']:
            ratio = subs_cut[j]['halo_vmax']/hosts_data[i]['halo_vmax']
            if ratio > vrat_frac:
               subcount[i] = subcount[i] + 1
         
hosts_data_alt =  append_fields(hosts_data, ('halo_cV', 'halo_cNFW', 'halo_satflag', 'halo_nsat'), (vratio_temp, cnfw_temp, nflagtemp, subcount))

# making matched catalog with which to repeat literally everything below with.
hosts_data_matched = hosts_data_alt[np.where(mask==True)]

# now that we have data, we need to mass correct all of our marks
mass_sort = np.sort(hosts_data_alt, order='halo_mass')

vratio_binned = stats.binned_statistic(np.log10(mass_sort['halo_mass']), 
                                       (mass_sort['halo_cV']),
                                       statistic='mean', bins=10)
vratio_fix = (mass_sort['halo_cV']) / vratio_binned.statistic[vratio_binned.binnumber-1]
mass_sort['halo_cV'] = vratio_fix

cnfw_binned = stats.binned_statistic(np.log10(mass_sort['halo_mass']), 
                                     (mass_sort['halo_cNFW']),
                                     statistic = 'mean', bins=10)
cnfw_fix = (mass_sort['halo_cNFW']) / cnfw_binned.statistic[cnfw_binned.binnumber-1]
mass_sort['halo_cNFW'] = cnfw_fix

shape_binned = stats.binned_statistic(np.log10(mass_sort['halo_mass']), 
                                      (mass_sort['halo_ctoa']),
                                      statistic='mean', bins=10)
shape_fix = (mass_sort['halo_ctoa']) / shape_binned.statistic[shape_binned.binnumber-1]
mass_sort['halo_ctoa'] = shape_fix

spin_binned = stats.binned_statistic(np.log10(mass_sort['halo_mass']), 
                                     (mass_sort['halo_spin']),
                                     statistic='mean', bins=10)
spin_fix = (mass_sort['halo_spin']) / spin_binned.statistic[spin_binned.binnumber-1]
mass_sort['halo_spin'] = spin_fix

# now all our marks have been fixed. First let's run through the marked correlation functions
# then we can take various correlation function comparisons

pos = np.vstack((mass_sort['halo_x'], mass_sort['halo_y'], mass_sort['halo_z'])).T

minlog = np.log10(3.0)
maxlog = np.log10(20.0)
steplog = (maxlog - minlog) / (nstep-1)
#logbins = np.arange(minlog, maxlog+steplog, steplog)
logbins = np.linspace(minlog, maxlog, num=nstep+1)
binmids = np.zeros(nstep)
for i in range(0,nstep):
    binmids[i] = (logbins[i]+logbins[i+1])/2.

x_rand = lbox * np.random.random(len(pos)*6)
y_rand = lbox * np.random.random(len(pos)*6)
z_rand = lbox * np.random.random(len(pos)*6)
randpos = np.vstack((x_rand,y_rand,z_rand)).T

xi, cov = mo.tpcf_jackknife(pos, randpos, rbins=10**logbins, Nsub=Nsub, period=lbox, num_threads='max')
error = np.sqrt(1./np.diag(cov))

mcf_vratio = mo.marked_tpcf(pos, 10**logbins, marks1=mass_sort['halo_cV'], period=lbox,
                            normalize_by='number_counts', wfunc=1, num_threads='max')
mcfn_vratio = (mcf_vratio - np.mean(mass_sort['halo_cV'])**2)/(np.var(mass_sort['halo_cV']))

mcf_cnfw = mo.marked_tpcf(pos, 10**logbins, marks1=mass_sort['halo_cNFW'], period=lbox,
                          normalize_by='number_counts', wfunc=1, num_threads='max')
mcfn_cnfw = (mcf_cnfw - np.mean(mass_sort['halo_cNFW'])**2)/(np.var(mass_sort['halo_cNFW']))

mcf_ctoa = mo.marked_tpcf(pos, 10**logbins, marks1=mass_sort['halo_ctoa'], period=lbox,
                          normalize_by='number_counts', wfunc=1, num_threads='max')
mcfn_ctoa = (mcf_ctoa - np.mean(mass_sort['halo_ctoa'])**2)/(np.var(mass_sort['halo_ctoa']))

mcf_spin = mo.marked_tpcf(pos, 10**logbins, marks1=mass_sort['halo_spin'], period=lbox,
                          normalize_by='number_counts', wfunc=1, num_threads='max')
mcfn_spin = (mcf_spin - np.mean(mass_sort['halo_spin'])**2)/(np.var(mass_sort['halo_spin']))

mass_sort_satflag = mass_sort[np.where(mass_sort['halo_satflag']==1)]

mcf_nsat = mo.marked_tpcf(pos[np.where(mass_sort['halo_satflag']==1)], 10**logbins, marks1=mass_sort_satflag['halo_nsat'], period=lbox, normalize_by='number_counts', wfunc=1, num_threads='max')

mcfn_nsat = (mcf_nsat - np.mean(mass_sort_satflag['halo_nsat'])**2)/(np.var(mass_sort_satflag['halo_nsat']))

# how we'll need to shuffle the marks N times, run the calculation N times,
# and determine the min and max range of the mark calculation. So:
mcf_vratio_rand = np.zeros((nstep, nrand))
mcfn_vratio_rand = np.zeros((nstep, nrand))
mcf_cnfw_rand = np.zeros((nstep, nrand))
mcfn_cnfw_rand = np.zeros((nstep, nrand))
mcf_ctoa_rand = np.zeros((nstep, nrand))
mcfn_ctoa_rand = np.zeros((nstep, nrand))
mcf_spin_rand = np.zeros((nstep, nrand))
mcfn_spin_rand = np.zeros((nstep, nrand))
mcf_nsat_rand = np.zeros((nstep, nrand))
mcfn_nsat_rand = np.zeros((nstep, nrand))

for i in range(0, nrand):
    randm_vratio = np.random.permutation(mass_sort['halo_cV'])
    randm_cnfw = np.random.permutation(mass_sort['halo_cNFW'])
    randm_ctoa = np.random.permutation(mass_sort['halo_ctoa'])
    randm_spin = np.random.permutation(mass_sort['halo_spin'])
    randm_nsat = np.random.permutation(mass_sort_satflag['halo_nsat'])

    mcf_vratio_rand[:,i] = mo.marked_tpcf(pos, 10**logbins, marks1=randm_vratio, period=lbox,
                                      normalize_by='number_counts', wfunc=1, num_threads='max')
    mcfn_vratio_rand[:,i] = (mcf_vratio_rand[:,i] - np.mean(randm_vratio)**2)/(np.var(randm_vratio))
    mcf_cnfw_rand[:,i] = mo.marked_tpcf(pos, 10**logbins, marks1=randm_cnfw, period=lbox,
                                        normalize_by='number_counts', wfunc=1, num_threads='max')
    mcfn_cnfw_rand[:,i] = (mcf_cnfw_rand[:,i] - np.mean(randm_cnfw)**2)/(np.var(randm_cnfw))
    mcf_ctoa_rand[:,i] = mo.marked_tpcf(pos, 10**logbins, marks1=randm_ctoa, period=lbox,
                                        normalize_by='number_counts', wfunc=1, num_threads='max')
    mcfn_ctoa_rand[:,i] = (mcf_ctoa_rand[:,i] - np.mean(randm_ctoa)**2)/(np.var(randm_ctoa))
    mcf_spin_rand[:,i] = mo.marked_tpcf(pos, 10**logbins, marks1=randm_spin, period=lbox,
                                        normalize_by='number_counts', wfunc=1, num_threads='max')
    mcfn_spin_rand[:,i] = (mcf_spin_rand[:,i] - np.mean(randm_spin)**2)/(np.var(randm_spin))
    mcf_nsat_rand[:,i] = mo.marked_tpcf(pos[np.where(mass_sort['halo_satflag']==1)], 10**logbins, marks1=randm_nsat, period=lbox, normalize_by='number_counts', wfunc=1, num_threads='max')
    mcfn_nsat_rand[:,i] = (mcf_nsat_rand[:,i] - np.mean(randm_nsat)**2)/(np.var(randm_nsat))

mcfn_vratio_min = np.zeros(nstep)
mcfn_vratio_max = np.zeros(nstep)
mcfn_cnfw_min = np.zeros(nstep)
mcfn_cnfw_max = np.zeros(nstep)
mcfn_ctoa_min = np.zeros(nstep)
mcfn_ctoa_max = np.zeros(nstep)
mcfn_spin_min = np.zeros(nstep)
mcfn_spin_max = np.zeros(nstep)
mcfn_nsat_min = np.zeros(nstep)
mcfn_nsat_max = np.zeros(nstep)

for i in range(0, nstep):
    mcfn_vratio_min[i] = np.percentile(mcfn_vratio_rand[i,:],2, interpolation='nearest')
    mcfn_vratio_max[i] = np.percentile(mcfn_vratio_rand[i,:],98,interpolation='nearest')
    mcfn_cnfw_min[i] = np.percentile(mcfn_cnfw_rand[i,:],2, interpolation='nearest')
    mcfn_cnfw_max[i] = np.percentile(mcfn_cnfw_rand[i,:],98, interpolation='nearest')
    mcfn_ctoa_min[i] = np.percentile(mcfn_ctoa_rand[i,:],2, interpolation='nearest')
    mcfn_ctoa_max[i] = np.percentile(mcfn_ctoa_rand[i,:],98, interpolation='nearest')
    mcfn_spin_min[i] = np.percentile(mcfn_spin_rand[i,:],2, interpolation='nearest')
    mcfn_spin_max[i] = np.percentile(mcfn_spin_rand[i,:],98, interpolation='nearest')
    mcfn_nsat_min[i] = np.percentile(mcfn_nsat_rand[i,:],2, interpolation='nearest')
    mcfn_nsat_max[i] = np.percentile(mcfn_nsat_rand[i,:],98, interpolation='nearest')

cnfw_sort = np.sort(mass_sort, order='halo_cNFW')
lowlim=int(np.floor(.2*len(cnfw_sort)))
highlim=int(np.floor(.8*len(cnfw_sort)))
low_cnfw_sort = cnfw_sort[0:lowlim]
high_cnfw_sort = cnfw_sort[highlim:-1]
lowpos = np.vstack((low_cnfw_sort['halo_x'], low_cnfw_sort['halo_y'], low_cnfw_sort['halo_z'])).T
highpos = np.vstack((high_cnfw_sort['halo_x'], high_cnfw_sort['halo_y'], high_cnfw_sort['halo_z'])).T
x_rand = lbox * np.random.random(len(lowpos)*6)
y_rand = lbox * np.random.random(len(lowpos)*6)
z_rand = lbox * np.random.random(len(lowpos)*6)
randpos = np.vstack((x_rand,y_rand,z_rand)).T
xi_cnfw_low, cov_cnfw_low = mo.tpcf_jackknife(lowpos, randoms=randpos, rbins=10**logbins, Nsub=Nsub, period=lbox, num_threads='max')
xi_cnfw_high, cov_cnfw_high = mo.tpcf_jackknife(highpos, randoms=randpos, rbins=10**logbins, Nsub=Nsub, period=lbox, num_threads='max')
error_cnfw_low = np.sqrt(1./np.diag(cov_cnfw_low))
error_cnfw_high = np.sqrt(1./np.diag(cov_cnfw_high))

spin_sort = np.sort(mass_sort, order='halo_spin')
low_spin_sort = spin_sort[0:lowlim]
high_spin_sort = spin_sort[highlim:-1]
lowpos = np.vstack((low_spin_sort['halo_x'], low_spin_sort['halo_y'], low_spin_sort['halo_z'])).T
highpos = np.vstack((high_spin_sort['halo_x'], high_spin_sort['halo_y'], high_spin_sort['halo_z'])).T
x_rand = lbox * np.random.random(len(lowpos)*6)
y_rand = lbox * np.random.random(len(lowpos)*6)
z_rand = lbox * np.random.random(len(lowpos)*6)
randpos = np.vstack((x_rand,y_rand,z_rand)).T
xi_spin_low, cov_spin_low = mo.tpcf_jackknife(lowpos, randoms=randpos, rbins=10**logbins, Nsub=Nsub, period=lbox, num_threads='max')
xi_spin_high, cov_spin_high = mo.tpcf_jackknife(highpos, randoms=randpos, rbins=10**logbins, Nsub=Nsub, period=lbox, num_threads='max')
error_spin_low = np.sqrt(1./np.diag(cov_spin_low))
error_spin_high = np.sqrt(1./np.diag(cov_spin_high))

ctoa_sort = np.sort(mass_sort, order='halo_ctoa')
low_ctoa_sort = ctoa_sort[0:lowlim]
high_ctoa_sort = ctoa_sort[highlim:-1]
lowpos = np.vstack((low_ctoa_sort['halo_x'], low_ctoa_sort['halo_y'], low_ctoa_sort['halo_z'])).T
highpos = np.vstack((high_ctoa_sort['halo_x'], high_ctoa_sort['halo_y'], high_ctoa_sort['halo_z'])).T
x_rand = lbox * np.random.random(len(lowpos)*6)
y_rand = lbox * np.random.random(len(lowpos)*6)
z_rand = lbox * np.random.random(len(lowpos)*6)
randpos = np.vstack((x_rand,y_rand,z_rand)).T
xi_ctoa_low, cov_ctoa_low = mo.tpcf_jackknife(lowpos, randoms=randpos, rbins=10**logbins, Nsub=Nsub, period=lbox, num_threads='max')
xi_ctoa_high, cov_ctoa_high = mo.tpcf_jackknife(highpos, randoms=randpos, rbins=10**logbins, Nsub=Nsub, period=lbox, num_threads='max')
error_ctoa_low = np.sqrt(1./np.diag(cov_ctoa_low))
error_ctoa_high = np.sqrt(1./np.diag(cov_ctoa_high))

np.savetxt(outname, np.transpose([10**binmids, xi, error, xi_cnfw_low, error_cnfw_low, xi_cnfw_high, error_cnfw_high, xi_ctoa_low, error_ctoa_low, xi_ctoa_high, error_ctoa_high, xi_spin_low, error_spin_low, xi_spin_high, error_spin_high, mcfn_vratio, mcfn_vratio_min, 
                                                  mcfn_vratio_max, mcfn_cnfw, mcfn_cnfw_min, mcfn_cnfw_max, mcfn_ctoa,
                                                  mcfn_ctoa_min, mcfn_ctoa_max, mcfn_spin, mcfn_spin_min, mcfn_spin_max, mcfn_nsat, mcfn_nsat_min, mcfn_nsat_max]), 
header='r xi +err xi_cnfw_low +err xi_cnfw_high +err xi_ctoa_low +err xi_ctoa_high +err xi_spin_low +err xi_spin_high + err mcfn_vratio +low +high mcfn_cnfw +low +high mcf_ctoa +low +high mcf_spin +low +high mcf_nsat +low +high')

print "Main Run Complete -> Beginning Matched Catalog Analysis"

mass_sort = np.sort(hosts_data_matched, order='halo_mass')

vratio_binned = stats.binned_statistic(np.log10(mass_sort['halo_mass']), 
                                       (mass_sort['halo_cV']),
                                       statistic='mean', bins=10)
vratio_fix = (mass_sort['halo_cV']) / vratio_binned.statistic[vratio_binned.binnumber-1]
mass_sort['halo_cV'] = vratio_fix

cnfw_binned = stats.binned_statistic(np.log10(mass_sort['halo_mass']), 
                                     (mass_sort['halo_cNFW']),
                                     statistic = 'mean', bins=10)
cnfw_fix = (mass_sort['halo_cNFW']) / cnfw_binned.statistic[cnfw_binned.binnumber-1]
mass_sort['halo_cNFW'] = cnfw_fix

shape_binned = stats.binned_statistic(np.log10(mass_sort['halo_mass']), 
                                      (mass_sort['halo_ctoa']),
                                      statistic='mean', bins=10)
shape_fix = (mass_sort['halo_ctoa']) / shape_binned.statistic[shape_binned.binnumber-1]
mass_sort['halo_ctoa'] = shape_fix

spin_binned = stats.binned_statistic(np.log10(mass_sort['halo_mass']), 
                                     (mass_sort['halo_spin']),
                                     statistic='mean', bins=10)
spin_fix = (mass_sort['halo_spin']) / spin_binned.statistic[spin_binned.binnumber-1]
mass_sort['halo_spin'] = spin_fix

# now all our marks have been fixed. First let's run through the marked correlation functions
# then we can take various correlation function comparisons

pos = np.vstack((mass_sort['halo_x'], mass_sort['halo_y'], mass_sort['halo_z'])).T

minlog = np.log10(3.0)
maxlog = np.log10(20.0)
steplog = (maxlog - minlog) / (nstep-1)
#logbins = np.arange(minlog, maxlog+steplog, steplog)
logbins = np.linspace(minlog, maxlog, num=nstep+1)
binmids = np.zeros(nstep)
for i in range(0,nstep):
    binmids[i] = (logbins[i]+logbins[i+1])/2.

x_rand = lbox * np.random.random(len(pos)*6)
y_rand = lbox * np.random.random(len(pos)*6)
z_rand = lbox * np.random.random(len(pos)*6)
randpos = np.vstack((x_rand,y_rand,z_rand)).T

xi, cov = mo.tpcf_jackknife(pos, randpos, rbins=10**logbins, Nsub=Nsub, period=lbox, num_threads='max')
error = np.sqrt(1./np.diag(cov))

mcf_vratio = mo.marked_tpcf(pos, 10**logbins, marks1=mass_sort['halo_cV'], period=lbox,
                            normalize_by='number_counts', wfunc=1, num_threads='max')
mcfn_vratio = (mcf_vratio - np.mean(mass_sort['halo_cV'])**2)/(np.var(mass_sort['halo_cV']))

mcf_cnfw = mo.marked_tpcf(pos, 10**logbins, marks1=mass_sort['halo_cNFW'], period=lbox,
                          normalize_by='number_counts', wfunc=1, num_threads='max')
mcfn_cnfw = (mcf_cnfw - np.mean(mass_sort['halo_cNFW'])**2)/(np.var(mass_sort['halo_cNFW']))

mcf_ctoa = mo.marked_tpcf(pos, 10**logbins, marks1=mass_sort['halo_ctoa'], period=lbox,
                          normalize_by='number_counts', wfunc=1, num_threads='max')
mcfn_ctoa = (mcf_ctoa - np.mean(mass_sort['halo_ctoa'])**2)/(np.var(mass_sort['halo_ctoa']))

mcf_spin = mo.marked_tpcf(pos, 10**logbins, marks1=mass_sort['halo_spin'], period=lbox,
                          normalize_by='number_counts', wfunc=1, num_threads='max')
mcfn_spin = (mcf_spin - np.mean(mass_sort['halo_spin'])**2)/(np.var(mass_sort['halo_spin']))

mass_sort_satflag = mass_sort[np.where(mass_sort['halo_satflag']==1)]

mcf_nsat = mo.marked_tpcf(pos[np.where(mass_sort['halo_satflag']==1)], 10**logbins, marks1=mass_sort_satflag['halo_nsat'], period=lbox, normalize_by='number_counts', wfunc=1, num_threads='max')

mcfn_nsat = (mcf_nsat - np.mean(mass_sort_satflag['halo_nsat'])**2)/(np.var(mass_sort_satflag['halo_nsat']))

# how we'll need to shuffle the marks N times, run the calculation N times,
# and determine the min and max range of the mark calculation. So:
mcf_vratio_rand = np.zeros((nstep, nrand))
mcfn_vratio_rand = np.zeros((nstep, nrand))
mcf_cnfw_rand = np.zeros((nstep, nrand))
mcfn_cnfw_rand = np.zeros((nstep, nrand))
mcf_ctoa_rand = np.zeros((nstep, nrand))
mcfn_ctoa_rand = np.zeros((nstep, nrand))
mcf_spin_rand = np.zeros((nstep, nrand))
mcfn_spin_rand = np.zeros((nstep, nrand))
mcf_nsat_rand = np.zeros((nstep, nrand))
mcfn_nsat_rand = np.zeros((nstep, nrand))

for i in range(0, nrand):
    randm_vratio = np.random.permutation(mass_sort['halo_cV'])
    randm_cnfw = np.random.permutation(mass_sort['halo_cNFW'])
    randm_ctoa = np.random.permutation(mass_sort['halo_ctoa'])
    randm_spin = np.random.permutation(mass_sort['halo_spin'])
    randm_nsat = np.random.permutation(mass_sort_satflag['halo_nsat'])

    mcf_vratio_rand[:,i] = mo.marked_tpcf(pos, 10**logbins, marks1=randm_vratio, period=lbox,
                                      normalize_by='number_counts', wfunc=1, num_threads='max')
    mcfn_vratio_rand[:,i] = (mcf_vratio_rand[:,i] - np.mean(randm_vratio)**2)/(np.var(randm_vratio))
    mcf_cnfw_rand[:,i] = mo.marked_tpcf(pos, 10**logbins, marks1=randm_cnfw, period=lbox,
                                        normalize_by='number_counts', wfunc=1, num_threads='max')
    mcfn_cnfw_rand[:,i] = (mcf_cnfw_rand[:,i] - np.mean(randm_cnfw)**2)/(np.var(randm_cnfw))
    mcf_ctoa_rand[:,i] = mo.marked_tpcf(pos, 10**logbins, marks1=randm_ctoa, period=lbox,
                                        normalize_by='number_counts', wfunc=1, num_threads='max')
    mcfn_ctoa_rand[:,i] = (mcf_ctoa_rand[:,i] - np.mean(randm_ctoa)**2)/(np.var(randm_ctoa))
    mcf_spin_rand[:,i] = mo.marked_tpcf(pos, 10**logbins, marks1=randm_spin, period=lbox,
                                        normalize_by='number_counts', wfunc=1, num_threads='max')
    mcfn_spin_rand[:,i] = (mcf_spin_rand[:,i] - np.mean(randm_spin)**2)/(np.var(randm_spin))
    mcf_nsat_rand[:,i] = mo.marked_tpcf(pos[np.where(mass_sort['halo_satflag']==1)], 10**logbins, marks1=randm_nsat, period=lbox, normalize_by='number_counts', wfunc=1, num_threads='max')
    mcfn_nsat_rand[:,i] = (mcf_nsat_rand[:,i] - np.mean(randm_nsat)**2)/(np.var(randm_nsat))

mcfn_vratio_min = np.zeros(nstep)
mcfn_vratio_max = np.zeros(nstep)
mcfn_cnfw_min = np.zeros(nstep)
mcfn_cnfw_max = np.zeros(nstep)
mcfn_ctoa_min = np.zeros(nstep)
mcfn_ctoa_max = np.zeros(nstep)
mcfn_spin_min = np.zeros(nstep)
mcfn_spin_max = np.zeros(nstep)
mcfn_nsat_min = np.zeros(nstep)
mcfn_nsat_max = np.zeros(nstep)

for i in range(0, nstep):
    mcfn_vratio_min[i] = np.percentile(mcfn_vratio_rand[i,:],2, interpolation='nearest')
    mcfn_vratio_max[i] = np.percentile(mcfn_vratio_rand[i,:],98,interpolation='nearest')
    mcfn_cnfw_min[i] = np.percentile(mcfn_cnfw_rand[i,:],2, interpolation='nearest')
    mcfn_cnfw_max[i] = np.percentile(mcfn_cnfw_rand[i,:],98, interpolation='nearest')
    mcfn_ctoa_min[i] = np.percentile(mcfn_ctoa_rand[i,:],2, interpolation='nearest')
    mcfn_ctoa_max[i] = np.percentile(mcfn_ctoa_rand[i,:],98, interpolation='nearest')
    mcfn_spin_min[i] = np.percentile(mcfn_spin_rand[i,:],2, interpolation='nearest')
    mcfn_spin_max[i] = np.percentile(mcfn_spin_rand[i,:],98, interpolation='nearest')
    mcfn_nsat_min[i] = np.percentile(mcfn_nsat_rand[i,:],2, interpolation='nearest')
    mcfn_nsat_max[i] = np.percentile(mcfn_nsat_rand[i,:],98, interpolation='nearest')

cnfw_sort = np.sort(mass_sort, order='halo_cNFW')
lowlim=int(np.floor(.2*len(cnfw_sort)))
highlim=int(np.floor(.8*len(cnfw_sort)))
low_cnfw_sort = cnfw_sort[0:lowlim]
high_cnfw_sort = cnfw_sort[highlim:-1]
lowpos = np.vstack((low_cnfw_sort['halo_x'], low_cnfw_sort['halo_y'], low_cnfw_sort['halo_z'])).T
highpos = np.vstack((high_cnfw_sort['halo_x'], high_cnfw_sort['halo_y'], high_cnfw_sort['halo_z'])).T
x_rand = lbox * np.random.random(len(lowpos)*6)
y_rand = lbox * np.random.random(len(lowpos)*6)
z_rand = lbox * np.random.random(len(lowpos)*6)
randpos = np.vstack((x_rand,y_rand,z_rand)).T
xi_cnfw_low, cov_cnfw_low = mo.tpcf_jackknife(lowpos, randoms=randpos, rbins=10**logbins, Nsub=Nsub, period=lbox, num_threads='max')
xi_cnfw_high, cov_cnfw_high = mo.tpcf_jackknife(highpos, randoms=randpos, rbins=10**logbins, Nsub=Nsub, period=lbox, num_threads='max')
error_cnfw_low = np.sqrt(1./np.diag(cov_cnfw_low))
error_cnfw_high = np.sqrt(1./np.diag(cov_cnfw_high))

spin_sort = np.sort(mass_sort, order='halo_spin')
low_spin_sort = spin_sort[0:lowlim]
high_spin_sort = spin_sort[highlim:-1]
lowpos = np.vstack((low_spin_sort['halo_x'], low_spin_sort['halo_y'], low_spin_sort['halo_z'])).T
highpos = np.vstack((high_spin_sort['halo_x'], high_spin_sort['halo_y'], high_spin_sort['halo_z'])).T
x_rand = lbox * np.random.random(len(lowpos)*6)
y_rand = lbox * np.random.random(len(lowpos)*6)
z_rand = lbox * np.random.random(len(lowpos)*6)
randpos = np.vstack((x_rand,y_rand,z_rand)).T
xi_spin_low, cov_spin_low = mo.tpcf_jackknife(lowpos, randoms=randpos, rbins=10**logbins, Nsub=Nsub, period=lbox, num_threads='max')
xi_spin_high, cov_spin_high = mo.tpcf_jackknife(highpos, randoms=randpos, rbins=10**logbins, Nsub=Nsub, period=lbox, num_threads='max')
error_spin_low = np.sqrt(1./np.diag(cov_spin_low))
error_spin_high = np.sqrt(1./np.diag(cov_spin_high))

ctoa_sort = np.sort(mass_sort, order='halo_ctoa')
low_ctoa_sort = ctoa_sort[0:lowlim]
high_ctoa_sort = ctoa_sort[highlim:-1]
lowpos = np.vstack((low_ctoa_sort['halo_x'], low_ctoa_sort['halo_y'], low_ctoa_sort['halo_z'])).T
highpos = np.vstack((high_ctoa_sort['halo_x'], high_ctoa_sort['halo_y'], high_ctoa_sort['halo_z'])).T
x_rand = lbox * np.random.random(len(lowpos)*6)
y_rand = lbox * np.random.random(len(lowpos)*6)
z_rand = lbox * np.random.random(len(lowpos)*6)
randpos = np.vstack((x_rand,y_rand,z_rand)).T
xi_ctoa_low, cov_ctoa_low = mo.tpcf_jackknife(lowpos, randoms=randpos, rbins=10**logbins, Nsub=Nsub, period=lbox, num_threads='max')
xi_ctoa_high, cov_ctoa_high = mo.tpcf_jackknife(highpos, randoms=randpos, rbins=10**logbins, Nsub=Nsub, period=lbox, num_threads='max')
error_ctoa_low = np.sqrt(1./np.diag(cov_ctoa_low))
error_ctoa_high = np.sqrt(1./np.diag(cov_ctoa_high))

np.savetxt(outname_match, np.transpose([10**binmids, xi, error, xi_cnfw_low, error_cnfw_low, xi_cnfw_high, error_cnfw_high, xi_ctoa_low, error_ctoa_low, xi_ctoa_high, error_ctoa_high, xi_spin_low, error_spin_low, xi_spin_high, error_spin_high, mcfn_vratio, mcfn_vratio_min, 
                                                  mcfn_vratio_max, mcfn_cnfw, mcfn_cnfw_min, mcfn_cnfw_max, mcfn_ctoa,
                                                  mcfn_ctoa_min, mcfn_ctoa_max, mcfn_spin, mcfn_spin_min, mcfn_spin_max, mcfn_nsat, mcfn_nsat_min, mcfn_nsat_max]), 
header='r xi +err xi_cnfw_low +err xi_cnfw_high +err xi_ctoa_low +err xi_ctoa_high +err xi_spin_low +err xi_spin_high + err mcfn_vratio +low +high mcfn_cnfw +low +high mcf_ctoa +low +high mcf_spin +low +high mcf_nsat +low +high')

