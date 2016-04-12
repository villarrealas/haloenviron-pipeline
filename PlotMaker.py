import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats

plt.ioff()

# r xi xi_low xi_high mcfn_vratio +low +high mcfn_cnfw +low +high mcf_ctoa +low +high mcf_spin +low +high / layout of files

l0500_d200= np.loadtxt('l0500_m4e12_d200b_nolog.dat')
l0500_d100= np.loadtxt('l0500_m5e12_d100b_nolog.dat')
l0500_d75=np.loadtxt('l0500_m6e12_d75b_nolog.dat')
l0500_d50=np.loadtxt('l0500_m7e12_d50b_nolog.dat')

l0500_d200_xicomp = (l0500_d200[:,3]-l0500_d200[:,2])/l0500_d200[:,1]
l0500_d100_xicomp = (l0500_d100[:,3]-l0500_d100[:,2])/l0500_d100[:,1]
l0500_d75_xicomp = (l0500_d75[:,3]-l0500_d75[:,2])/l0500_d75[:,1]
l0500_d50_xicomp = (l0500_d50[:,3]-l0500_d50[:,2])/l0500_d50[:,1]

l0250_d200= np.loadtxt('l0250_m7e11_d200b_nolog.dat')
l0250_d100= np.loadtxt('l0250_m8e11_d100b_nolog.dat')
l0250_d75=np.loadtxt('l0250_m9e11_d75b_nolog.dat')
l0250_d50=np.loadtxt('l0250_m1.5e12_d50b_nolog.dat')

l0250_d200_xicomp = (l0250_d200[:,3]-l0250_d200[:,2])/l0250_d200[:,1]
l0250_d100_xicomp = (l0250_d100[:,3]-l0250_d100[:,2])/l0250_d100[:,1]
l0250_d75_xicomp = (l0250_d75[:,3]-l0250_d75[:,2])/l0250_d75[:,1]
l0250_d50_xicomp = (l0250_d50[:,3]-l0250_d50[:,2])/l0250_d50[:,1]

l0125_d200 = np.loadtxt('l0125_m7e10_d200b_nolog.dat')
l0125_d100 = np.loadtxt('l0125_m8e10_d100b_nolog.dat')
l0125_d75 = np.loadtxt('l0125_m9e10_d75b_nolog.dat')
l0125_d50 = np.loadtxt('l0125_m1e11_d50b_nolog.dat')

l0125_d200_xicomp = (l0125_d200[:,3]-l0125_d200[:,2])/l0125_d200[:,1]
l0125_d100_xicomp = (l0125_d100[:,3]-l0125_d100[:,2])/l0125_d100[:,1]
l0125_d75_xicomp = (l0125_d75[:,3]-l0125_d75[:,2])/l0125_d75[:,1]
l0125_d50_xicomp = (l0125_d50[:,3]-l0125_d50[:,2])/l0125_d50[:,1]

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

plt.semilogx(l0500_d200[:,0], l0500_d200_xicomp, 'b-', label=r'$\Delta = 200$')
plt.plot(l0500_d100[:,0], l0500_d100_xicomp, 'r-', label=r'$\Delta = 100$')
plt.plot(l0500_d75[:,0], l0500_d75_xicomp, 'g-', label=r'$\Delta = 75$')
plt.plot(l0500_d50[:,0], l0500_d50_xicomp, 'c-', label=r'$\Delta = 50$')
plt.xlim(4,18)
#plt.ylim(-1,0.02)
plt.xlabel('$r \ (h^{-1}\mathrm{Mpc})$')
plt.ylabel(r'$(\xi_{\mathrm{high}} - \xi_{\mathrm{low}} ) / \xi_{\mathrm{all}}$')
plt.title('L0500')
plt.legend(loc='lower right', numpoints=1, prop={'size': 10})
plt.tight_layout()
plt.savefig('./FIGS/l0500_cfcompare.pdf')
plt.clf()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

plt.semilogx(l0250_d200[:,0], l0250_d200_xicomp, 'b-', label=r'$\Delta = 200$')
plt.plot(l0250_d100[:,0], l0250_d100_xicomp, 'r-', label=r'$\Delta = 100$')
plt.plot(l0250_d75[:,0], l0250_d75_xicomp, 'g-', label=r'$\Delta = 75$')
plt.plot(l0250_d50[:,0], l0250_d50_xicomp, 'c-', label=r'$\Delta = 50$')
plt.xlim(4,18)
#plt.ylim(-1,0.02)
plt.xlabel('$r \ (h^{-1}\mathrm{Mpc})$')
plt.ylabel(r'$(\xi_{\mathrm{high}} - \xi_{\mathrm{low}} ) / \xi_{\mathrm{all}}$')
plt.title('L0250')
plt.legend(loc='lower right', numpoints=1, prop={'size': 10})
plt.tight_layout()
plt.savefig('./FIGS/l0250_cfcompare.pdf')
plt.clf()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

plt.semilogx(l0125_d200[:,0], l0125_d200_xicomp, 'b-', label=r'$\Delta = 200$')
plt.plot(l0125_d100[:,0], l0125_d100_xicomp, 'r-', label=r'$\Delta = 100$')
plt.plot(l0125_d75[:,0], l0125_d75_xicomp, 'g-', label=r'$\Delta = 75$')
plt.plot(l0125_d50[:,0], l0125_d50_xicomp, 'c-', label=r'$\Delta = 50$')
plt.xlim(4,18)
#plt.ylim(-0.3,0.4)
plt.xlabel('$r \ (h^{-1}\mathrm{Mpc})$')
plt.ylabel(r'$(\xi_{\mathrm{high}} - \xi_{\mathrm{low}} ) / \xi_{\mathrm{all}}$')
plt.title('L0125')
plt.legend(loc='lower right', numpoints=1, prop={'size': 10})
plt.tight_layout()
plt.savefig('./FIGS/l0125_cfcompare.pdf')
plt.clf()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}


matplotlib.rc('font', **font)

plt.axhline(y=0, color='k', linestyle='--')
plt.fill_between(l0500_d200[:,0], l0500_d200[:,5], l0500_d200[:,6], facecolor='b', edgecolor='k', alpha=0.1)
plt.fill_between(l0500_d100[:,0], l0500_d100[:,5], l0500_d100[:,6], facecolor='r', edgecolor='k', alpha=0.1)
plt.fill_between(l0500_d75[:,0], l0500_d75[:,5], l0500_d75[:,6], facecolor='c', edgecolor='k', alpha=0.1)
plt.fill_between(l0500_d50[:,0], l0500_d50[:,5], l0500_d50[:,6], facecolor='g', edgecolor='k', alpha=0.1)

plt.semilogx(l0500_d200[:,0], l0500_d200[:,4], 'b-', 
             label='$\Delta = 200, M_{\odot}h^{-1} \geq 4e12$')
plt.plot(l0500_d100[:,0], l0500_d100[:,4], 'r-', 
             label='$\Delta = 100, M_{\odot}h^{-1} \geq 5e12$')
plt.plot(l0500_d75[:,0], l0500_d75[:,4], 'c-', 
             label='$\Delta = 75, M_{\odot}h^{-1} \geq 6e12$')
plt.plot(l0500_d50[:,0], l0500_d50[:,4], 'g-', 
             label='$\Delta = 50, M_{\odot}h^{-1} \geq 7e12$')

plt.legend(loc='lower right', prop={'size': 8})

plt.xlim(4,18)
#plt.ylim(-0.03,0.04)
plt.xlabel(r'$r \ (h^{-1} \mathrm{Mpc})$')
plt.ylabel(r'$\mathcal{M}_{c_{\mathrm{V}}}$')
plt.title('L0500')
plt.tight_layout()
plt.savefig('./FIGS/l0500_mcf_cV.pdf')
plt.clf()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}


matplotlib.rc('font', **font)

plt.axhline(y=0, color='k', linestyle='--')
plt.fill_between(l0250_d200[:,0], l0250_d200[:,5], l0250_d200[:,6], facecolor='b', edgecolor='k', alpha=0.1)
plt.fill_between(l0250_d100[:,0], l0250_d100[:,5], l0250_d100[:,6], facecolor='r', edgecolor='k', alpha=0.1)
plt.fill_between(l0250_d75[:,0], l0250_d75[:,5], l0250_d75[:,6], facecolor='c', edgecolor='k', alpha=0.1)
plt.fill_between(l0250_d50[:,0], l0250_d50[:,5], l0250_d50[:,6], facecolor='g', edgecolor='k', alpha=0.1)

plt.semilogx(l0250_d200[:,0], l0250_d200[:,4], 'b-', 
             label='$\Delta = 200, M_{\odot}h^{-1} \geq 7e11$')
plt.plot(l0250_d100[:,0], l0250_d100[:,4], 'r-', 
             label='$\Delta = 100, M_{\odot}h^{-1} \geq 8e11$')
plt.plot(l0250_d75[:,0], l0250_d75[:,4], 'c-', 
             label='$\Delta = 75, M_{\odot}h^{-1} \geq 9e11$')
plt.plot(l0250_d50[:,0], l0250_d50[:,4], 'g-', 
             label='$\Delta = 50, M_{\odot}h^{-1} \geq 1.5e12$')

plt.legend(loc='lower right', prop={'size': 8})

plt.xlim(4,18)
#plt.ylim(-0.02,0.02)
plt.xlabel(r'$r \ (h^{-1} \mathrm{Mpc})$')
plt.ylabel(r'$\mathcal{M}_{c_{\mathrm{V}}}$')
plt.title('L0250')
plt.tight_layout()
plt.savefig('./FIGS/l0250_mcf_cV.pdf')
plt.clf()

matplotlib.rc('font', **font)

plt.axhline(y=0, color='k', linestyle='--')
plt.fill_between(l0125_d200[:,0], l0125_d200[:,5], l0125_d200[:,6], facecolor='b', edgecolor='k', alpha=0.1)
plt.fill_between(l0125_d100[:,0], l0125_d100[:,5], l0125_d100[:,6], facecolor='r', edgecolor='k', alpha=0.1)
plt.fill_between(l0125_d75[:,0], l0125_d75[:,5], l0125_d75[:,6], facecolor='c', edgecolor='k', alpha=0.1)
plt.fill_between(l0125_d50[:,0], l0125_d50[:,5], l0125_d50[:,6], facecolor='g', edgecolor='k', alpha=0.1)

plt.semilogx(l0125_d200[:,0], l0125_d200[:,4], 'b-', 
             label='$\Delta = 200, M_{\odot}h^{-1} \geq 7e10$')
plt.plot(l0125_d100[:,0], l0125_d100[:,4], 'r-', 
             label='$\Delta = 100, M_{\odot}h^{-1} \geq 8e10$')
plt.plot(l0125_d75[:,0], l0125_d75[:,4], 'c-', 
             label='$\Delta = 75, M_{\odot}h^{-1} \geq 9e10$')
plt.plot(l0125_d50[:,0], l0125_d50[:,4], 'g-', 
             label='$\Delta = 50, M_{\odot}h^{-1} \geq 1e11$')

plt.legend(loc='lower right', prop={'size': 8})

plt.xlim(4,18)
#plt.ylim(-0.01,0.05)
plt.xlabel(r'$r \ (h^{-1} \mathrm{Mpc})$')
plt.ylabel(r'$\mathcal{M}_{c_{\mathrm{V}}}$')
plt.title('L0125')
plt.tight_layout()
plt.savefig('./FIGS/l0125_mcf_cV.pdf')
plt.clf()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

plt.axhline(y=0, color='k', linestyle='--')
plt.fill_between(l0500_d200[:,0], l0500_d200[:,8], l0500_d200[:,9], facecolor='b', edgecolor='k', alpha=0.1)
plt.fill_between(l0500_d100[:,0], l0500_d100[:,8], l0500_d100[:,9], facecolor='r', edgecolor='k', alpha=0.1)
plt.fill_between(l0500_d75[:,0], l0500_d75[:,8], l0500_d75[:,9], facecolor='c', edgecolor='k', alpha=0.1)
plt.fill_between(l0500_d50[:,0], l0500_d50[:,8], l0500_d50[:,9], facecolor='g', edgecolor='k', alpha=0.1)

plt.semilogx(l0500_d200[:,0], l0500_d200[:,7], 'b-', 
             label='$\Delta = 200, M_{\odot}h^{-1} \geq 4e12$')
plt.plot(l0500_d100[:,0], l0500_d100[:,7], 'r-', 
             label='$\Delta = 100, M_{\odot}h^{-1} \geq 5e12$')
plt.plot(l0500_d75[:,0], l0500_d75[:,7], 'c-', 
             label='$\Delta = 75, M_{\odot}h^{-1} \geq 6e12$')
plt.plot(l0500_d50[:,0], l0500_d50[:,7], 'g-', 
             label='$\Delta = 50, M_{\odot}h^{-1} \geq 7e12$')

plt.legend(loc='lower right', prop={'size': 8})

plt.xlim(4,18)
#plt.ylim(-0.02,0.02)
plt.xlabel(r'$r \ (h^{-1} \mathrm{Mpc})$')
plt.ylabel(r'$\mathcal{M}_{c_{\mathrm{NFW}}}$')
plt.title('L0500')
plt.tight_layout()
plt.savefig('./FIGS/l0500_mcf_cNFW.pdf')
plt.clf()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

plt.axhline(y=0, color='k', linestyle='--')
plt.fill_between(l0250_d200[:,0], l0250_d200[:,8], l0250_d200[:,9], facecolor='b', edgecolor='k', alpha=0.1)
plt.fill_between(l0250_d100[:,0], l0250_d100[:,8], l0250_d100[:,9], facecolor='r', edgecolor='k', alpha=0.1)
plt.fill_between(l0250_d75[:,0], l0250_d75[:,8], l0250_d75[:,9], facecolor='c', edgecolor='k', alpha=0.1)
plt.fill_between(l0250_d50[:,0], l0250_d50[:,8], l0250_d50[:,9], facecolor='g', edgecolor='k', alpha=0.1)

plt.semilogx(l0250_d200[:,0], l0250_d200[:,7], 'b-', 
             label='$\Delta = 200, M_{\odot}h^{-1} \geq 7e11$')
plt.plot(l0250_d100[:,0], l0250_d100[:,7], 'r-', 
             label='$\Delta = 100, M_{\odot}h^{-1} \geq 8e11$')
plt.plot(l0250_d75[:,0], l0250_d75[:,7], 'c-', 
             label='$\Delta = 75, M_{\odot}h^{-1} \geq 9e11$')
plt.plot(l0250_d50[:,0], l0250_d50[:,7], 'g-', 
             label='$\Delta = 50, M_{\odot}h^{-1} \geq 1.5e12$')

plt.legend(loc='lower right', prop={'size': 8})

plt.xlim(4,18)
#plt.ylim(-0.02,0.02)
plt.xlabel(r'$r \ (h^{-1} \mathrm{Mpc})$')
plt.ylabel(r'$\mathcal{M}_{c_{\mathrm{NFW}}}$')
plt.title('L0250')
plt.tight_layout()
plt.savefig('./FIGS/l0250_mcf_cNFW.pdf')
plt.clf()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

plt.axhline(y=0, color='k', linestyle='--')
plt.fill_between(l0125_d200[:,0], l0125_d200[:,8], l0125_d200[:,9], facecolor='b', edgecolor='k', alpha=0.1)
plt.fill_between(l0125_d100[:,0], l0125_d100[:,8], l0125_d100[:,9], facecolor='r', edgecolor='k', alpha=0.1)
plt.fill_between(l0125_d75[:,0], l0125_d75[:,8], l0125_d75[:,9], facecolor='c', edgecolor='k', alpha=0.1)
plt.fill_between(l0125_d50[:,0], l0125_d50[:,8], l0125_d50[:,9], facecolor='g', edgecolor='k', alpha=0.1)

plt.semilogx(l0125_d200[:,0], l0125_d200[:,7], 'b-', 
             label='$\Delta = 200, M_{\odot}h^{-1} \geq 7e10$')
plt.plot(l0125_d100[:,0], l0125_d100[:,7], 'r-', 
             label='$\Delta = 100, M_{\odot}h^{-1} \geq 8e10$')
plt.plot(l0125_d75[:,0], l0125_d75[:,7], 'c-', 
             label='$\Delta = 75, M_{\odot}h^{-1} \geq 9e10$')
plt.plot(l0125_d50[:,0], l0125_d50[:,7], 'g-', 
             label='$\Delta = 50, M_{\odot}h^{-1} \geq 1e11$')

plt.legend(loc='lower right', prop={'size': 8})

plt.xlim(4,18)
#plt.ylim(-0.01,0.05)
plt.xlabel(r'$r \ (h^{-1} \mathrm{Mpc})$')
plt.ylabel(r'$\mathcal{M}_{c_{\mathrm{NFW}}}$')
plt.title('L0125')
plt.tight_layout()
plt.savefig('./FIGS/l0125_mcf_cNFW.pdf')
plt.clf()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

plt.axhline(y=0, color='k', linestyle='--')
plt.fill_between(l0500_d200[:,0], l0500_d200[:,11], l0500_d200[:,12], facecolor='b', edgecolor='k', alpha=0.1)
plt.fill_between(l0500_d100[:,0], l0500_d100[:,11], l0500_d100[:,12], facecolor='r', edgecolor='k', alpha=0.1)
plt.fill_between(l0500_d75[:,0], l0500_d75[:,11], l0500_d75[:,12], facecolor='c', edgecolor='k', alpha=0.1)
plt.fill_between(l0500_d50[:,0], l0500_d50[:,11], l0500_d50[:,12], facecolor='g', edgecolor='k', alpha=0.1)

plt.semilogx(l0500_d200[:,0], l0500_d200[:,10], 'b-', 
             label='$\Delta = 200, M_{\odot}h^{-1} \geq 4e12$')
plt.plot(l0500_d100[:,0], l0500_d100[:,10], 'r-', 
             label='$\Delta = 100, M_{\odot}h^{-1} \geq 5e12$')
plt.plot(l0500_d75[:,0], l0500_d75[:,10], 'c-', 
             label='$\Delta = 75, M_{\odot}h^{-1} \geq 6e12$')
plt.plot(l0500_d50[:,0], l0500_d50[:,10], 'g-', 
             label='$\Delta = 50, M_{\odot}h^{-1} \geq 7e12$')

plt.legend(loc='upper right', prop={'size': 8})

plt.xlim(4,18)
#plt.ylim(-0.02,0.15)
plt.xlabel(r'$r \ (h^{-1} \mathrm{Mpc})$')
plt.ylabel(r'$\mathcal{M}_{\mathrm{shape}}$')
plt.title('L0500')
plt.tight_layout()
plt.savefig('./FIGS/l0500_mcf_ctoa.pdf')
plt.clf()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

plt.axhline(y=0, color='k', linestyle='--')
plt.fill_between(l0250_d200[:,0], l0250_d200[:,11], l0250_d200[:,12], facecolor='b', edgecolor='k', alpha=0.1)
plt.fill_between(l0250_d100[:,0], l0250_d100[:,11], l0250_d100[:,12], facecolor='r', edgecolor='k', alpha=0.1)
plt.fill_between(l0250_d75[:,0], l0250_d75[:,11], l0250_d75[:,12], facecolor='c', edgecolor='k', alpha=0.1)
plt.fill_between(l0250_d50[:,0], l0250_d50[:,11], l0250_d50[:,12], facecolor='g', edgecolor='k', alpha=0.1)

plt.semilogx(l0250_d200[:,0], l0250_d200[:,10], 'b-', 
             label='$\Delta = 200, M_{\odot}h^{-1} \geq 7e11$')
plt.plot(l0250_d100[:,0], l0250_d100[:,10], 'r-', 
             label='$\Delta = 100, M_{\odot}h^{-1} \geq 8e11$')
plt.plot(l0250_d75[:,0], l0250_d75[:,10], 'c-', 
             label='$\Delta = 75, M_{\odot}h^{-1} \geq 9e11$')
plt.plot(l0250_d50[:,0], l0250_d50[:,10], 'g-', 
             label='$\Delta = 50, M_{\odot}h^{-1} \geq 1.5e12$')

plt.legend(loc='upper right', prop={'size': 8})

plt.xlim(4,18)
#plt.ylim(-0.02,0.1)
plt.xlabel(r'$r \ (h^{-1} \mathrm{Mpc})$')
plt.ylabel(r'$\mathcal{M}_{\mathrm{shape}}$')
plt.title('L0250')
plt.tight_layout()
plt.savefig('./FIGS/l0250_mcf_ctoa.pdf')
plt.clf()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

plt.axhline(y=0, color='k', linestyle='--')
plt.fill_between(l0125_d200[:,0], l0125_d200[:,11], l0125_d200[:,12], facecolor='b', edgecolor='k', alpha=0.1)
plt.fill_between(l0125_d100[:,0], l0125_d100[:,11], l0125_d100[:,12], facecolor='r', edgecolor='k', alpha=0.1)
plt.fill_between(l0125_d75[:,0], l0125_d75[:,11], l0125_d75[:,12], facecolor='c', edgecolor='k', alpha=0.1)
plt.fill_between(l0125_d50[:,0], l0125_d50[:,11], l0125_d50[:,12], facecolor='g', edgecolor='k', alpha=0.1)

plt.semilogx(l0125_d200[:,0], l0125_d200[:,10], 'b-', 
             label='$\Delta = 200, M_{\odot}h^{-1} \geq 7e10$')
plt.plot(l0125_d100[:,0], l0125_d100[:,10], 'r-', 
             label='$\Delta = 100, M_{\odot}h^{-1} \geq 8e10$')
plt.plot(l0125_d75[:,0], l0125_d75[:,10], 'c-', 
             label='$\Delta = 75, M_{\odot}h^{-1} \geq 9e10$')
plt.plot(l0125_d50[:,0], l0125_d50[:,10], 'g-', 
             label='$\Delta = 50, M_{\odot}h^{-1} \geq 1e11$')

plt.legend(loc='upper right', prop={'size': 8})

plt.xlim(4,18)
#plt.ylim(-0.01,0.02)
plt.xlabel(r'$r \ (h^{-1} \mathrm{Mpc})$')
plt.ylabel(r'$\mathcal{M}_{\mathrm{shape}}$')
plt.title('L0125')
plt.tight_layout()
plt.savefig('./FIGS/l0125_mcf_ctoa.pdf')
plt.clf()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

plt.axhline(y=0, color='k', linestyle='--')
plt.fill_between(l0500_d200[:,0], l0500_d200[:,14], l0500_d200[:,15], facecolor='b', edgecolor='k', alpha=0.1)
plt.fill_between(l0500_d100[:,0], l0500_d100[:,14], l0500_d100[:,15], facecolor='r', edgecolor='k', alpha=0.1)
plt.fill_between(l0500_d75[:,0], l0500_d75[:,14], l0500_d75[:,15], facecolor='c', edgecolor='k', alpha=0.1)
plt.fill_between(l0500_d50[:,0], l0500_d50[:,14], l0500_d50[:,15], facecolor='g', edgecolor='k', alpha=0.1)

plt.semilogx(l0500_d200[:,0], l0500_d200[:,13], 'b-', 
             label='$\Delta = 200, M_{\odot}h^{-1} \geq 4e12$')
plt.plot(l0500_d100[:,0], l0500_d100[:,13], 'r-', 
             label='$\Delta = 100, M_{\odot}h^{-1} \geq 5e12$')
plt.plot(l0500_d75[:,0], l0500_d75[:,13], 'c-', 
             label='$\Delta = 75, M_{\odot}h^{-1} \geq 6e12$')
plt.plot(l0500_d50[:,0], l0500_d50[:,13], 'g-', 
             label='$\Delta = 50, M_{\odot}h^{-1} \geq 7e12$')

plt.legend(loc='upper right', prop={'size': 8})

plt.xlim(4,18)
#plt.ylim(-0.02,0.06)
plt.xlabel(r'$r \ (h^{-1} \mathrm{Mpc})$')
plt.ylabel(r'$\mathcal{M}_{\mathrm{spin}}$')
plt.title('L0500')
plt.tight_layout()
plt.savefig('./FIGS/l0500_mcf_spin.pdf')
plt.clf()


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

plt.axhline(y=0, color='k', linestyle='--')
plt.fill_between(l0250_d200[:,0], l0250_d200[:,14], l0250_d200[:,15], facecolor='b', edgecolor='k', alpha=0.1)
plt.fill_between(l0250_d100[:,0], l0250_d100[:,14], l0250_d100[:,15], facecolor='r', edgecolor='k', alpha=0.1)
plt.fill_between(l0250_d75[:,0], l0250_d75[:,14], l0250_d75[:,15], facecolor='c', edgecolor='k', alpha=0.1)
plt.fill_between(l0250_d50[:,0], l0250_d50[:,14], l0250_d50[:,15], facecolor='g', edgecolor='k', alpha=0.1)

plt.semilogx(l0250_d200[:,0], l0250_d200[:,13], 'b-', 
             label='$\Delta = 200, M_{\odot}h^{-1} \geq 7e11$')
plt.plot(l0250_d100[:,0], l0250_d100[:,13], 'r-', 
             label='$\Delta = 100, M_{\odot}h^{-1} \geq 8e11$')
plt.plot(l0250_d75[:,0], l0250_d75[:,13], 'c-', 
             label='$\Delta = 75, M_{\odot}h^{-1} \geq 9e11$')
plt.plot(l0250_d50[:,0], l0250_d50[:,13], 'g-', 
             label='$\Delta = 50, M_{\odot}h^{-1} \geq 1.5e12$')

plt.legend(loc='upper right', prop={'size': 8})

plt.xlim(4,18)
#plt.ylim(-0.02,0.06)
plt.xlabel(r'$r \ (h^{-1} \mathrm{Mpc})$')
plt.ylabel(r'$\mathcal{M}_{\mathrm{spin}}$')
plt.title('L0250')
plt.tight_layout()
plt.savefig('./FIGS/l0250_mcf_spin.pdf')
plt.clf()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}


matplotlib.rc('font', **font)

plt.axhline(y=0, color='k', linestyle='--')
plt.fill_between(l0125_d200[:,0], l0125_d200[:,14], l0125_d200[:,15], facecolor='b', edgecolor='k', alpha=0.1)
plt.fill_between(l0125_d100[:,0], l0125_d100[:,14], l0125_d100[:,15], facecolor='r', edgecolor='k', alpha=0.1)
plt.fill_between(l0125_d75[:,0], l0125_d75[:,14], l0125_d75[:,15], facecolor='c', edgecolor='k', alpha=0.1)
plt.fill_between(l0125_d50[:,0], l0125_d50[:,14], l0125_d50[:,15], facecolor='g', edgecolor='k', alpha=0.1)

plt.semilogx(l0125_d200[:,0], l0125_d200[:,13], 'b-', 
             label='$\Delta = 200, M_{\odot}h^{-1} \geq 7e10$')
plt.plot(l0125_d100[:,0], l0125_d100[:,13], 'r-', 
             label='$\Delta = 100, M_{\odot}h^{-1} \geq 8e10$')
plt.plot(l0125_d75[:,0], l0125_d75[:,13], 'c-', 
             label='$\Delta = 75, M_{\odot}h^{-1} \geq 9e10$')
plt.plot(l0125_d50[:,0], l0125_d50[:,13], 'g-', 
             label='$\Delta = 50, M_{\odot}h^{-1} \geq 1e11$')

plt.legend(loc='upper right', prop={'size': 8})

plt.xlim(4,18)
#plt.ylim(-0.01,0.01)
plt.xlabel(r'$r \ (h^{-1} \mathrm{Mpc})$')
plt.ylabel(r'$\mathcal{M}_{\mathrm{spin}}$')
plt.title('L0125')
plt.tight_layout()
plt.savefig('./FIGS/l0125_mcf_spin.pdf')
plt.clf()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

plt.axhline(y=0, color='k', linestyle='--')
plt.fill_between(l0500_d200[:,0], l0500_d200[:,17], l0500_d200[:,18], facecolor='b', edgecolor='k', alpha=0.1)
plt.fill_between(l0500_d100[:,0], l0500_d100[:,17], l0500_d100[:,18], facecolor='r', edgecolor='k', alpha=0.1)
plt.fill_between(l0500_d75[:,0], l0500_d75[:,17], l0500_d75[:,18], facecolor='c', edgecolor='k', alpha=0.1)
plt.fill_between(l0500_d50[:,0], l0500_d50[:,17], l0500_d50[:,18], facecolor='g', edgecolor='k', alpha=0.1)

plt.semilogx(l0500_d200[:,0], l0500_d200[:,16], 'b-', 
             label='$\Delta = 200, M_{\odot}h^{-1} \geq 4e12$')
plt.plot(l0500_d100[:,0], l0500_d100[:,16], 'r-', 
             label='$\Delta = 100, M_{\odot}h^{-1} \geq 5e12$')
plt.plot(l0500_d75[:,0], l0500_d75[:,16], 'c-', 
             label='$\Delta = 75, M_{\odot}h^{-1} \geq 6e12$')
plt.plot(l0500_d50[:,0], l0500_d50[:,16], 'g-', 
             label='$\Delta = 50, M_{\odot}h^{-1} \geq 7e12$')

plt.legend(loc='upper right', prop={'size': 8})

plt.xlim(4,18)
#plt.ylim(-0.02,0.06)
plt.xlabel(r'$r \ (h^{-1} \mathrm{Mpc})$')
plt.ylabel(r'$\mathcal{M}_{\mathrm{Nsat}}$')
plt.title('L0500')
plt.tight_layout()
plt.savefig('./FIGS/l0500_mcf_nsat.pdf')
plt.clf()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

plt.axhline(y=0, color='k', linestyle='--')
plt.fill_between(l0250_d200[:,0], l0250_d200[:,17], l0250_d200[:,18], facecolor='b', edgecolor='k', alpha=0.1)
plt.fill_between(l0250_d100[:,0], l0250_d100[:,17], l0250_d100[:,18], facecolor='r', edgecolor='k', alpha=0.1)
plt.fill_between(l0250_d75[:,0], l0250_d75[:,17], l0250_d75[:,18], facecolor='c', edgecolor='k', alpha=0.1)
plt.fill_between(l0250_d50[:,0], l0250_d50[:,17], l0250_d50[:,18], facecolor='g', edgecolor='k', alpha=0.1)

plt.semilogx(l0250_d200[:,0], l0250_d200[:,16], 'b-', 
             label='$\Delta = 200, M_{\odot}h^{-1} \geq 7e11$')
plt.plot(l0250_d100[:,0], l0250_d100[:,16], 'r-', 
             label='$\Delta = 100, M_{\odot}h^{-1} \geq 8e11$')
plt.plot(l0250_d75[:,0], l0250_d75[:,16], 'c-', 
             label='$\Delta = 75, M_{\odot}h^{-1} \geq 9e11$')
plt.plot(l0250_d50[:,0], l0250_d50[:,16], 'g-', 
             label='$\Delta = 50, M_{\odot}h^{-1} \geq 1.5e12$')

plt.legend(loc='upper right', prop={'size': 8})

plt.xlim(4,18)
#plt.ylim(-0.02,0.06)
plt.xlabel(r'$r \ (h^{-1} \mathrm{Mpc})$')
plt.ylabel(r'$\mathcal{M}_{\mathrm{Nsat}}$')
plt.title('L0250')
plt.tight_layout()
plt.savefig('./FIGS/l0250_mcf_nsat.pdf')
plt.clf()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

plt.axhline(y=0, color='k', linestyle='--')
plt.fill_between(l0125_d200[:,0], l0125_d200[:,17], l0125_d200[:,18], facecolor='b', edgecolor='k', alpha=0.1)
plt.fill_between(l0125_d100[:,0], l0125_d100[:,17], l0125_d100[:,18], facecolor='r', edgecolor='k', alpha=0.1)
plt.fill_between(l0125_d75[:,0], l0125_d75[:,17], l0125_d75[:,18], facecolor='c', edgecolor='k', alpha=0.1)
plt.fill_between(l0125_d50[:,0], l0125_d50[:,17], l0125_d50[:,18], facecolor='g', edgecolor='k', alpha=0.1)

plt.semilogx(l0125_d200[:,0], l0125_d200[:,16], 'b-', 
             label='$\Delta = 200, M_{\odot}h^{-1} \geq 7e10$')
plt.plot(l0125_d100[:,0], l0125_d100[:,16], 'r-', 
             label='$\Delta = 100, M_{\odot}h^{-1} \geq 8e10$')
plt.plot(l0125_d75[:,0], l0125_d75[:,16], 'c-', 
             label='$\Delta = 75, M_{\odot}h^{-1} \geq 9e10$')
plt.plot(l0125_d50[:,0], l0125_d50[:,16], 'g-', 
             label='$\Delta = 50, M_{\odot}h^{-1} \geq 1e11$')

plt.legend(loc='upper right', prop={'size': 8})

plt.xlim(4,18)
#plt.ylim(-0.02,0.06)
plt.xlabel(r'$r \ (h^{-1} \mathrm{Mpc})$')
plt.ylabel(r'$\mathcal{M}_{\mathrm{Nsat}}$')
plt.title('L0125')
plt.tight_layout()
plt.savefig('./FIGS/l0125_mcf_nsat.pdf')
plt.clf()


