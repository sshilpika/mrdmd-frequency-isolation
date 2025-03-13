import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos
from numpy.linalg import inv, eig, pinv, solve
from scipy.linalg import svd, svdvals
from math import floor, ceil # python 3.x
import time
import random
from joblib import Parallel, delayed
import itertools

class MrDMDZscore():
    '''
    Code modified from https://humaticlabs.com/blog/mrdmd-python/
    '''

    def svht(self, X, sv=None):
        # svht for sigma unknown
        m,n = sorted(X.shape) # ensures m <= n
        beta = m / n # ratio between 0 and 1
        if sv is None:
            sv = svdvals(X)
        sv = np.squeeze(sv)
        omega_approx = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        return np.median(sv) * omega_approx
    
    def dmd(self, X, Y, truncate=None):
        if truncate == 0:
            # return empty vectors
            mu = np.array([], dtype='complex')
            Phi = np.zeros([X.shape[0], 0], dtype='complex')
        else:
            U2,Sig2,Vh2 = svd(X, False) # SVD of input matrix
            r = len(Sig2) if truncate is None else truncate # rank truncation
            U = U2[:,:r]
            Sig = diag(Sig2)[:r,:r]
            V = Vh2.conj().T[:,:r]
            Atil = dot(dot(dot(U.conj().T, Y), V), inv(Sig)) # build A tilde
            mu,W = eig(Atil)
            Phi = dot(dot(dot(Y, V), inv(Sig)), W) # build DMD modes
        return mu, Phi
    
    def mrdmd(self, D, level=0, bin_num=0, offset=0, max_levels=7, max_cycles=2, do_svht=True, do_parallel=False):
       
        # 4 times nyquist limit to capture cycles
        nyq = 8 * max_cycles
    
        # time bin size
        bin_size = D.shape[1]
        if bin_size < nyq:
            return []
    
        # extract subsamples 
        step = floor(bin_size / nyq) # max step size to capture cycles
        _D = D[:,::step]
        X = _D[:,:-1]
        Y = _D[:,1:]
    
        # determine rank-reduction
        if do_svht:
            _sv = svdvals(_D)
            tau = self.svht(_D, sv=_sv)
            r = sum(_sv > tau)
        else:
            r = min(X.shape)
    
        # compute dmd
        mu,Phi = self.dmd(X, Y, r)
    
        # frequency cutoff (oscillations per timestep)
        rho = max_cycles / bin_size
    
        # consolidate slow eigenvalues (as boolean mask)
        slow = (np.abs(np.log(mu) / ((2 * pi * step) + 1e-20 ))) <= rho
        n = sum(slow) # number of slow modes
    
        # extract slow modes (perhaps empty)
        mu = mu[slow]
        Phi = Phi[:,slow]
    
        if n > 0:
    
            # vars for the objective function for D (before subsampling)
            Vand = np.vander(power(mu, 1/step), bin_size, True)
            P = multiply(dot(Phi.conj().T, Phi), np.conj(dot(Vand, Vand.conj().T)))
            q = np.conj(diag(dot(dot(Vand, D.conj().T), Phi)))
    
            # find optimal b solution
            b_opt = solve(P, q).squeeze()
    
            # time evolution
            Psi = (Vand.T * b_opt).T
    
        else:
    
            # zero time evolution
            b_opt = np.array([], dtype='complex')
            Psi = np.zeros([0, bin_size], dtype='complex')
    
        # dmd reconstruction
        D_dmd = dot(Phi, Psi)   
    
        # remove influence of slow modes
        D = D - D_dmd
    
        # record keeping
        node = type('Node', (object,), {})()
        node.level = level            # level of recursion
        node.bin_num = bin_num        # time bin number
        node.bin_size = bin_size      # time bin size
        node.start = offset           # starting index
        node.stop = offset + bin_size # stopping index
        node.step = step              # step size
        node.rho = rho                # frequency cutoff
        node.r = r                    # rank-reduction
        node.n = n                    # number of extracted modes
        node.mu = mu                  # extracted eigenvalues
        node.Phi = Phi                # extracted DMD modes
        node.Psi = Psi                # extracted time evolution
        node.b_opt = b_opt            # extracted optimal b vector
    
        # if do_parallel: 
        #     nodes=node
        # else:
        nodes = [node]
    
        # split data into two and do recursion
        if level < max_levels:
                split = ceil(bin_size / 2) # where to split
        
                if not do_parallel:
                    
                    nodes += self.mrdmd(
                        D[:,:split],
                        level=level+1,
                        bin_num=2*bin_num,
                        offset=offset,
                        max_levels=max_levels,
                        max_cycles=max_cycles,
                        do_svht=do_svht,
                        do_parallel=do_parallel
                        )
                    nodes += self.mrdmd(
                        D[:,split:],
                        level=level+1,
                        bin_num=2*bin_num+1,
                        offset=offset+split,
                        max_levels=max_levels,
                        max_cycles=max_cycles,
                        do_svht=do_svht,
                        do_parallel=do_parallel
                        )
    
                else:
                
                    args = [[D[:,:split], level+1, 2*bin_num, offset, max_levels, max_cycles, do_svht, do_parallel], 
                                   [D[:,split:], level+1, 2*bin_num+1, offset+split, max_levels, max_cycles, do_svht, do_parallel]]
                
                    nodes_ = Parallel(n_jobs=2)(delayed(self.mrdmd)(*arg) for arg in args)
                    nodes +=  list(itertools.chain(*nodes_))
    
        
        return nodes
    
    
    def stitch(self, nodes, level):
        
        # get length of time dimension
        start = min([nd.start for nd in nodes])
        stop = max([nd.stop for nd in nodes])
        t = stop - start
    
        # extract relevant nodes
        nodes = [n for n in nodes if n.level == level]
        nodes = sorted(nodes, key=lambda n: n.bin_num)
        
        # stack DMD modes
        Phi = np.hstack([n.Phi for n in nodes])
        
        # allocate zero matrix for time evolution
        nmodes = sum([n.n for n in nodes])
        Psi = np.zeros([nmodes, t], dtype='complex')
        
        # copy over time evolution for each time bin
        i = 0
        for n in nodes:
            _nmodes = n.Psi.shape[0]
            Psi[i:i+_nmodes,n.start:n.stop] = n.Psi
            i += _nmodes
        
        return Phi,Psi


    def compute_zscore(self, data, all_splits, nodes, baseline_indx, n_baseline_indx, std_baselines = 0.00366, for_baseline=False, plot=True):
        #showing zscore computation only for data 
        # for_baseline: True for computing the std deviation for baselines the function returns the std deviation
        # for_baseline: False z-scores for data are returned
        # For more details refer https://doi.org/10.1016/j.jneumeth.2015.10.010
        
        zscore_list = []
                
        D__ = data.copy()
        
        meas_count = data.shape[0]
        
        ax = plt.subplot()
        std_currs = []
        tsIDs = []
        
        brkPoints_levels = [[(0,data.shape[1])]]
        for splits in all_splits:
            split = [0] + splits + [data.shape[1]]
            brkPoints_levels.append(list(zip(split[:-1], split[1:])))
            
        # Here we choose to compute the zscores for the entire timeline - range 1
        for blev in range(1):
            for tsID in brkPoints_levels[blev]:
                level = len(brkPoints_levels)
        
                t = np.arange(0,np.diff(tsID)[0],1)
                
                sorted_nodes = [n for l in range(level) 
                        for n in self.get_sorted_nodes_in_level(nodes,l)[0] if n.start <= tsID[1] and 
                        n.stop >= tsID[0]]
        
                mu = np.hstack([n.mu for n in sorted_nodes])
        
                dt = None
                if len(t) == 1:
                    dt = 1
                else:
                    dt = 1/len(t)
        
                omega = np.hstack([np.log(n.mu)*100/(n.step) for n in sorted_nodes])
                phi = np.hstack([n.Phi for n in sorted_nodes])
        
                f = abs(omega.imag/2*np.pi)
                P = diag(np.matmul(phi.conj().T, phi))
        
                #mrdmd power spectrum
                if plot:
                    plt.scatter(x=f, y=P.real, c='red',edgecolor='black', s=20 ,label= "Not Baseline" )
            
                    plt.xlabel('Frequencies')
                    plt.ylabel('DMD power')
                    plt.title("new DMD")
        
                Xaug_small = D__[:, tsID[0]:tsID[1]]
        
                indxs = []
        
                #choose the appropriate frequency range 
                for ind, (fr, ph) in enumerate(zip(f, P)):
                    if fr > 0 and fr < 60:
                        indxs.append(ind)
                if len(indxs) == 0:
                    continue
                    indxs = [i for i in range(len(P))]
        
                tsIDs.append(tsID)
           
                dmd_mode_freqs = phi[:,indxs]
                dmd_freqs_mean = np.mean(abs(dmd_mode_freqs), axis=1)
                dmd_freqs_mean = dmd_freqs_mean[:meas_count]
                
                # subtract baseline mode with the current readings (curr) 
                baseline_mean = np.mean(dmd_freqs_mean[baseline_indx])
                n_baseline_dmd_modes = dmd_freqs_mean[n_baseline_indx] - baseline_mean
                val = [np.power(dmd_freqs_mean[i]-baseline_mean, 2) for i in n_baseline_indx]
                std_curr = np.sqrt(np.mean(val))
                std_currs.append(std_curr)

                if not for_baseline:
                    std_curr = std_baselines
                  
                #zscore
                dmd_mode_n_baseline = dmd_freqs_mean#[n_baseline_indx]
                zscore_1 = (dmd_mode_n_baseline - baseline_mean)/std_curr
                zscore_list.append(zscore_1)
            if plot:
                plt.show()
                plt.figure(figsize= (15,5))
                for zscore_1 in zscore_list: 
            
                    ax = plt.scatter( x = np.arange(len(zscore_1)), y = zscore_1, s=3, edgecolors='black')
                    plt.scatter(x = baseline_indx, y = zscore_1[baseline_indx], c='red',s=5,edgecolors='red')
                plt.title("Z-scores")
                plt.show()

        if not for_baseline:
            return zscore_list   
        else:
            return std_currs


    def get_splt(self, tstep = 1000, max_levels = 12, ):
        
        lll = max_levels
        splt = [[] for _ in range(lll+1)]
        
        def gen_half_splits(j=np.arange(5000),lll = 0, levels=12):
            if lll > levels:
                return
            spl = ceil(len(j) / 2)
            splt[lll].append(spl)
            gen_half_splits(j = j[:spl],lll=lll+1,levels=levels)
            gen_half_splits(j = j[spl:],lll=lll+1,levels=levels)
        
        gen_half_splits(np.arange(tstep),lll=0, levels=max_levels)
    
        for ind, v in enumerate(splt):
            splt[ind] = list(np.cumsum(v*2))
    
        for ind, v in enumerate(splt):
            v[-1] = tstep-3
    
        for ind, v in enumerate(splt):
            splt[ind] = [vv for vv in v if vv < tstep]
     
        splt[0] = [splt[0][0]]  
        return splt
    
    
    def get_sorted_nodes_in_level(self, nodes, level):
        # get length of time dimension
        start = min([nd.start for nd in nodes])
        stop = max([nd.stop for nd in nodes])
        t = stop - start
    
        # extract relevant nodes
        nodes = [n for n in nodes if n.level == level]
        nodes = sorted(nodes, key=lambda n: n.bin_num)
        return nodes, t    

             
        
