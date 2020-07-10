import numpy as np
from itertools import combinations as comb

def combn(m, n):
    return np.array(list(comb(range(m), n)))

def Borda(mat):
    np.fill_diagonal(mat, 1)
    mat = mat/(mat+mat.T)
    np.fill_diagonal(mat, 0)
    
    return np.sum(mat, axis=1)

def BTL(Data, probs=False, max_iter=10**5):
    '''
    computes the parameters using maximum likelihood principle.
    This function is adapted from the Matlab version provided by David Hunter
    http://personal.psu.edu/drh20/code/btmatlab
    '''
    wm = Data
    if probs:
        np.fill_diagonal(wm, 1)
        wm = wm/(wm+wm.T)
        np.fill_diagonal(wm, 0)
    n = wm.shape[0]
    nmo = n-1
    pi = np.ones(nmo, dtype=float)
    gm = (wm[:,range(nmo)]).T + wm[range(nmo),:]
    wins = np.sum(wm[range(nmo),], axis=1)
    gind = gm>0
    z = np.zeros((nmo,n))
    pisum = z
    for _ in range(max_iter):
        pius = np.repeat(pi, n).reshape(nmo, -1)
        piust = (pius[:,range(nmo)]).T
        piust = np.column_stack((piust, np.repeat(1,nmo)))
        pisum[gind] = pius[gind]+piust[gind]
        z[gind] = gm[gind] / pisum[gind]
        newpi = wins / np.sum(z, axis=1)
        if np.linalg.norm(newpi - pi, ord=np.inf) <= 1e-6:
            newpi = np.append(newpi, 1)
            return newpi/sum(newpi)
        pi = newpi
    raise RuntimeError('did not converge')

'''
AB: numpy array where each row (instance) is \in [-1,1]^d
CD: numpy array where each row (instance) is \in [-1,1]^d
'''
def analogy(AB,CD):
    ''' equivalent analogies a:b::c:d b:a::d:c c:d::a:b d:c::b:a '''
    ''' equivalent analogies a:b::d:c b:a::c:d c:d::b:a d:c::a:b '''
    
    S = 1 - np.abs(AB-CD)
    cond0 = AB*CD < 0
    cond1 = (AB==0) & (CD!=0)
    cond2 = (AB!=0) & (CD==0)
    S[ cond0 | cond1 | cond2 ] = 0
    if S.ndim==1:
        S = S.reshape(-1, len(S))
    return np.mean(S, axis=1)

'''
arr_trn: numpy array containing n instances \in [0,1]^d
y_trn: numpy array of length n containing the rank of instances in arr_trn
arr_tst: numpy array containing n instances \in [0,1]^d
k: (integer) the no. of nearest neighbors
agg: (string) aggregation function to be used 
'''
def able2rank_arithmetic(arr_trn, y_trn, arr_tst, k, agg):
    arr_trn = arr_trn[ np.argsort(y_trn),: ]
    nr_trn = arr_trn.shape[0]
    nr_tst = arr_tst.shape[0]
    nc = arr_trn.shape[1]
    cmb_trn = combn(nr_trn, 2)
    a_minus_b = arr_trn[ cmb_trn[:,0] ] - arr_trn[ cmb_trn[:,1] ]
    cmb_tst = combn(nr_tst, 2)
    mat = np.identity(nr_tst)-1
    for t in range(cmb_tst.shape[0]):
        i, j = cmb_tst[t,:]
        
        c_minus_d = (arr_tst[i,:] - arr_tst[j,:]).reshape(-1, nc)
        c_minus_d = np.repeat( c_minus_d, cmb_trn.shape[0], axis=0 )
        
        d_minus_c = -c_minus_d
        
        abcd = analogy(a_minus_b, c_minus_d)
        abdc = analogy(a_minus_b, d_minus_c)

        '''assuming arr_trn is ranked from top to bottom'''
        merged = np.column_stack((abcd, abdc))
        score = np.amax(merged, axis=1)
        top_k_inds = np.argsort(-score)[range(k)]
        c_pref_d = (abcd>abdc)[top_k_inds]
        mat[i,j] = np.sum(c_pref_d)
        mat[j,i] = k - mat[i,j]
    
    theta = BTL(mat.copy(), probs=False) if agg=="BTL" else Borda(mat.copy())
    return np.argsort(-theta), mat
