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
