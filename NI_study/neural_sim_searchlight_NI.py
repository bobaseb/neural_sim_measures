'''
 10 Oct 2018

@author: Sebastian Bobadilla-Suarez

This script is a searchlight version of the neural_sim_per_roi.py script. It does not optimize the number of voxels.

'''

import mvpa2.suite as mvpa2
import sys
import pickle
import os
import numpy as np
import pandas as pd
import itertools
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.spatial import distance
import scipy as sp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from sklearn.covariance import ledoit_wolf
from scipy.spatial.distance import pdist, squareform

from mvpa2.clfs.skl.base import SKLLearnerAdapter
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(5)

from sklearn import linear_model
from sklearn.svm import NuSVC

def distcorr(X, Y):
    """ Compute the distance correlation function
	https://gist.github.com/satra/aa3d19a12b74e9ab7941
	Satrajit Ghosh
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

def var_info(mat1, mat2):
    n = mat1.shape[1]
    mat0 = np.hstack([mat1,mat2])
    cov_mat0 = ledoit_wolf(mat0)[0]
    cov_mat1 = ledoit_wolf(mat1)[0]
    cov_mat2 = ledoit_wolf(mat2)[0]
    (sign0, logdet0) = np.linalg.slogdet(cov_mat0)
    (sign1, logdet1) = np.linalg.slogdet(cov_mat1)
    (sign2, logdet2) = np.linalg.slogdet(cov_mat2)
    ln_det_mat0 = logdet0
    ln_det_mat1 = logdet1
    ln_det_mat2 = logdet2
    H_mat1 = 0.5*np.log(np.power((2*np.exp(1)*np.pi), n)) + 0.5*ln_det_mat1
    H_mat2 = 0.5*np.log(np.power((2*np.exp(1)*np.pi), n)) + 0.5*ln_det_mat2
    MI = 0.5*(ln_det_mat1 + ln_det_mat2 - ln_det_mat0)
    return H_mat1 + H_mat2 - 2*MI;

def bhdist (mu1, mu2, mat1, mat2, cov_est=1):
    #Bhattacharyya_distance assuming normal distros
    diff_mn_mat = np.matrix(mu1-mu2).T
    if(cov_est==0):
         cov_mat1 = np.cov(np.matrix(mat1).T)
         cov_mat2 = np.cov(np.matrix(mat2).T)
    elif(cov_est==1):
         cov_mat1 = ledoit_wolf(mat1)[0]
         cov_mat2 = ledoit_wolf(mat2)[0]
    elif(cov_est==2):
         cov_mat1 = diag_covmat(mat1)
         cov_mat2 = diag_covmat(mat2)
    cov_mat_mn = (cov_mat1 + cov_mat2)/2
    icov_mat_mn = invcov_mah(cov_mat_mn,0)
    term1 = np.dot(np.dot(diff_mn_mat.T, icov_mat_mn), diff_mn_mat)/8
    (sign1, logdet1) = np.linalg.slogdet(cov_mat1)
    (sign2, logdet2) = np.linalg.slogdet(cov_mat2)
    (sign_mn, logdet_mn) = np.linalg.slogdet(cov_mat_mn)
    ln_det_mat1 = logdet1
    ln_det_mat2 = logdet2
    ln_det_mat_mn = logdet_mn
    term2 = (ln_det_mat_mn/2) - (ln_det_mat1+ln_det_mat2)/4 #np.log(det_mat_mn/np.sqrt(det_mat1*det_mat2))/2
    result = term1+term2;
    return result[0,0];

def invcov_mah(mat,cov_est=1):
    #this function is mostly for regularising the covariance matrix used in the mahalanobis and bhattacharyya distances (cov_est setting). It also attempts to compute either en inverse or pseudo-inverse covariance matrix.
    if(cov_est==0):
        mat = np.cov(np.matrix(mat).T)
    elif(cov_est==1):
        mat = ledoit_wolf(mat)[0]
    elif(cov_est==2):
        mat = diag_covmat(mat)
    try:
        icov_mat = np.linalg.inv(mat)
    except:
        icov_mat = np.linalg.pinv(mat)
    return icov_mat;

def diag_covmat(mat):
    #this function only keeps the main diagonal of a covariance matrix
    mat = np.cov(np.matrix(mat).T)
    mat = np.diag(np.diag(mat)) #only keep diagonal of covariance matrix
    return mat;

#these are the distance functions we cycle through in the dist_wrapper function
dists = ['euclidean','pearson','mahalanobis']

def distance_funcs(mu1, mu2, mat0=0, mat1=0, mat2=0, dist_num=0):
    #this is a wrapper listing all the distance functions
    if(dist_num=='inner_prod'):
        dist = -1*np.inner(mu1,mu2) #inner product
    elif(dist_num=='cosine'):
        dist = distance.cosine(mu1,mu2) #cosine
    elif(dist_num=='cityblock'):
        dist = distance.cityblock(mu1,mu2) #manhattan/city-block
    elif(dist_num=='euclidean'):
        dist = distance.euclidean(mu1,mu2) #euclidean
    elif(dist_num=='minkowski(5)'):
        dist = distance.minkowski(mu1,mu2,5) #minkoswki 5
    elif(dist_num=='minkowski(10)'):
        dist = distance.minkowski(mu1,mu2,10) #minkoswki 10
    elif(dist_num=='minkowski(50)'):
        dist = distance.minkowski(mu1,mu2,50) #minkoswki 50
    elif(dist_num=='chebyshev'):
        dist = distance.chebyshev(mu1,mu2) #chebyshev
    elif(dist_num=='spearman'):
        dist = 1 - spearmanr(mu1,mu2)[0] #spearman correlation
    elif(dist_num=='pearson'):
        dist = 1 - pearsonr(mu1,mu2)[0] #pearson correlation
    elif(dist_num=='mahalanobis'):
        VI = invcov_mah(mat0) #inverse covariance matrix for mahalanobis
        dist = distance.mahalanobis(mu1,mu2, VI) #mahalanobis
    elif(dist_num=='bhattacharya'):
        dist = bhdist(mu1,mu2,mat1,mat2) #bhattacharya (voxel space)
    elif(dist_num=='var_info'):
        dist = var_info(mat1,mat2)  #variation of information
    elif(dist_num=='distcorr'):
        dist = 1 - distcorr(mat1,mat2)  #variation of information
    elif(dist_num=='mahab_noreg'):
        VI = invcov_mah(mat0,0) #inverse covariance matrix for mahalanobis
        dist = distance.mahalanobis(mu1,mu2, VI) #mahalanobis
    elif(dist_num=='mahab_diag'):
        VI = invcov_mah(mat0,2) #inverse covariance matrix for mahalanobis
        dist = distance.mahalanobis(mu1,mu2, VI) #mahalanobis
    elif(dist_num=='bhat_noreg'):
        dist = bhdist(mu1,mu2,mat1,mat2,0) #bhattacharya (voxel space)
    elif(dist_num=='bhat_diag'):
        dist = bhdist(mu1,mu2,mat1,mat2,2) #bhattacharya (voxel space)
    return dist;

def clf_wrapper(ds):
    #1 of 2 main functions. computes the crossvalidated classifier we base our inferences on
    test_accs_per_chunk=[] #array with accuracies for each test fold
    val_accs_per_chunk=[]
    nfs_all_chunks=[]
    val_chunks=[]
    mvpa2.zscore(ds, chunks_attr = 'chunks')
    for chunk in chunk_num: #does LOOCV
        val_chunk = np.random.choice(ds.uniquechunks[ds.uniquechunks!=chunk])
        val_chunks.append(val_chunk)
        def optimize_clf(nf, optimize=1):
            acc_list=[] #array with accuracies for each pair within each LOOVC fold
            def nf_select(nf):
                #fselector = mvpa2.FixedNElementTailSelector(np.round(nf), tail='upper',mode='select', sort=False)
                #sbfs = mvpa2.SensitivityBasedFeatureSelection(mvpa2.OneWayAnova(), fselector, enable_ca=['sensitivities'], auto_train=True)
                if(optimize>=1):
                    not_test_ds = ds[ds.chunks!=chunk]
                    val_ds = not_test_ds[not_test_ds.chunks==val_chunk]
                    train_ds = not_test_ds[not_test_ds.chunks!=val_chunk]
                    #sbfs.train(train_ds)
                    #train_ds = sbfs(train_ds)
                    #val_ds = sbfs(val_ds)
                    return train_ds, val_ds;
                elif(optimize==0):
                    train_ds = ds[ds.chunks!=chunk]
                    test_ds = ds[ds.chunks==chunk]
                    #sbfs.train(train_ds)
                    #train_ds = sbfs(train_ds)
                    #test_ds = sbfs(test_ds)
                    return train_ds, test_ds;
            train_ds, not_train_ds = nf_select(nf)
            for y in range(0, len(pair_list2)):
                def mask(y, train_ds, test_ds):
                    stim_mask1 = (train_ds.targets==pair_list2[y][0]) | (train_ds.targets==pair_list2[y][1])
                    stim_mask2 = (not_train_ds.targets==pair_list2[y][0]) | (not_train_ds.targets==pair_list2[y][1])
                    ds_temp_train = train_ds[stim_mask1]
                    ds_temp_not_train = not_train_ds[stim_mask2]
                    return ds_temp_train, ds_temp_not_train;
                ds_temp_train, ds_temp_not_train = mask(y, train_ds, not_train_ds)
                #clf = mvpa2.LinearNuSVMC(nu=0.5)#defines a classifier, linear SVM in this case
                clf = NuSVC(nu=0.5, max_iter=2000)
                #clf = SKLLearnerAdapter(knn)
                #clf = SKLLearnerAdapter(linear_model.SGDClassifier())
                #clf.train(ds_temp_train)
                clf.fit(ds_temp_train.samples,ds_temp_train.targets)
                #predictions = clf.predict(ds_temp_not_train)
                predictions = clf.predict(ds_temp_not_train.samples)
                labels = ds_temp_not_train.targets
                bool_vec = predictions==labels
                acc_list.append(sum(bool_vec)/float(len(bool_vec))) #array with accuracies for each pair
            if(optimize==1):
                #print len(acc_list)
                #print np.mean(acc_list)
                return 1-np.mean(acc_list);
            else:
                #print np.mean(acc_list), 'for chunk:', chunk
                return acc_list;
        #f = minimize_scalar(optimize_clf, bounds=(1, 1500), method='bounded', options={'maxiter': 20, 'xatol': 1e-05})
        #nf = int(np.round(f.x))
	nf = ds.shape[1]
        #val_accs = optimize_clf(nf, optimize=2)
        #val_accs_per_chunk.append(val_accs)
        test_accs = optimize_clf(nf, optimize=0)
        test_accs_per_chunk.append(test_accs)
        nfs_all_chunks.append(nf)
    #return test_accs_per_chunk,0,nfs_all_chunks,0;
    corrs1, pvals1, dist_list1, nf_list_dists1 = dist_wrapper(ds,test_accs,0,0)
    return corrs1, pvals1;

def dist_wrapper(ds, test_accs, val_accs, val_chunks):
    dist_list_per_chunk=[]
    corrs_per_chunk=[]
    nf_list_per_chunk=[]
    pvals_per_chunk=[]
    i=-1 #counter for test_accs, val_accs & val_chunk (validation chunk)
    for chunk in chunk_num: #does LOOCV
        dist_list=[]
        corrs=[]
        nf_list=[]
        pvals=[]
        i+=1
        #val_chunk = val_chunks[i]
        for dist in dists:
            def optimize_dist(nf, optimize=1):
                dist_vec=[] #array with accuracies for each pair within each LOOVC fold
                def nf_select(nf):
                    #fselector = mvpa2.FixedNElementTailSelector(np.round(nf), tail='upper',mode='select', sort=False)
                    #sbfs = mvpa2.SensitivityBasedFeatureSelection(mvpa2.OneWayAnova(), fselector, enable_ca=['sensitivities'], auto_train=True)
                    if(optimize==1):
                        not_test_ds = ds[ds.chunks!=chunk]
                        train_ds = not_test_ds[not_test_ds.chunks!=val_chunk]
                        #sbfs.train(train_ds)
                        #ds2 = sbfs(not_test_ds) #optimize nf & include validation set for computing dists
                    elif(optimize==0):
                        train_ds = ds[ds.chunks!=chunk] #retrain with all data if not optimizing
                        #sbfs.train(train_ds)
                        #ds2 = sbfs(ds) #pick top features with training & use whole dataset for computing dists
                    return ds2;
                #ds2 = nf_select(nf)
                for y in range(0, len(pair_list2)):
                    def mask(y, ds):
                        stim_mask_train0 = (ds.targets==pair_list2[y][0])
                        stim_mask_train1 =(ds.targets==pair_list2[y][1])
                        ds_stim1 = ds[stim_mask_train0]
                        ds_stim2 = ds[stim_mask_train1]
                        return ds_stim1, ds_stim2;
                    ds_stim1, ds_stim2 = mask(y, ds)
                    dist_vec.append(distance_funcs(np.mean(ds_stim1,axis=0),np.mean(ds_stim2,axis=0),
                                    ds.samples,ds_stim1,ds_stim2, dist))
                if(optimize==1):
                    corr_test = spearmanr(val_accs,dist_vec)
                    #corr_test = pearsonr(val_accs[i],dist_vec)
                elif(optimize==0):
                    corr_test = spearmanr(test_accs,dist_vec)
                    #corr_test = pearsonr(test_accs[i],dist_vec)
                corr = corr_test[0]
                pval = corr_test[1]
                #print corr, ',', pval, 'distance:', dist, np.round(nf), ', features,', 'chunk', chunk
                if(optimize==1):
                    return 1-corr;
                elif(optimize==0):
                    return corr, pval, dist_vec;
            #if (dist=='mahalanobis')|(dist=='bhattacharya')|(dist=='var_info')|(dist=='mahab_noreg')|(dist=='mahab_diag')|(dist=='bhat_noreg')| (dist=='bhat_diag'):
                #corr_temp = np.nan
                #nf_temp = 50
                #while(np.isnan(corr_temp)==True):
                    #nf_temp = nf_temp - 5
                    #corr_temp = optimize_dist(nf_temp) #find an upper bound for naughty measures
                #ub = nf_temp
                #print ub
                #f = minimize_scalar(optimize_dist, bounds=(2, ub), method='bounded', options={'maxiter': 20, 'xatol': 1e-05})
            #else:
                #f = minimize_scalar(optimize_dist, bounds=(2, 1500), method='bounded', options={'maxiter': 20, 'xatol': 1e-05})
            #nf = int(np.round(f.x))
            nf = ds.shape[1]
            corr, pval, dist_vec = optimize_dist(nf,optimize=0)
            corrs.append(corr)
            pvals.append(pval)
            dist_list.append(dist_vec)
            nf_list.append(nf)
        corrs_per_chunk.append(corrs)
        pvals_per_chunk.append(pvals)
        dist_list_per_chunk.append(dist_list)
        nf_list_per_chunk.append(nf_list)
    return corrs_per_chunk, pvals_per_chunk, dist_list_per_chunk, nf_list_per_chunk, test_accs, val_accs, val_chunks;

subs = ['01','02','03','04','05','06','08','09','10','11','12','13','14','15']

#cwd1 = '/media/seb/Seagate Backup Plus Drive/odb_slim'
cwd1 = '/home/ucjtbob/odb_slim'
#cwd1 = '/home/sebastian/Documents/odb_slim'

job_num = int(sys.argv[1])-1
#plan is to make a table with all cv folds so that we can read off job number from there, code above needs to be modified; for loops need to be eliminated...
print job_num

job_path = sys.argv[2]
print job_path

sub6_chunks = [1,2,3,8,9,10,11,12,13,14,15,16]
job_table=[]
for sub_tmp in range(0,len(subs)):
        for fset in range(100): #fset is feature set (100 voxel regions)
                if (subs[sub_tmp]=='06'):
                        for chunk_tmp in range(0,len(sub6_chunks)):
                                job_table.append([subs[sub_tmp], fset, sub6_chunks[chunk_tmp]])
                else:
                        for chunk_tmp in range(0,16):
                                job_table.append([subs[sub_tmp], fset, chunk_tmp+1])

sub = job_table[job_num][0]
fset_num = job_table[job_num][1] #used to be a loop but now runs on cluster in parallel
chunk_num = [job_table[job_num][2]]
filename = sub + '_fset' + str(fset_num) + '_chunk' + str(chunk_num[0])

print sub

print np.array(job_table).shape

behav_file = 'sub'+sub+'_attr.txt'

bold_fname = os.path.join(cwd1, sub, 'betas_sub'+sub+'.nii.gz') #full functional timeseries (beta series)

attr_fname = os.path.join(cwd1, 'all_attr', behav_file) #codes stimuli number and run number
attr = mvpa2.SampleAttributes(attr_fname) #loads attributes into pymvpa

ds = mvpa2.fmri_dataset(bold_fname, targets=attr.targets, chunks=attr.chunks)

ds = mvpa2.remove_nonfinite_features(ds)
ds = mvpa2.remove_invariant_features(ds)

#this basically breaks up the brain into 100 different areas (to parallelize the searchlight)
try:
	ds = ds[:,fset_num*1000:(fset_num*1000)+1000]
except:
	ds = ds[:,fset_num*1000:]

stimuli = []
for i in range(0,54):
    stimuli.append(ds.uniquetargets[i])

#create all possible pairs for confusion matrix
pair_list = list(itertools.combinations(range(len(stimuli)), 2))
pair_list2 = []
for x in range(0, len(pair_list)):
    pair_list2.append([stimuli[pair_list[x][0]],stimuli[pair_list[x][1]]])


# enable debug output for searchlight call
#if __debug__:
mvpa2.debug.active += ["SLC"]

sl = mvpa2.sphere_searchlight(clf_wrapper, radius=3, space='voxel_indices',nproc=1)
ds = ds.copy(deep=False,sa=['targets', 'chunks'],fa=['voxel_indices'],a=['mapper'])
sl_map = sl(ds)

data_file1 = os.path.join(job_path, filename)

file_path_full = open(data_file1, 'w')
pickle.dump(sl_map, file_path_full)

#jobs up to 22000 now... (12 hours each?)
