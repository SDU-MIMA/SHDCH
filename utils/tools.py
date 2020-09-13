# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:54:23 2020

@author: BigFly
"""

import scipy.io as sio
import numpy as np
import os
# from curve import Curve, creat_fig

def get_prec_rec_map_prc(qh, rh, ql, rl):
    ql = ql[:, 8: len(ql[0])]
    rl = rl[:, 8: len(rl[0])]
    q_n = qh.shape[0]
    r_n = rh.shape[0]
    bit_n = qh.shape[1] 
    
    hamdist = ((bit_n*1.0 - np.dot(qh, rh.T)) / 2).astype(np.float16)
    rank = np.argsort(hamdist, 1)
    
    sim = (np.dot(ql, rl.T) >0).astype(np.int)
    sim_num_of_query = np.sum(sim, 1)
    sim = np.array([s[rk] for s,rk in zip(sim,rank)], np.int)
    
    #prc
    pointnum = 1000
    step = r_n//pointnum
    prc = np.zeros((2,pointnum))
    for t in range( pointnum):
        topn = (t+1) *step
        s = np.sum( sim[:, 0: topn ], 1 )
        prc[1][t] = np.mean( s / topn )
        prc[0][t] = np.mean( s /sim_num_of_query )
#    Curve( [prc]).draw()
        
    # p,r,map
    precs = np.zeros((q_n, bit_n + 1))
    recs = np.zeros((q_n, bit_n + 1))
    mAPs = np.zeros((q_n, bit_n + 1))   

    for i in range(q_n):
        imatch = sim[i]
        all_sim_num = np.sum(imatch)
        counts = np.bincount(hamdist[i, :].astype(np.int64)) #0~maxd 次数
        for r in range(bit_n + 1):
            if r >= len(counts):
                precs[i, r] = precs[i, r - 1]
                recs[i, r] = recs[i, r - 1]
                mAPs[i, r] = mAPs[i, r - 1]
                continue
            all_num = np.sum(counts[0: r + 1])
            if all_num != 0:
                match_num = np.sum(imatch[0:all_num])
                precs[i, r] = np.float(match_num) / all_num
                recs[i, r] = np.float(match_num) / all_sim_num
                rel = match_num
                Lx = np.cumsum(imatch[0:all_num])
                Px = Lx.astype(float) / np.arange(1, all_num + 1, 1)
                if rel != 0:
                    mAPs[i, r] = np.sum(Px * imatch[0:all_num]) / rel
    return np.mean(np.array(precs), 0), \
           np.mean(np.array(recs), 0), \
           np.mean(np.array(mAPs), 0), prc

           
if __name__=='__main__':
    bit = 16
    
    label = sio.loadmat(r'label\label_cifar10')['label'].astype(np.int16)
    datanum = label.shape[0]
    queryidx = slice(0,1000)
    dbidx = slice(1000,datanum)
    ql,rl = label[queryidx], label[dbidx]
    
    for data in os.listdir("data327"):
        hashcode = sio.loadmat("data327\\"+data)['U_logical_DB'].astype(np.int16)
        qh, rh = hashcode[queryidx], hashcode[dbidx]
        prec, rec, mAP ,prc = get_prec_rec_map_prc(qh, rh, ql, rl)
        print(data, mAP[-1])
        np.save( "data327\\"+data[:-4]+ "_data",  {"prec": prec,\
                          "rec": rec,\
                          "mAP": mAP,\
                          "prc": np.array(prc)  } )

#    hashcode = sio.loadmat(r"hashcodes\RESULT\cifar10\COS\CIF_COS_16")['U_logical_DB'].astype(np.int16)
#    hashcode[hashcode==0]=-1
#    qh, rh = hashcode[queryidx], hashcode[dbidx]
###    
##    
##    qh = sio.loadmat(r"hashcodes\RESULT\TEACH2\qBX_16")['qBX_16'].astype(np.int16)
##    rh = sio.loadmat(r"hashcodes\RESULT\TEACH2\AllBY_16")['AllBY_16'].astype(np.int16)
##    ql = sio.loadmat(r"hashcodes\RESULT\TEACH2\LTest")['LTest'].astype(np.int16)
##    rl = sio.loadmat(r"hashcodes\RESULT\TEACH2\LTrain")['LTrain'].astype(np.int16)
##    prec,rec,mAP= pr_curve_wrong(qh, rh, ql, rl)
#    
#    #
#    #m0,d0 = pr_curve_wrong(qh, rh, ql, rl)
#    #recs, precs = pr_curve(qh, rh, ql, rl)
#    
#    #prc = Curve( recs,[precs],[], "prc", "rec","prec",212)
#    #creat_fig([prc], "temp")
#    p,r,m = get_prec_rec_map(qh, rh, ql, rl)
#    print(m[-1])
#    p,r,m ,prc= get_prec_rec_map_prc(qh, rh, ql, rl)
#    print(m[-1])
##    prc = pr_curve(qh, rh, ql, rl)
##    
##    Curve([prc] ).draw()
#    
#
