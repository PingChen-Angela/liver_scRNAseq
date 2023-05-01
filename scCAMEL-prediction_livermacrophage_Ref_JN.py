import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.utils.data as data_utils
from matplotlib import cm
import numpy as np
import pandas as pd
import pickle as pickle
from scipy.spatial.distance import cdist, pdist, squareform
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit  
from collections import defaultdict
from sklearn import preprocessing
import matplotlib.patches as mpatches
import torch.nn.functional as F
import math
import urllib.request
import os.path
from scipy.io import loadmat
from math import floor
import anndata
torch.manual_seed(1) 
import scCAMEL as scm
from scCAMEL import CamelPrefiltering
from scCAMEL import CamelSwapline
from scCAMEL import CamelEvo

screfall=anndata.read("LiverMacrophage_474cells_Ref2023-01-16_MergeCluster_35epch.h5ad")
scref=screfall
scref.X=scref.X.todense()
scref=scm.CamelPrefiltering.DataScaling(scref)
scref.var['Filter1']=[True]*scref.var.shape[0]
scref=CamelPrefiltering.SelectFeatures(datax=scref, clustername='Cluster',
                                       methodname='wilcoxon', numbergenes=1000, folderchange=1.5)

scref =scm.CamelPrefiltering.LabelGene_Scaling(datax=scref,TPTT=100000,mprotogruop=scref.obs["Cluster"].values,commongene=None,
                                               sharedMVgenes=None,std_scaling=True, 
                                               tftable="FantomTF2CLUSTER_human_official.txt",
                                               learninggroup="train")

net=scm.CamelPrefiltering.NNclassifer(
   datax=scref,
    epochNum=200,
    learningRate=0.03,
    verbose=0,
    optimizerMmentum=0.8,
    dropout=0.3,
    #imizer__nesterov=True,
    )

ax=scm.CamelPrefiltering.AccuracyPlot( nnModel=net, accCutoff=0.85,
                 Xlow=-1, Ylow=0.0, Yhigh=1)
plt.savefig("upload_CurvePlot_learningAccuracy.pdf",bbox_inches='tight')

net=scm.CamelPrefiltering.NNclassifer(
   datax=scref,
    epochNum=35,
    learningRate=0.03,
    verbose=0,
    optimizerMmentum=0.8,
    dropout=0.3,
    #imizer__nesterov=True,
    )

scref=scm.CamelSwapline.addcolor(datax=scref,clustername="Cluster", colorcode="color")
clist=[]
for item in scref.obs["Cluster"]:
    if item=="LM1":
        clist.append("#ABD9E9")
    elif item=="LM2-C2":
        clist.append("#2C7BB6")
    elif item=="LM2-C1":
        clist.append("purple")
    elif item=="LM3":
        clist.append("#FDAE61")
    elif item=="LM4":
        clist.append("#D7191C")
scref.obs["color"]=clist

scref.uns['refcolor_dict']={'LM2-C1': [44, 123, 182],
                            'LM2-C2': [143, 0, 255],
                            "LM3": [253,174,97],
                            "LM4": [215,25,28],
                            'LM1': [171, 217, 233],
                           }

scref.uns["mwanted_order"] =list(sort(list(set(scref.obs["Cluster"]))))
scref=scm.CamelSwapline.prediction(datax=scref, mcolor_dict=scref.uns["refcolor_dict"],
                                   net=net,learninggroup="train",radarplot=True,
                                   fontsizeValue=18,ncolnm=3, bbValue=(1.2, 1.05))

plt.savefig("upload_RadarPlot_Merged_cluster.pdf",bbox_inches='tight')




