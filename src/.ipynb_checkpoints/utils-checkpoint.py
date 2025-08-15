import numpy as np
import pickle as pkl
import scipy.io as sio
import scipy.sparse as sp

import torch
import torch.nn.functional as F


def sparse_mx_to_torch_sparse_tensor(sparse_mx,cuda=False):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    if cuda:
        sparse_tensor = sparse_tensor.cuda()
    return sparse_tensor


def normalize(mx):
	"""Row-normalize sparse matrix"""
	rowsum = np.array(mx.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx


def dataTransform(data,cuda):
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    A, X, T, PO = data['network'],data['features'],data["T"],data["PO"]
    X = Tensor(normalize(X))
    A,T,PO = Tensor(A),Tensor(T),Tensor(PO)

    return A, X, T, PO


def load_data(args):
    print ("================================Dataset================================")
    print ("Model:{}, Dataset:{}, expID:{}".format(args.model,args.dataset,args.expID))
    dataset,expID,cuda = args.dataset,args.expID,args.cuda
    
    if dataset == "BC":
        file = "./NDiVAE/BC/simulation/"+str(dataset)+"_expID_"+str(expID)+".pkl"
    if dataset == "Flickr":
        file = "./NDiVAE/Flickr/simulation/"+str(dataset)+"_expID_"+str(expID)+".pkl"
    if dataset == "BC_hete":
        file = "./NDiVAE/BC_hete/simulation/"+str(dataset)+"_expID_"+str(expID)+".pkl"
    if dataset == "Flickr_hete":
        file = "./NDiVAE/Flickr_hete/simulation/"+str(dataset)+"_expID_"+str(expID)+".pkl"

    with open(file,"rb") as f:
        data = pkl.load(f)
    dataTrain,dataVal,dataTest = data["train"],data["val"],data["test"]

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    trainA, trainX, trainT, POTrain = dataTransform(dataTrain,cuda)
    valA, valX, valT, POVal = dataTransform(dataVal,cuda)
    testA, testX, testT, POTest = dataTransform(dataTest,cuda)

    
    # Train set
    train_t1z = dataTrain["train_t1z"]
    train_t0z = dataTrain["train_t0z"]
    train_t0z0 = dataTrain["train_t0z0"]
    train_tz = dataTrain["train_tz"]
    train_tz0 = dataTrain["train_tz0"]
    train_zt = dataTrain["train_zt"]
    train_zc = dataTrain["train_zc"]
    train_zy = dataTrain["train_zy"]
    train_u = dataTrain["u"]


    # Validation set
    val_t1z = dataVal["val_t1z"]
    val_t0z = dataVal["val_t0z"]
    val_t0z0 = dataVal["val_t0z0"]
    val_tz = dataVal["val_tz"]
    val_tz0 = dataVal["val_tz0"]
    val_zt = dataVal["val_zt"]
    val_zc = dataVal["val_zc"]
    val_zy = dataVal["val_zy"]
    val_u = dataVal["u"]

    # Test set
    test_t1z = dataTest["test_t1z"]
    test_t0z = dataTest["test_t0z"]
    test_t0z0 = dataTest["test_t0z0"]
    test_tz = dataTest["test_tz"]
    test_tz0 = dataTest["test_tz0"]
    test_zt = dataTest["test_zt"]
    test_zc = dataTest["test_zc"]
    test_zy = dataTest["test_zy"]
    test_u = dataTest["u"]




    return trainA, trainX, trainT, POTrain,valA, valX, valT, POVal,\
        testA, testX, testT, POTest,\
        train_t1z, train_t0z, train_t0z0, train_tz, train_tz0,\
        val_t1z, val_t0z, val_t0z0, val_tz, val_tz0,\
        test_t1z, test_t0z, test_t0z0, test_tz, test_tz0,\
        train_zt, train_zc, train_zy,train_u, \
        val_zt, val_zc, val_zy,val_u, \
        test_zt, test_zc, test_zy,  test_u
    



def PO_normalize(normy,base,PO,cfPO):
    """Normalize PO"""
    if  normy:
        ym, ys = torch.mean(base), torch.std(base)
        YF,YCF = (PO - ym) / ys, (cfPO - ym) / ys
    else:
        YF,YCF =  PO,cfPO
    
    return YF,YCF 

def PO_normalize_recover(normy,base,nPO):
    
    if normy:
        ym, ys = torch.mean(base), torch.std(base)
        pred_PO = (nPO * ys + ym)
    else:
        pred_PO = nPO
    
    return pred_PO




"""
The following codes are originally by Ruocheng Guo for the WSDM'20 paper
@inproceedings{guo2020learning,
  title={Learning Individual Causal Effects from Networked Observational Data},
  author={Guo, Ruocheng and Li, Jundong and Liu, Huan},
  booktitle={Proceedings of the 13th International Conference on Web Search and Data Mining},
  pages={232--240},
  year={2020}
}
https://github.com/rguo12/network-deconfounder-wsdm20
"""

import torch
import torch.nn.functional as F

def wasserstein(x, y, p=0.5, lam=10, its=10, sq=False, backpropT=False, cuda=False):
    """Return Wasserstein distance between x and y"""
    
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    # x = x.squeeze().to(device)
    # y = y.squeeze().to(device)

    nx = x.shape[0]
    ny = y.shape[0]

    M = pdist(x, y).to(device)

    # Estimate lambda and delta
    M_mean = torch.mean(M)
    M_drop = F.dropout(M, 10.0 / (nx * ny))
    delta = torch.max(M_drop).detach()
    eff_lam = (lam / M_mean).detach()

    # Compute new distance matrix
    Mt = M
    row = delta * torch.ones((1, ny), device=device)
    col = torch.cat([delta * torch.ones((nx, 1), device=device), torch.zeros((1, 1), device=device)], dim=0)
    Mt = torch.cat([M, row], dim=0)
    Mt = torch.cat([Mt, col], dim=1)

    # Compute marginals
    a = torch.cat([p * torch.ones((nx, 1), device=device) / nx,
                   (1 - p) * torch.ones((1, 1), device=device)], dim=0)
    b = torch.cat([(1 - p) * torch.ones((ny, 1), device=device) / ny,
                   p * torch.ones((1, 1), device=device)], dim=0)

    # Compute kernel
    Mlam = eff_lam * Mt
    temp_term = torch.tensor(1e-6, device=device)
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K / a

    u = a
    for i in range(its):
        u = 1.0 / (ainvK.matmul(b / (u.t().matmul(K)).t()))

    v = b / (u.t().matmul(K)).t()
    upper_t = u * (v.t() * K).detach()
    E = upper_t * Mt
    D = 2 * torch.sum(E)

    return D, Mlam

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute matrix of pairwise distances."""

    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)

    if norm == 2.:
        norms_1 = torch.sum(sample_1 ** 2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2 ** 2, dim=1, keepdim=True)
        norms = norms_1.expand(n_1, n_2) + norms_2.t().expand(n_1, n_2)
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2)
        return (eps + inner) ** (1. / norm)




def MI(x,y,z,N):
    if x == 0:
        return torch.FloatTensor([0])
    else:
        return (x/N)*torch.log2((N*x)/(y*z))

def NMI(set1,set2,threshold=0.5):
    set1 =  torch.FloatTensor([(set1 >= threshold).sum(),(set1 < threshold).sum()])
    set2 =  torch.FloatTensor([(set2 >= threshold).sum(),(set2 < threshold).sum()])
    set1 = set1.reshape(1,-1)
    set2 = set2.reshape(1,-1)
    res = torch.cat((set1,set2),0).T
    N = torch.sum(torch.sum(res))
    NW = torch.sum(res,1)
    NC  = torch.sum(res,0)
    HC = -((NC[0]/N)*torch.log2(NC[0]/N)+(NC[1]/N)*torch.log2(NC[1]/N))
    HW = -((NW[0]/N)*torch.log2(NW[0]/N)+(NW[1]/N)*torch.log2(NW[1]/N))
    IF = MI(res[0][0],NW[0],NC[0],N)+MI(res[0][1],NW[0],NC[1],N)+MI(res[1][0],NW[1],NC[0],N)+MI(res[1][1],NW[1],NC[1],N)
    return (IF/torch.sqrt(HC*HW)).cuda()

def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val**2




