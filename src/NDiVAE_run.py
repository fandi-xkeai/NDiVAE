import argparse
import logging
import numpy as np
import torch
import pandas as pd
import pyro
import utils
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from NDiVAE_gpu import NDiVAE_Model
import random

logging.getLogger("pyro").setLevel(logging.DEBUG)
logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    
def get_args():
    parser = argparse.ArgumentParser(description="Run NDiVAE Training and Evaluation")

    parser.add_argument("--model", type=str, default="NDiVAE")
    parser.add_argument("--dataset", type=str, default="BC",
                        help="dataset")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--learning_rate_decay", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)    
    parser.add_argument("--normy", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=256)#64
    parser.add_argument("--latent_dim_Zt", type=int, default=2)#5
    parser.add_argument("--latent_dim_Zc", type=int, default=2)#5
    parser.add_argument("--latent_dim_Zy", type=int, default=2)#5
    parser.add_argument("--latent_dim_u", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)  
    parser.add_argument("--lambda_elbo", type=float, default=1, help="Weight for ELBO loss")
    parser.add_argument("--lambda_y", type=float, default=1, help="Weight for MSE_y loss")
    parser.add_argument("--lambda_t", type=float, default=1, help="Weight for BCE_t loss")
    parser.add_argument("--lambda_wass", type=float, default=1, help="Weight for Wasserstein loss")
    parser.add_argument("--lambda_reg_zt", type=float, default=0.01, help="Weight for zt t BCE loss")
    parser.add_argument("--lambda_reg_zy", type=float, default=0.01, help="Weight for zy y L2 loss")
    return parser.parse_args()


def compute_ATE_and_PEHE(
    y1_hat: np.ndarray, y0_hat: np.ndarray,
    y1_true: np.ndarray, y0_true: np.ndarray
):
    diff_true = y1_true - y0_true
    diff_hat  = y1_hat  - y0_hat

    ATE_true = np.mean(diff_true)
    ATE_hat  = np.mean(diff_hat)
    ATE_err  = abs(ATE_hat - ATE_true)
    PEHE     = np.sqrt(np.mean((diff_hat - diff_true) ** 2))
    return ATE_true, ATE_hat, ATE_err, PEHE


def run_train_and_infer(args):
    pyro.enable_validation(__debug__)
    pyro.clear_param_store()
    device = torch.device("cuda" if args.cuda else "cpu")

    data = utils.load_data(args)
    (
        trainA, trainX, trainT, POTrain,
        valA,   valX,   valT,   POVal,
        testA,  testX,  testT,  POTest,
        train_t1z, train_t0z, train_t0z0, train_tz, train_tz0,
        val_t1z,   val_t0z,   val_t0z0,   val_tz,   val_tz0,
        test_t1z,  test_t0z,  test_t0z0,  test_tz,  test_tz0,
        train_zt,  train_zc,  train_zy, train_u,
        val_zt,    val_zc,    val_zy,   val_u,
        test_zt,   test_zc,   test_zy,  test_u,
    ) = data
    

    def to_tensor_on_device(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(device)
        elif torch.is_tensor(x):
            return x.float().to(device)
        else:
            raise ValueError(f"Unsupported data type: {type(x)}")

    train_zt = to_tensor_on_device(train_zt)
    train_zc = to_tensor_on_device(train_zc)
    train_zy = to_tensor_on_device(train_zy)


    val_zt = to_tensor_on_device(val_zt)
    val_zc = to_tensor_on_device(val_zc)
    val_zy = to_tensor_on_device(val_zy)


    test_zt = to_tensor_on_device(test_zt)
    test_zc = to_tensor_on_device(test_zc)
    test_zy = to_tensor_on_device(test_zy)

    train_u = to_tensor_on_device(train_u)  # (n_train, latent_dim_u)
    val_u   = to_tensor_on_device(val_u)    # (n_val,   latent_dim_u)
    test_u  = to_tensor_on_device(test_u)   # (n_test,  latent_dim_u)

    true_z_dicts = {
        "train": {
            "zt": train_zt.cpu().numpy().squeeze(),
            "zc": train_zc.cpu().numpy().squeeze(),
            "zy": train_zy.cpu().numpy().squeeze(),
            
        },
        "val": {
            "zt": val_zt.cpu().numpy().squeeze(),
            "zc": val_zc.cpu().numpy().squeeze(),
            "zy": val_zy.cpu().numpy().squeeze(),
           
        },
        "test": {
            "zt": test_zt.cpu().numpy().squeeze(),
            "zc": test_zc.cpu().numpy().squeeze(),
            "zy": test_zy.cpu().numpy().squeeze(),
           
        },
    }

    
    trainX  = to_tensor_on_device(trainX)            
    trainT  = to_tensor_on_device(trainT).float()     
    POTrain = to_tensor_on_device(POTrain).float()    

    valX   = to_tensor_on_device(valX)                
    valT   = to_tensor_on_device(valT).float()        
    POVal  = to_tensor_on_device(POVal).float()       

    testX  = to_tensor_on_device(testX)               
    testT  = to_tensor_on_device(testT).float()       
    POTest = to_tensor_on_device(POTest).float()      

    trainA = to_tensor_on_device(trainA)              
    valA   = to_tensor_on_device(valA)                
    testA  = to_tensor_on_device(testA)              

    if args.normy:
        ym, ys = torch.mean(POTrain), torch.std(POTrain)
        ys = ys if ys > 0 else torch.tensor(1.0, device=POTrain.device)

        def normalize(y): return (y - ym) / ys
        def recover(yn):  return yn * ys + ym

        POTrain_norm = normalize(POTrain)
        POVal_norm   = normalize(POVal)
        POTest_norm  = normalize(POTest)
    else:
        POTrain_norm = POTrain
        POVal_norm   = POVal
        POTest_norm  = POTest

        recover = lambda y: y
  
    true_y_dicts = {
        "train": POTrain.cpu().numpy().squeeze(),
        "val":   POVal.cpu().numpy().squeeze(),
        "test":  POTest.cpu().numpy().squeeze(),
    }

    model = NDiVAE_Model(
        feature_dim=trainX.shape[1],   
        latent_dim_Zt=args.latent_dim_Zt,
        latent_dim_Zc=args.latent_dim_Zc,
        latent_dim_Zy=args.latent_dim_Zy,
        latent_dim_u=args.latent_dim_u,  
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)

    model.fit(
        trainX, train_u, trainT, POTrain_norm, trainA,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        learning_rate_decay=args.learning_rate_decay,
        weight_decay=args.weight_decay,
        lambda_wass =args.lambda_wass,
        lambda_elbo=args.lambda_elbo,
        lambda_y=args.lambda_y,
        lambda_t=args.lambda_t,
        lambda_reg_zt=args.lambda_reg_zt,
        lambda_reg_zy=args.lambda_reg_zy
    )

    est_z_dicts = {}
    est_y_dicts = {"train": {}, "val": {}, "test": {}}
    est_t_dicts = {"train": {}, "val": {}, "test": {}}

    true_PO = {
        "train": {
            "t1z": train_t1z, "t0z": train_t0z, "t0z0": train_t0z0,
            "tz": train_tz,  "tz0": train_tz0
        },
        "val": {
            "t1z": val_t1z,  "t0z": val_t0z,  "t0z0": val_t0z0,
            "tz":  val_tz,   "tz0": val_tz0
        },
        "test": {
            "t1z": test_t1z, "t0z": test_t0z, "t0z0": test_t0z0,
            "tz":  test_tz,  "tz0": test_tz0
        }
    }

    for split in ["train", "val", "test"]:
        x_split, u_split, t_orig, y_norm, adj_split = (
            (trainX, train_u, trainT, POTrain_norm, trainA) if split=="train" else
            (valX,   val_u,   valT,   POVal_norm,   valA  ) if split=="val"   else
            (testX,  test_u,  testT,  POTest_norm,  testA )
        )

        N = x_split.size(0)
        deg = adj_split.sum(dim=1, keepdim=True).clamp(min=1.0)
        agg_t_default = (adj_split @ t_orig.unsqueeze(1)) / deg
        agg_t_default = agg_t_default.squeeze(1)

        z_dict = model.guide.get_latent_z(x_split, u= u_split)
        est_z_dicts[split] = {
            "zt": z_dict["zt"].cpu().detach().numpy().squeeze(),
            "zc": z_dict["zc"].cpu().detach().numpy().squeeze(),
            "zy": z_dict["zy"].cpu().detach().numpy().squeeze(),
           
        }

        ones_tensor  = torch.ones(N, device=device)
        zeros_tensor = torch.zeros(N, device=device)
        scenario_settings = {
            "t1z":  (ones_tensor,        agg_t_default),
            "t0z":  (zeros_tensor,       agg_t_default),
            "t0z0": (zeros_tensor,       zeros_tensor),
            "tz":   (t_orig,             agg_t_default),
            "tz0":  (t_orig,             zeros_tensor),
        }

        est_y_dicts[split] = {}
        zc_tensor = z_dict["zc"].squeeze(-1)  
        zy_tensor = z_dict["zy"].squeeze(-1)  
        for name, (t_override, agg_t_override) in scenario_settings.items():
            yhat_norm  = model.model.predict_y(
                zc_tensor,
                zy_tensor,
                t_override,
                adj_split,
                agg_t_override=agg_t_override
            )  
            y_hat_tensor = recover(yhat_norm)
            est_y_dicts[split][name] = y_hat_tensor.cpu().detach().numpy().squeeze()

        zt_tensor = z_dict["zt"].squeeze(-1) 
        zc_tensor = z_dict["zc"].squeeze(-1) 
        t_hat_tensor = model.model.predict_t(
            zt_tensor, zc_tensor, adj_split
        )  
        t_hat_prob = torch.sigmoid(t_hat_tensor) 
        est_t_dicts[split] = t_hat_prob.cpu().detach().numpy().squeeze()

    return est_z_dicts, true_z_dicts, est_y_dicts, true_PO, true_y_dicts, est_t_dicts, {
        "train": trainT.cpu().numpy().squeeze(),
        "val":   valT.cpu().numpy().squeeze(),
        "test":  testT.cpu().numpy().squeeze()
    }


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    all_results = []  

    expID_list = [0,1,2,3,4]
    for expID in expID_list:
        
        args.expID = expID
        print(f"\n=== Processing expID = {expID} ===")

        est_z_dicts, true_z_dicts, est_y_dicts, true_PO, true_y_dicts, est_t_dicts, true_t_dicts = run_train_and_infer(args)

        
        for split in ["train", "val", "test"]:
            true_t1z   = true_PO[split]["t1z"].squeeze()
            true_t0z   = true_PO[split]["t0z"].squeeze()
            true_t0z0  = true_PO[split]["t0z0"].squeeze()
            true_tz    = true_PO[split]["tz"].squeeze()
            true_tz0   = true_PO[split]["tz0"].squeeze()

            est_t1z    = est_y_dicts[split]["t1z"].squeeze()
            est_t0z    = est_y_dicts[split]["t0z"].squeeze()
            est_t0z0   = est_y_dicts[split]["t0z0"].squeeze()
            est_tz     = est_y_dicts[split]["tz"].squeeze()
            est_tz0    = est_y_dicts[split]["tz0"].squeeze()

            ME_true, ME_hat, ME_err, ME_PEHE = compute_ATE_and_PEHE(
                y1_hat=est_t1z,   y0_hat=est_t0z,
                y1_true=true_t1z, y0_true=true_t0z
            )
            SE_true, SE_hat, SE_err, SE_PEHE = compute_ATE_and_PEHE(
                y1_hat=est_tz,    y0_hat=est_tz0,
                y1_true=true_tz,  y0_true=true_tz0
            )
            TE_true, TE_hat, TE_err, TE_PEHE = compute_ATE_and_PEHE(
                y1_hat=est_t1z,    y0_hat=est_t0z0,
                y1_true=true_t1z,  y0_true=true_t0z0
            )


            all_results.append({
                "expID":     expID,
                "split":     split,
                "ME_err":    ME_err,
                "ME_PEHE":   ME_PEHE,
                "SE_err":    SE_err,
                "SE_PEHE":   SE_PEHE
            })

            print(
                f"  [Effects] exp{expID}, {split}: "
                f"ME_err={ME_err:.4f}, ME_PEHE={ME_PEHE:.4f}; "
                f"SE_err={SE_err:.4f}, SE_PEHE={SE_PEHE:.4f}; "
                f"TE_err={TE_err:.4f}, TE_PEHE={TE_PEHE:.4f}; "
            )
    
