import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO
from pyro.infer.util import torch_item
from pyro.nn import PyroModule
import utils
from layers import GraphConvolution


logger = logging.getLogger(__name__)


class FullyConnected(nn.Sequential):
    def __init__(self, sizes, final_activation=None):
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ELU())
        layers.pop(-1)
        if final_activation is not None:
            layers.append(final_activation)
        super().__init__(*layers)

    def append(self, layer):
        assert isinstance(layer, nn.Module)
        self.add_module(str(len(self)), layer)

        
class DistributionNet(nn.Module):
    @staticmethod
    def get_class(dtype):
        for cls in DistributionNet.__subclasses__():
            if cls.__name__.lower() == dtype + "net":
                return cls
        raise ValueError("dtype not supported: {}".format(dtype))

        
class DiagNormalNet(nn.Module):
    def __init__(self, sizes):
        assert len(sizes) >= 2
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [self.dim * 2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., :self.dim].clamp(min=-1e2, max=1e2)
        scale = nn.functional.softplus(loc_scale[..., self.dim:]).add(1e-3).clamp(max=1e2)
        return loc, scale

    
class DependencyDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, zc_embed, t, agg_t):
        x = torch.cat([zc_embed, t, agg_t], dim=-1)
        return self.net(x)


class Guide(PyroModule):
    def __init__(self, config):
        super().__init__()
        self.latent_dims = {
            "zt": config["latent_dim_Zt"],
            "zc": config["latent_dim_Zc"],
            "zy": config["latent_dim_Zy"],
        }
        in_dim = config["feature_dim"] + config["latent_dim_u"]
        hid, layers = config["hidden_dim"], config["num_layers"]
        self.encoders = nn.ModuleDict({
             name: FullyConnected(
                [in_dim] + [hid] * (layers - 1),
                 final_activation=nn.ELU()
             )
             for name in self.latent_dims
        })
        self.latent_nns = nn.ModuleDict({
            name: DiagNormalNet([hid, dim])
            for name, dim in self.latent_dims.items()
        })

    def forward(self, x, u=None, adj=None, size=None):
        if size is None:
            size = x.size(0)
        x_, u_ = x.float(), u.float()
        inp = torch.cat([x_, u_], dim=1)

        with pyro.plate("data", size, subsample_size=x.size(0)):
            for name in self.latent_dims:
                h = self.encoders[name](inp)
                loc, scale = self.latent_nns[name](h)
                pyro.sample(name, dist.Normal(loc, scale).to_event(1))

    def get_latent_z(self, x, u=None):
        x_, u_ = x.float(), u.float()
        inp = torch.cat([x_, u_], dim=1)
        z_dict = {}

        with pyro.plate("data", x.size(0)):
            for name in self.latent_dims:
                h = self.encoders[name](inp)
                loc, scale = self.latent_nns[name](h)
                z_dict[name] = pyro.sample(name, dist.Normal(loc, scale).to_event(1))
        return z_dict


class TraceCausalEffect_ELBO(Trace_ELBO):
    def _differentiable_loss_particle(self, model_trace, guide_trace):
        blocked_names = [
            name for name, site in guide_trace.nodes.items()
            if site["type"] == "sample" and site["is_observed"]
        ]
        blocked_guide_trace = guide_trace.copy()
        for name in blocked_names:
            del blocked_guide_trace.nodes[name]
        loss, surrogate_loss = super()._differentiable_loss_particle(
            model_trace, blocked_guide_trace
        )
        for name in blocked_names:
            log_q = guide_trace.nodes[name]["log_prob_sum"]
            loss = loss - torch_item(log_q)
            surrogate_loss = surrogate_loss - log_q
        return loss, surrogate_loss

    @torch.no_grad()
    def loss(self, model, guide, *args, **kwargs):
        return torch_item(self.differentiable_loss(model, guide, *args, **kwargs))


class GCN(nn.Module):
    def __init__(self, nfeat, nclass):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nclass)

    def forward(self, x, adj):
        num = adj.shape[0]
        I = torch.eye(num, device=x.device)
        h = self.gc1(x, adj + I)
        h = F.relu(h)
        return h
    
    
class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.predict1 = nn.Linear(input_size, hidden_size1)
        self.predict2 = nn.Linear(hidden_size1, hidden_size2)
        self.predict3 = nn.Linear(hidden_size2, output_size)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.predict1(x)
        x = self.act(x)
        x = self.predict2(x)
        x = self.act(x)
        x = self.predict3(x)
        return x


class Model(PyroModule):
    def __init__(self, config):
        super().__init__()
        self.latent_dims = {
            "zt": config["latent_dim_Zt"],
            "zc": config["latent_dim_Zc"],
            "zy": config["latent_dim_Zy"],

        }
        hid, layers = config["hidden_dim"], config["num_layers"]
        self.prior_encoders = nn.ModuleDict({
            name: FullyConnected(
                [config["latent_dim_u"]] + [hid] * (layers - 1),
                final_activation=nn.ELU()
            )
            for name in self.latent_dims
        })
        self.latent_nns = nn.ModuleDict({
            name: DiagNormalNet([hid, dim]) for name, dim in self.latent_dims.items()
        })
        total_z = sum(self.latent_dims.values())
        self.x_nn = DiagNormalNet([total_z] + [hid] * layers + [config["feature_dim"]])

        self.gcn_zc = GCN(config["latent_dim_Zc"], config["latent_dim_Zc"])
        self.mlp_zc = FullyConnected([config["latent_dim_Zc"] + config["latent_dim_Zc"]] + [config["hidden_dim"]]+[config["hidden_dim"]//2]+[config["latent_dim_Zc"]],
                final_activation=nn.ELU())
        
        self.gcn_zy = GCN(config["latent_dim_Zy"], config["latent_dim_Zy"])
        self.mlp_zy = FullyConnected([config["latent_dim_Zy"] + config["latent_dim_Zy"]] + [config["hidden_dim"]] +[config["hidden_dim"]//2]+[config["latent_dim_Zy"]],
                final_activation=nn.ELU())

        self.gcn_zt = GCN(config["latent_dim_Zt"],config["latent_dim_Zt"])
        self.mlp_zt = FullyConnected([config["latent_dim_Zt"] + config["latent_dim_Zt"]] + [config["hidden_dim"]] +[config["hidden_dim"]//2]+[config["latent_dim_Zt"]],
                final_activation=nn.ELU())
        
        inp_y = config["latent_dim_Zc"] + config["latent_dim_Zy"] + 1
        self.predictor_y0 = Predictor(inp_y,  hid,hid//2, 1)
        self.predictor_y1 = Predictor(inp_y, hid,hid//2, 1)
        inp_t = config["latent_dim_Zc"] + config["latent_dim_Zt"]
        self.predictor_t = Predictor(inp_t, hid, hid//2, 1)

    def forward(self, x, u=None, adj=None, size=None):
        if size is None:
            size = x.size(0)
        ux = u.float()  
        with pyro.plate("data", size):
            zs = {}
            for name in self.latent_dims:
                h = self.prior_encoders[name](ux)
                loc, scale = self.latent_nns[name](h)
                
                z = pyro.sample(name, dist.Normal(loc, scale).to_event(1))
                zs[name] = z

            zcat = torch.cat([zs["zt"], zs["zc"], zs["zy"]], dim=1)
            loc_x, raw_scale_x = self.x_nn(zcat)
                        
            scale_x = raw_scale_x.clamp(min=0.4)
            pyro.sample("x_obs", dist.Normal(loc_x, scale_x).to_event(1), obs=x)

    def predict_y(self, zc_tensor, zy_tensor, t_tensor, adj, agg_t_override=None):
        N = adj.size(0)
        device = adj.device

        h_gcn_zc = self.gcn_zc(zc_tensor, adj)  
        h_gcn_zy = self.gcn_zy(zy_tensor, adj)  

        h2_zc = self.mlp_zc(torch.cat([h_gcn_zc, zc_tensor], dim=1))  
        h2_zy = self.mlp_zy(torch.cat([h_gcn_zy, zy_tensor], dim=1))  

        if agg_t_override is None:
            deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
            agg_t = (adj @ t_tensor.unsqueeze(1)) / deg  
        else:
            agg_t = agg_t_override.unsqueeze(1) 

        pred_input = torch.cat([h2_zc, h2_zy, agg_t], dim=1) 

        mask0 = (t_tensor == 0).unsqueeze(1)
        mask1 = (t_tensor == 1).unsqueeze(1)
        mu0 = self.predictor_y0(pred_input) 
        mu1 = self.predictor_y1(pred_input)  
        mu_y = torch.where(mask0, mu0, mu1).squeeze(1)  
        return mu_y


    def predict_t(self, zt_tensor, zc_tensor, adj):
        h_gcn_zc = self.gcn_zc(zc_tensor, adj)  
        h2_zc = self.mlp_zc(torch.cat([h_gcn_zc, zc_tensor], dim=1))  

        h_gcn_zt = self.gcn_zt(zt_tensor, adj)  
        h2_zt = self.mlp_zt(torch.cat([h_gcn_zt, zt_tensor], dim=1))  

        pred_input = torch.cat([h2_zc, h2_zt], dim=1)  
        t_logit = self.predictor_t(pred_input).squeeze(1)  
        return t_logit


class NDiVAE_Model(nn.Module):
    def __init__(self, feature_dim,
                 latent_dim_Zt, latent_dim_Zc, latent_dim_Zy, latent_dim_u,
                 hidden_dim, num_layers):
        super().__init__()
        config = dict(
            feature_dim=feature_dim,
            latent_dim_Zt=latent_dim_Zt,
            latent_dim_Zc=latent_dim_Zc,
            latent_dim_Zy=latent_dim_Zy,
            latent_dim_u  = latent_dim_u,  
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        self.model = Model(config)
        self.guide = Guide(config)

        disc_input_dim = config['latent_dim_Zc'] + 2
        self.discriminator = DependencyDiscriminator(disc_input_dim, config['hidden_dim'])

        self.g_zy = nn.Sequential(
            nn.Linear(config["latent_dim_Zy"] + 1, config["hidden_dim"]), 
            nn.ELU(),
            nn.Linear(config["hidden_dim"], 1)
        )

        self.g_zt = nn.Sequential(
            nn.Linear(config["latent_dim_Zt"], config["hidden_dim"]),
            nn.ELU(),
            nn.Linear(config["hidden_dim"], 1)
        )


    @staticmethod
    def generate_dependency_labels(zc, t, agg_t):
        device = zc.device
        N = zc.size(0)

        pos_zc = zc      
        pos_t = t        
        pos_agg = agg_t   
        pos_label = torch.ones((N, 1), device=device)

        idx = torch.randperm(N, device=device)
        neg_zc = zc
        neg_t = t[idx]
        neg_agg = agg_t[idx]
        neg_label = torch.zeros((N, 1), device=device)

        zc_all = torch.cat([pos_zc, neg_zc], dim=0)  
        t_all = torch.cat([pos_t, neg_t], dim=0)       
        agg_all = torch.cat([pos_agg, neg_agg], dim=0) 
        labels = torch.cat([pos_label, neg_label], dim=0)

        return zc_all, t_all, agg_all, labels

    
    @staticmethod
    def compute_sample_weights(discriminator, zc, t, agg_t, eps=1e-4):
        zc_ = zc if zc.dim() == 2 else zc.unsqueeze(1)            
        t_ = t if t.dim() == 2 else t.unsqueeze(1)                     
        agg_ = agg_t if agg_t.dim() == 2 else agg_t.unsqueeze(1)  
        with torch.no_grad():
            prob_real = discriminator(zc_, t_, agg_).clamp(min=eps, max=1-eps)
            prob_fake = 1.0 - prob_real
            weights = (prob_fake / prob_real).view(-1)
        with torch.no_grad():
            D = discriminator(zc_, t_, agg_)
        return weights

    @staticmethod
    def weighted_mse_loss(y_pred, y_true, weights):
        w_stats = weights.detach().cpu()
        return torch.mean(weights * (y_pred - y_true) ** 2)

    def fit(self, x, u, t, y, adj,
            num_epochs,
            learning_rate,
            learning_rate_decay,
            weight_decay,
            lambda_wass,    
            lambda_elbo,    
            lambda_y,       
            lambda_t,      
            lambda_reg_zt,  
            lambda_reg_zy
    ):
        N = x.size(0)
        assert t.shape == (N,)
        assert y.shape == (N,)
        assert adj.shape == (N, N)

        device = x.device
        x_tensor = x.float().to(device)
        t_tensor = t.float().to(device)
        y_tensor = y.float().to(device)
        u_tensor = u.float().to(device)
        adj_tensor = adj.float().to(device)

        optimizer_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=learning_rate,                    
            weight_decay=0)

        params_main = (
            list(self.model.parameters()) +
            list(self.guide.parameters()) +
            list(self.g_zy.parameters()) +
            list(self.g_zt.parameters())
        )
        optimizer_main = torch.optim.AdamW(
            params_main,
            lr=learning_rate,
            weight_decay=weight_decay
        )

        elbo_obj = TraceCausalEffect_ELBO().differentiable_loss
        losses = []

        for epoch in range(num_epochs):
            z_dict = self.guide.get_latent_z(x_tensor, u=u_tensor)
            zc_tensor = z_dict["zc"] 
            zy_tensor = z_dict["zy"]  
            zt_tensor = z_dict["zt"]  
            deg = adj_tensor.sum(dim=1, keepdim=True).clamp(min=1.0)
            agg_t = (adj_tensor @ t_tensor.unsqueeze(1)) / deg 

            h_gcn_zc = self.model.gcn_zc(zc_tensor, adj_tensor)              
            zc_embed = self.model.mlp_zc(torch.cat([h_gcn_zc, zc_tensor],1)) 

            self.discriminator.train()
            
            for _ in range(10):
                optimizer_d.zero_grad()

                zc_all, t_all, agg_all, labels = self.generate_dependency_labels(
                        zc_embed.detach(),
                        t_tensor.unsqueeze(1).detach(),
                        agg_t.detach()
                    )
                pred_d = self.discriminator(zc_all, t_all, agg_all)
                loss_d = F.binary_cross_entropy_with_logits(pred_d, labels)
                loss_d.backward()
                optimizer_d.step()

            self.discriminator.eval()
            raw_weights = self.compute_sample_weights(
                    self.discriminator,
                    zc_embed,
                    t_tensor,
                    agg_t
                )
                                
            raw_weights = torch.clamp(raw_weights, min=1e-3, max=1e3)
            alpha = 5.0
            log_weights = torch.log1p(alpha * raw_weights)
            weights = torch.clamp(log_weights / (log_weights.mean() + 1e-6),
                                      min=1e-3, max=1e3)
                  
            optimizer_main.zero_grad()

            elbo_loss = elbo_obj(
                self.model,
                self.guide,
                x=x_tensor,      
                u=u_tensor,       
                adj=adj_tensor,   
                size=N
            )
            
            elbo_loss = elbo_loss / N

        
            mu_y = self.model.predict_y(
                zc_tensor,     
                zy_tensor,    
                t_tensor,
                adj_tensor,
                agg_t_override=None
            )  

            mse_y = self.weighted_mse_loss(mu_y, y_tensor, weights)

            t_logit = self.model.predict_t(
                zt_tensor,    
                zc_tensor,   
                adj_tensor
            ) 
            bce_loss_t = F.binary_cross_entropy_with_logits(t_logit, t_tensor)

           
            mask_t1 = (t_tensor > 0.5).nonzero(as_tuple=False).squeeze(-1)
            mask_t0 = (t_tensor < 0.5).nonzero(as_tuple=False).squeeze(-1)
            if mask_t1.numel() > 0 and mask_t0.numel() > 0:
                zy_t1 = zy_tensor[mask_t1] 
                zy_t0 = zy_tensor[mask_t0] 
                d_z_loss, _ = utils.wasserstein(
                    zy_t1, zy_t0,
                    cuda=(device.type == "cuda")
                )
            else:
                d_z_loss = torch.tensor(0.0, device=device)

            Zy = zy_tensor             
            h_gcn_zy = self.model.gcn_zy(Zy, adj_tensor)    
            h2_zy = self.model.mlp_zy(torch.cat([h_gcn_zy, Zy], dim=1))  
            
            t_input = t_tensor.unsqueeze(1) 
            gzy_input = torch.cat([h2_zy, t_input], dim=1) 
            y_pred_from_zy = self.g_zy(gzy_input).squeeze(1)   
            loss_zy_reg = F.mse_loss(y_pred_from_zy, y_tensor)

            Zt = zt_tensor           
            h_gcn_zt = self.model.gcn_zt(Zt, adj_tensor)    
            h2_zt = self.model.mlp_zt(torch.cat([h_gcn_zt, Zt], dim=1))  
            t_pred_from_zt = self.g_zt(h2_zt).squeeze(1)     
            loss_zt_reg = F.binary_cross_entropy_with_logits(t_pred_from_zt, t_tensor)

            total_loss = (
                lambda_elbo * elbo_loss
              + lambda_y    * mse_y
              + lambda_t    * bce_loss_t
              + lambda_wass * d_z_loss
              + lambda_reg_zy * loss_zy_reg
              + lambda_reg_zt * loss_zt_reg
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params_main, max_norm=10.0)
            optimizer_main.step()

            record = {
                "epoch": epoch + 1,
                "ELBO":           lambda_elbo * elbo_loss.item(),
                "Weighted_MSE_y": lambda_y * mse_y.item(),
                "BCE_t":          lambda_t * bce_loss_t.item(),
                "Wass_zy":        lambda_wass * d_z_loss.item(),
                "Reg_zy":         lambda_reg_zy * loss_zy_reg.item(),
                "Reg_zt":         lambda_reg_zt * loss_zt_reg.item(),
                "Total":          total_loss.item()
            }

            losses.append(record)

            
            if (epoch + 1) % 5 == 0:
                info = record
                print(
                    f"Epoch {info['epoch']}/{num_epochs}: "
                    f"ELBO={info['ELBO']:.6f}, "
                    f"Weighted_MSE_y={info['Weighted_MSE_y']:.6f}, "
                    f"BCE_t={info['BCE_t']:.6f}, "
                    f"Wass_zy={info['Wass_zy']:.6f}, "
                    f"Reg_zy={info['Reg_zy']:.6f}, "
                    f"Reg_zt={info['Reg_zt']:.6f}, "
                    f"Total={info['Total']:.6f}"
                )


        return losses
