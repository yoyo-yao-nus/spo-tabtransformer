# -*- coding: utf-8 -*-
"""
Created on Tue May 27 20:53:03 2025

@author: Yoyooooo
"""

from mpax import create_lp, r2HPDHG, raPDHG
from utils import *
from opt_utils import *
class SPOTTF(nn.Module):
    def __init__(self, categories, num_continuous, dim_out, lr=1e-4):
        super(SPOTTF, self).__init__()
        self.model = TabTransformer(
                        categories = categories,
                        num_continuous = num_continuous,
                        dim = 16,
                        dim_out = dim_out,
                        depth = 6,
                        heads = 8,
                        attn_dropout = 0.1,
                        ff_dropout = 0.1,
                        mlp_hidden_mults = (4, 2),
                        mlp_act = nn.ReLU(),
                    )
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.lmbdas = np.load(os.path.join(TabularDataset.DIRECTORY,'Data/lmbdas.npy'))
        self.gammas = np.load(os.path.join(TabularDataset.DIRECTORY,'Data/gammas.npy'))
        self.setup_flag = False

    def mpax_setup(self, constrs_dict):
        a = [0.3,0.05,0.05,0.3,0.05]
        b = [0.35,0.1,0.1,0.35,0.1]
        seg = constrs_dict['seg']
        di = [0.3,0.3,0.2,0.1,0.1,0]
        N = len(seg)
        G, h = build_G_and_h_sparse(N, seg, a, b, di, self.lmbdas, self.gammas)
        self.A = jnp.empty((0,N*6))
        self.b = jnp.empty((0,))
        self.G = G
        self.h = h
        self.l = jnp.zeros(N*6)
        self.u = jnp.ones(N*6)
        self.setup_flag = True
        print('An LP model is set up.')

    def optimize(self, c):
        c = c.reshape([-1])

        lp = create_lp(-c, self.A, self.b, self.G, self.h, self.l, self.u)

        # solver = r2HPDHG(eps_abs=1e-3, eps_rel=1e-3, verbose=False)
        solver = raPDHG(verbose=False)

        result = solver.optimize(lp)


        # obj = result.primal_objective
        solution = result.primal_solution
        obj = np.dot(-c, solution)

        return solution, obj


    def forward(self, x_cat, x_num, constrs_dict, training=True):
        preds = self.model(x_cat, x_num)
        if training:
          return preds
        with torch.no_grad():
          solution, objective = self.optimize(preds.detach().cpu().numpy())
        return preds, torch.tensor(np.array(solution),dtype=torch.float32)

    def update(self, x_cat, x_num, y, constrs_dict):
        if not self.setup_flag:
          self.mpax_setup(constrs_dict)
        preds = self.forward(x_cat, x_num, constrs_dict, training=True)
        with torch.no_grad():
          solution, objective = self.optimize((2 * preds - y).detach().cpu().numpy())
          # print('1st Optimization done.')

          true_sol, true_obj = self.optimize(y.detach().cpu().numpy())
          # print('2nd optimization done.')

        # preds_jax, y_jax = jnp.array(preds.detach().cpu().numpy()), jnp.array(y.detach().cpu().numpy())
        # spo_loss, _ = pso_fun(preds_jax, y_jax, true_sol, true_obj)
        # print(f'SPO Loss: {spo_loss}')
        # grad = jax.grad(pso_fun)(preds_jax, y_jax, true_sol, true_obj)
        spo_loss = -objective + 2 * torch.sum(-torch.tensor(np.array(true_sol),device=self.device) * preds.reshape([-1])) - true_obj
        # print(f'SPO Loss: {spo_loss}')
        grad = -2 * (true_sol - solution)
        grad = torch.tensor(np.array(grad), dtype=torch.float32, device=self.device)
        loss = torch.sum(grad * preds.reshape([-1]))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return spo_loss

