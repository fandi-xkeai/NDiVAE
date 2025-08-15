This repository implements the method presented in the paper **"Identifiable Disentangled Representation Learning for Causal Inference under Networked Interference"**.

## Contents

- **Theoretical Analysis**  
  Includes all theorem proofs from **Section 3.5** of the paper, providing a formal analysis of the identifiability of the proposed model.

- **Experimental Results**  
  Contains causal effect estimation results from **Section 4**:
  - BC(hete) dataset
  - Flickr(hete) dataset

- **Code**  
  The repository provides the full implementation in the `src` folder. The main script can be run as follows:
  ```bash
  python NDiVAE_run.py

# Theoretical Analysis

## Proof of Theorem 3.2
Follows directly from Theorem 1 in [19].

## Proof of Theorem 3.3
Recall that $\mathbf{x}_i$ serves as a proxy variables for the latent factors. Under Assumptions 1–4, we have:

$$
\begin{aligned}
&\mathbb{E}\bigl[y_i(t_i^{\prime},g_i^{\prime})| \mathbf{x}_i,\{\mathbf{x}_j\}_{j\in\mathcal{N}_i}\bigr]
\\=&\int
\mathbb{E}\bigl[y_i(t_i^{\prime},g_i^{\prime})| \mathbf{z}_{i},\{\mathbf{z}_{j}\}_{j\in\mathcal{N}_i}\bigr]
p\bigl(\mathbf{z}_{i},\{\mathbf{z}_{j}\}_{j\in\mathcal{N}_i}| \mathbf{x}_i,\{\mathbf{x}_j\}_{j\in\mathcal{N}_i}\bigr)
d\mathbf{z}_{i}d\{\mathbf{z}_{j}\}_{j\in\mathcal{N}_i}
\\=&\int
\mathbb{E}\bigl[y_i(t_i^{\prime},g_i^{\prime})| t_i^{\prime},g_i^{\prime},\mathbf{z}_{i},\{\mathbf{z}_{j}\}_{j\in\mathcal{N}_i}\bigr]
p\bigl(\mathbf{z}_{i},\{\mathbf{z}_{j}\}_{j\in\mathcal{N}_i}| \mathbf{x}_i,\{\mathbf{x}_j\}_{j\in\mathcal{N}_i}\bigr)
d\mathbf{z}_{i}d\{\mathbf{z}_{j}\}_{j\in\mathcal{N}_i}
\\=&\int
\mathbb{E}\bigl[y_i| t_i^{\prime},g_i^{\prime},\mathbf{z}_{i},\{\mathbf{z}_{j}\}_{j\in\mathcal{N}_i}\bigr]
p\bigl(\mathbf{z}_{i},\{\mathbf{z}_{j}\}_{j\in\mathcal{N}_i}| \mathbf{x}_i,\{\mathbf{x}_j\}_{j\in\mathcal{N}_i}bigr)
d\mathbf{z}_{i}d\{\mathbf{z}_{j}\}_{j\in\mathcal{N}_i}
\\=&\int
\mathbb{E}\bigl[y_i| t_i^{\prime},g_i^{\prime},\mathbf{z}_{c,i},\{\mathbf{z}_{c,j}\}_{j\in\mathcal{N}_i},\mathbf{z}_{y,i},\{\mathbf{z}_{y,j}\}_{j\in\mathcal{N}_i}\bigr]
p\bigl(\mathbf{z}_{c,i},\{\mathbf{z}_{c,j}\}_{j\in\mathcal{N}_i}, \\&\mathbf{z}_{y,i},\{\mathbf{z}_{y,j}\}_{j\in\mathcal{N}_i}| \mathbf{x}_i,\{\mathbf{x}_j\}\bigr)
d\mathbf{z}_{c,i}d\{\mathbf{z}_{c,j}\}_{j\in\mathcal{N}_i}d\mathbf{z}_{y,i}d\{\mathbf{z}_{y,j}\}_{j\in\mathcal{N}_i}.
\end{aligned}
$$

By Theorem 3.2, the joint posterior distribution

$$
p\bigl(\mathbf{z}_{c,i},\{\mathbf{z}_{c,j}\}_{j\in\mathcal{N}_i}, \mathbf{z}_{y,i},\{\mathbf{z}_{y,j}\}_{j\in\mathcal{N}_i}| \mathbf{x}_i,\{\mathbf{x}_j\}_{j\in\mathcal{N}_i}\bigr)
$$

is identifiable. Thus, the entire integral is identifiable. An analogous derivation applies to 

$$
\mathbb{E}[y_i(t_i^{\prime\prime},g_i^{\prime\prime})| \mathbf{x}_i,\{\mathbf{x}_j\}_{j\in\mathcal{N}_i}],
$$

and hence, the individual treatment effect $\tau_i$ is identifiable.

## Proof of Theorem 3.7
We first expand the definition of the average counterfactual loss:

$$
\begin{aligned}
\epsilon_{CF} 
&= \mathbb{E}_{p(t,g)}  \mathbb{E}_{p(t',g')} 
\int_{\tilde{\mathcal{Z}}_{cy}} 
\ell_{h,\phi_{cy}}(\tilde{\mathbf{z}}_{cy}, t, g) 
p(\tilde{\mathbf{z}}_{cy} | t',g')  d\tilde{\mathbf{z}}_{cy} \\
&= \int_{\mathcal{T} \times \mathcal{G}} \int_{\tilde{\mathcal{Z}}_{cy}}
\ell_{h,\phi_{cy}}(\tilde{\mathbf{z}}_{cy}, t, g) 
p(\tilde{\mathbf{z}}_{cy}) 
p(t,g)  d\tilde{\mathbf{z}}_{cy}  dt  dg
\end{aligned}
$$

Similarly, the weighted average factual loss:

$$
\epsilon_F^{(w)}
= \int_{\mathcal{T} \times \mathcal{G}} \int_{\tilde{\mathcal{Z}}_{cy}}
\ell_{h,\phi_{cy}}(\tilde{\mathbf{z}}_{cy}, t, g) 
w(\tilde{\mathbf{z}}_{cy}, t, g) 
p(\tilde{\mathbf{z}}_{cy}, t, g)  d\tilde{\mathbf{z}}_{cy}  dt  dg
$$

Their difference:
$$
\begin{aligned}
\epsilon_{CF} - \epsilon_F^{(w)}
&= \int_{\mathcal{T} \times \mathcal{G}} \int_{\tilde{\mathcal{Z}}_{cy}}
\ell_{h,\phi_{cy}}(\tilde{\mathbf{z}}_{cy}, t, g)
[ p(\tilde{\mathbf{z}}_{cy})p(t,g) 
- w(\tilde{\mathbf{z}}_{cy}, t, g) p(\tilde{\mathbf{z}}_{cy}, t, g) ]
 d\tilde{\mathbf{z}}_{cy}  dt  dg\nonumber\\
% By the assumption \(\frac{1}{B_{\phi_{cy}}} \ell_{h,\phi_{cy}}(\psi_{cy}(\mathbf{r}_{cy}), t, g) \in \mathcal{G}\), we have:
% \begin{align}
% \epsilon_{CF} - \epsilon_F^{(w)}
&\le B_{\phi_{cy}}  \sup_{g \in \mathcal{G}}
\int_{\mathcal{T} \times \mathcal{G}} \int_{\tilde{\mathcal{Z}}_{cy}}
g(\tilde{\mathbf{z}}_{cy}, t, g) 
[ p(\tilde{\mathbf{z}}_{cy})p(t,g) 
-w(\tilde{\mathbf{z}}_{cy}, t, g) p(\tilde{\mathbf{z}}_{cy}, t, g) ]
 d\tilde{\mathbf{z}}_{cy}  dt  dg.\nonumber
 \end{aligned}
$$

Now, perform the change of variables $ \mathbf{r}_{cy} = \phi_{cy}(\tilde{\mathbf{z}}_{cy}) $ with inverse $ \psi_{cy} $, and let $ \widetilde{\mathcal{G}} = \left\{ \tilde{g} : \tilde{g}(\tilde{\mathbf{z}}) = g(\phi_{cy}(\tilde{\mathbf{z}})) \big| \det J_{\psi_{cy}}(\phi_{cy}(\tilde{\mathbf{z}})) \big|, \ g \in \mathcal{G} \right\} $. Then, the integrals over $ \tilde{\mathbf{z}}_{cy} $ can be rewritten over $ \mathbf{r}_{cy} $, yielding $ \epsilon_{CF} - \epsilon_F^{(w)} \le B_{\phi_{cy}} \ \mathrm{IPM}_{\widetilde{\mathcal{G}}} \Big( p_{\phi_{cy}}(\mathbf{r}_{cy}) \, p(t,g), \ w(\psi_{cy}(\mathbf{r}_{cy}), t, g) \, p(\mathbf{r}_{cy}, t, g) \Big) $.

## Proof of Theorem 3.8
$$
\begin{aligned}
\epsilon_{\mathrm{PEHE}}(f,\phi_{cy})
\overset{(\mathrm{i})}{=}&
\mathbb{E}_{p(t,g)p(t',g')p(\tilde{\mathbf z}_{cy})}
\Big[
\big(\hat{\tau}_{(t,g),(t',g')}(\tilde{\mathbf z}_{cy}) - \tau_{(t,g),(t',g')}(\tilde{\mathbf z}_{cy})\big)^2
\Big] \nonumber\\
\overset{(\mathrm{ii})}{\le}&
4  \mathbb{E}_{p(t,g)p(\tilde{\mathbf z}_{cy})}
\Big[
\big(f(\phi_{cy}(\tilde{\mathbf z}_{cy}),t,g)-m(t,g;\tilde{\mathbf z}_{cy})\big)^2
\Big] \nonumber\\
\overset{(\mathrm{iii})}{=}&
4\big(\epsilon_{CF} - \sigma_{Y}\big) \nonumber\\
\overset{(\mathrm{iv})}{\le}&
4\epsilon_F^{(w)}
+ 4B_{\phi_{cy}}\mathrm{IPM}_{\widetilde{\mathcal{G}}}\!\big(
p_{\phi_{cy}}(\mathbf r_{cy})p(t,g),
w(\psi_{cy}(\mathbf r_{cy}),t,g)p(\mathbf r_{cy},t,g)
\big) - 4\sigma_{Y}. \label{eq:pehe_chain}
\end{aligned}
$$
Here, (i) is the definition of PEHE as the expectation of the squared estimation error between $\hat{\tau}$ and $\tau$; (ii) follows by substituting $\hat{\tau}-\tau = [f(\phi_{cy}(\tilde{\mathbf z}_{cy}),t,g)-m(t,g;\tilde{\mathbf z}_{cy})] - [f(\phi_{cy}(\tilde{\mathbf z}_{cy}),t',g')-m(t',g';\tilde{\mathbf z}_{cy})]$, applying the pointwise inequality $(a-b)^2 \le 2a^2 + 2b^2$ and using the symmetry of $p(t,g)p(t',g')$ to obtain the factor 4; (iii) is the bias–variance decomposition under squared loss: $\ell_{h,\phi_{cy}}(\tilde{\mathbf z}_{cy},t,g) = (f-m)^2 + \mathrm{Var}(y(t,g)|\tilde{\mathbf z}_{cy})$, integration w.r.t. $p(t,g)p(\tilde{\mathbf z}_{cy})$ yields $\mathbb{E}[(f-m)^2] = \epsilon_{CF}-\sigma_Y$; (iv) applies the bound on $\epsilon_{CF}$ proved in Theorem 3.7 and multiplies by 4.


# Experimental Results

## Experimental results of causal effect estimation on the BC(hete) dataset.
The best result is highlighted in **bold**.

| metric             | setting        | effect | CFR+g             | TNRNet+g          | NetDeconf+g       | TEDVAE+g            | TNDGVA+g        | NetEst           | NDiVAE_W        | NDiVAE_R        | NDiVAE              |
| ------------------ | -------------- | ------ | ----------------- | ----------------- | ----------------- | ------------------- | --------------- | ---------------- | --------------- | --------------- | ------------------- |
| \epsilon_{average} | within sample  | AME    | 0.1948 ± 0.0763   | 0.2830 ± 0.3795   | 0.5995 ± 0.5194   | **0.0780 ± 0.0775** | 0.8412 ± 0.5181 | 1.4273 ± 2.5528  | 0.1191 ± 0.0967 | 0.3581 ± 0.5117 | 0.1505 ± 0.0629     |
|                    |                | ASE    | 0.2095 ± 0.1515   | 0.2298 ± 0.2402   | 0.1856 ± 0.1945   | 0.2276 ± 0.1530     | 0.2577 ± 0.4707 | 0.1445 ± 0.2790  | 0.1414 ± 0.1876 | 0.0891 ± 0.0983 | **0.0735 ± 0.0929** |
|                    |                | ATE    | 1.0272 ± 0.7736   | 0.2910 ± 0.2341   | 1.7178 ± 1.0755   | 0.2689 ± 0.2163     | 1.6612 ± 2.0534 | 1.6478 ± 2.8578  | 0.2762 ± 0.3181 | 0.2827 ± 0.2514 | **0.2253 ± 0.0962** |
|                    | without sample | AME    | 0.6021 ± 0.8549   | 0.5980 ± 0.9114   | 2.2709 ± 2.2264   | **0.1075 ± 0.0978** | 1.6205 ± 1.1874 | 1.8334 ± 2.2307  | 0.1229 ± 0.0895 | 0.3306 ± 0.4977 | 0.1291 ± 0.0650     |
|                    |                | ASE    | 0.1805 ± 0.1330   | 0.2215 ± 0.2289   | 0.1834 ± 0.1829   | 0.2219 ± 0.1501     | 0.2639 ± 0.4620 | 0.2413 ± 0.2900  | 0.1381 ± 0.1877 | 0.0893 ± 0.0959 | **0.0736 ± 0.0907** |
|                    |                | ATE    | 1.3072 ± 0.8870   | 0.9268 ± 0.8936   | 2.4284 ± 1.8841   | 0.2613 ± 0.1693     | 2.5085 ± 1.8839 | 2.0422 ± 2.5235  | 0.2790 ± 0.2988 | 0.2552 ± 0.2382 | **0.2062 ± 0.1071** |
| \epsilon_{RPEHE}   | within sample  | IME    | 4.2282 ± 5.3061   | 8.3690 ± 9.8779   | 1.1664 ± 0.4168   | 0.4114 ± 0.2176     | 1.9453 ± 2.3938 | 1.7130 ± 2.4636  | 0.4039 ± 0.1466 | 0.6451 ± 0.3766 | **0.3756 ± 0.1667** |
|                    |                | ISE    | 0.6607 ± 0.2735   | 0.3901 ± 0.2573   | 0.9147 ± 0.2369   | 0.2682 ± 0.1393     | 0.4787 ± 0.8060 | 0.3789 ± 0.3019  | 0.2377 ± 0.2138 | 0.1664 ± 0.1354 | **0.1228 ± 0.0923** |
|                    |                | ITE    | 4.3843 ± 5.3103   | 8.3724 ± 9.8678   | 0.8271 ± 0.3332   | 0.2404 ± 0.0864     | 0.6711 ± 0.3784 | 0.9263 ± 0.6795  | 0.2142 ± 0.0783 | 0.1616 ± 0.0864 | **0.1445 ± 0.0984** |
|                    | without sample | IME    | 13.9890 ± 12.3407 | 19.2261 ± 32.0467 | 14.8738 ± 21.7055 | 0.4278 ± 0.2287     | 4.0199 ± 3.8876 | 7.8289 ± 12.0267 | 0.4230 ± 0.1351 | 0.6371 ± 0.4094 | **0.3878 ± 0.1560** |
|                    |                | ISE    | 0.6540 ± 0.2436   | 0.3848 ± 0.2493   | 0.9291 ± 0.2105   | 0.2643 ± 0.1343     | 0.6043 ± 0.7679 | 6.5214 ± 12.4877 | 0.2371 ± 0.2113 | 0.1663 ± 0.1326 | **0.1248 ± 0.0890** |
|                    |                | ITE    | 14.2980 ± 11.9550 | 19.3698 ± 31.9472 | 15.1435 ± 21.4677 | 0.5124 ± 0.1810     | 4.7971 ± 3.6604 | 6.9958 ± 9.7088  | 0.5173 ± 0.2044 | 0.5545 ± 0.2015 | **0.4286 ± 0.1302** |


## Experimental results of causal effect estimation on the Flickr(hete) dataset.
The best result is highlighted in **bold**.
| metric    | setting        | effect | CFR+g            | TNRNet+g          | NetDeconf+g      | TEDVAE+g            | TNDGVA+g        | NetEst          | NDiVAE_W        | NDiVAE_R        | NDiVAE              |
| --------- | -------------- | ------ | ---------------- | ----------------- | ---------------- | ------------------- | --------------- | --------------- | --------------- | --------------- | ------------------- |
| ε_average | within sample  | AME    | 0.2696 ± 0.2157  | 0.8409 ± 0.5796   | 0.1216 ± 0.0580  | 0.0960 ± 0.0411     | 0.1198 ± 0.1153 | 0.4744 ± 0.2007 | 0.1443 ± 0.1322 | 0.1756 ± 0.0972 | **0.0768 ± 0.0531** |
|           |                | ASE    | 0.2919 ± 0.1175  | 0.5373 ± 0.0884   | 0.7570 ± 0.1802  | 0.5225 ± 0.1251     | 0.1242 ± 0.0990 | 0.6529 ± 0.2635 | 0.1986 ± 0.1962 | 0.2918 ± 0.1639 | **0.0912 ± 0.0756** |
|           |                | ATE    | 1.7368 ± 0.8151  | 1.0377 ± 0.8717   | 4.1648 ± 0.6236  | 0.5500 ± 0.2229     | 0.7705 ± 0.9841 | 1.7540 ± 0.7662 | 0.3853 ± 0.2076 | 0.2187 ± 0.1088 | **0.1347 ± 0.0778** |
|           | without sample | AME    | 0.4248 ± 0.3504  | 1.3250 ± 1.5017   | 2.2430 ± 2.1002  | 0.1361 ± 0.0995     | 0.3752 ± 0.4160 | 0.9135 ± 0.4818 | 0.1982 ± 0.1622 | 0.1655 ± 0.1032 | **0.0995 ± 0.0681** |
|           |                | ASE    | 0.2997 ± 0.1252  | 0.5535 ± 0.0622   | 0.7781 ± 0.1689  | 0.5288 ± 0.1190     | 0.1127 ± 0.1088 | 0.6762 ± 0.2146 | 0.1924 ± 0.1900 | 0.2826 ± 0.1627 | **0.0836 ± 0.0713** |
|           |                | ATE    | 1.7324 ± 1.0218  | 1.5460 ± 0.9225   | 2.4723 ± 1.7292  | 0.4797 ± 0.2579     | 0.9047 ± 0.9503 | 2.1547 ± 0.5408 | 0.4284 ± 0.2343 | 0.2450 ± 0.1391 | **0.1447 ± 0.0944** |
| ε_RPEHE   | within sample  | IME    | 3.4173 ± 2.7402  | 11.3895 ± 8.4302  | 1.1291 ± 0.1009  | **0.4937 ± 0.1088** | 1.1379 ± 0.4752 | 1.4507 ± 0.3710 | 0.6292 ± 0.0864 | 0.6358 ± 0.1020 | 0.6006 ± 0.0959     |
|           |                | ISE    | 0.6920 ± 0.3666  | 0.8773 ± 0.1611   | 1.7612 ± 0.1195  | 0.5426 ± 0.1198     | 0.3154 ± 0.3830 | 0.7835 ± 0.2815 | 0.3017 ± 0.1223 | 0.3427 ± 0.1026 | **0.1348 ± 0.0768** |
|           |                | ITE    | 4.0828 ± 2.3227  | 11.5279 ± 8.2275  | 4.2549 ± 0.6263  | 0.7521 ± 0.2675     | 1.4236 ± 0.9662 | 2.1774 ± 0.7702 | 0.7289 ± 0.1344 | 0.6541 ± 0.1028 | **0.6056 ± 0.1101** |
|           | without sample | IME    | 9.8262 ± 6.5535  | 19.1161 ± 18.9897 | 11.8373 ± 9.5396 | 0.6748 ± 0.1158     | 3.2427 ± 2.9759 | 2.3803 ± 1.1103 | 0.6613 ± 0.0905 | 0.6724 ± 0.1250 | **0.6152 ± 0.0926** |
|           |                | ISE    | 0.6921 ± 0.3917  | 0.8917 ± 0.2027   | 1.6908 ± 0.1402  | 0.5548 ± 0.1135     | 0.3908 ± 0.3648 | 1.7230 ± 0.9834 | 0.2969 ± 0.1201 | 0.3384 ± 0.1031 | **0.1324 ± 0.0729** |
|           |                | ITE    | 10.2425 ± 6.0521 | 19.1675 ± 18.9077 | 12.6194 ± 8.3968 | 0.8475 ± 0.1496     | 3.4753 ± 2.8550 | 3.0258 ± 0.8466 | 0.7635 ± 0.1433 | 0.7043 ± 0.1174 | **0.6183 ± 0.1152** |