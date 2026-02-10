06 — Learned Diffusion (Full‑Rank L) as a Toggle on Top of UOT

Goal

Add a toggleable diffusion learning head to the morphology‑space ODE so the model can learn state‑dependent process noise (SDE view) without changing the UOT pipeline. When the toggle is on, we add a diffusion loss; when off, training reduces to the original UOT‑supervised drift fit.

⸻

Big picture
	•	UOT (minibatch, unbalanced): gives teacher velocities \hat v(z,t) from local couplings between adjacent time bins (and optionally an estimate of dispersion).
	•	Drift head f_\theta(z,t): learns the mean field that ODEs already use.
	•	New diffusion head L_\theta(z,t) (full‑rank): learns a factor such that D_\theta = L_\theta L_\theta^\top is the per‑unit‑time covariance of process noise.
	•	Toggle: --learn_diffusion adds a diffusion loss; simulation uses Euler–Maruyama when enabled, otherwise standard ODE stepping.

⸻

Why this helps
	•	Captures intrinsic heterogeneity (aleatoric) that plain ODEs miss.
	•	Produces trajectory distributions (not only mean paths).
	•	Gives directional variance: where/along which axes embryos diverge.
	•	Keeps pipeline modular: UOT unchanged; only adds a head + loss + (optional) SDE integrator.

⸻

Pre‑requisites (already in repo from 01–03)
	1.	Minibatch UOT that yields teacher velocities \hat v_i and row masses r_i for pairs (X_t, Y_{t+\Delta t}).
	2.	Time‑weighted spectral clustering utility to form meaningful local minibatches.
	3.	(Optional) Self‑edge lower bound to bias known next‑frame matches.

⸻

Data used by this module
	•	Pairs (z_t, z_{t+h}) with their timestamps (h = \Delta t per pair). These are already present while sampling adjacent time bins for UOT.
	•	Teacher velocity \hat v_t from UOT at the source point. (Or use UOT‑weighted regression to form \hat v_t.)

⸻

Model additions
	•	Shared encoder: h = \mathrm{enc}(z_t, t).
	•	Drift head: f_\theta = W_f h + b_f  \in \mathbb{R}^d   (unchanged from ODE version).
	•	Diffusion head (full‑rank): predict the lower triangle of L_\theta \in \mathbb{R}^{d\times d}. Use softplus on diagonal (plus \varepsilon) to ensure positivity. Define D_\theta = L_\theta L_\theta^\top.
NOTE: Include a code comment about a low‑rank variant later (L_\theta \in \mathbb{R}^{d\times q}, q\ll d) for efficiency; same loss and sampler.

⸻

Losses (toggleable)

Let r = (z_{t+h} - z_t) - f_\theta(z_t,t)\cdot h be the drift‑subtracted residual displacement.
	1.	Drift loss (as before; choose one):
	•	MSE (mass‑weighted):  \mathcal L_{\text{drift}} = \sum_i w_i\, \| f_\theta(z_i,t_i) - \hat v_i \|^2
	•	or Gaussian NLL using an auxiliary diagonal covariance on regression residuals.
	2.	Diffusion loss (only if --learn_diffusion):
	•	Local diffusion target (method of moments):  \widehat D(z_t,t) \approx \mathbb E[ r r^\top / h ].
	•	Option A (pairwise): average over neighbors in the minibatch.
	•	Option B (UOT‑weighted): r_j = (y_j - x_i) - \hat v_i\cdot h, weight by \Pi_{ij}, then \widehat D_i = (\sum_j \Pi_{ij} r_j r_j^\top)/(h \sum_j \Pi_{ij}).
	•	Objective: \mathcal L_{\text{diff}} = \| L_\theta L_\theta^\top - \widehat D \|_F^2.
	•	Total: \mathcal L = \mathcal L_{\text{drift}} + \lambda_{\text{diff}}\, \mathcal L_{\text{diff}}.

⸻

Implementation steps
	1.	Wire the head (PyTorch sketch):
	•	Build TimeEmbed, Encoder (z,t \to features).
	•	DriftHead(H→d) as before.
	•	CholeskyHead(H→lower_triangle(d)): map to lower‑tri entries; diag := softplus(diag_raw)+ε.
	•	Reconstruct L_θ from lower triangle; compute D_pred = L_θ @ L_θ.T.
	2.	Compute residuals and \widehat D:
	•	For each source z_t in a minibatch, collect its matched targets (either tracked next frame or UOT neighbors).
	•	r = (z_{t+h} - z_t) - f_\theta\cdot h. If using UOT neighbors, use r_j per match and average with \Pi weights.
	•	Set \widehat D = \text{mean}(r r^\top / h) over the local set; add small \varepsilon I for stability.
	3.	Assemble losses:
	•	\mathcal L_{\text{drift}} from \hat v (mass‑weighted by UOT row mass r_i or thresholded by r_i \ge \eta).
	•	If learn_diffusion:
	•	\mathcal L_{\text{diff}} = \mathrm{mean}\,\| D_{\text{pred}} - \widehat D \|_F^2.
	•	\mathcal L_{\text{total}} = \mathcal L_{\text{drift}} + \lambda_{\text{diff}} \cdot \mathcal L_{\text{diff}}.
	4.	Training toggle:
	•	CLI/Config: --learn_diffusion true|false, --lambda_diff 0.1, --diffusion_low_rank false, --q 8 (ignored for full‑rank), --diffusion_epsilon 1e-6.
	•	When off: do not instantiate the diffusion head; skip \widehat D and \mathcal L_{\text{diff}}; keep ODE training unchanged.
	5.	Simulation toggle:
	•	If learn_diffusion=false: integrate ODE with standard solver (RK4/Dopri).
	•	If true: integrate Euler–Maruyama:
z_{t+h} = z_t + f_\theta(z_t,t)\,h + L_\theta(z_t,t)\,\sqrt{h}\,Z,\quad Z\sim\mathcal N(0,I_d).
(EM is fine; higher‑order SDE solvers are optional.)

⸻

Key equations (for reference)
	•	Diffusion per unit time:  \(D_\theta(z,t) = L_\theta(z,t) L_\theta(z,t)^\top  \succeq 0\).
	•	Residual:  r = (z_{t+h} - z_t) - f_\theta(z_t,t)\,h.
	•	Method of moments:  \widehat D(z_t,t) \approx \mathbb E[ r r^\top / h ]  (pairwise or UOT‑weighted).
	•	Loss:  \mathcal L = \mathcal L_{\text{drift}} + \lambda_{\text{diff}} \cdot \| L_\theta L_\theta^\top - \widehat D \|_F^2.
	•	Simulation (SDE):  z_{t+h} = z_t + f_\theta\,h + L_\theta\,\sqrt{h}\,Z,  Z\sim\mathcal N(0,I).

⸻

How it fits with 01–03
	•	01_minibatch_uot: unchanged. Still produces \hat v and \Pi; optionally provides r_j for UOT‑weighted \widehat D.
	•	02_time_weighted_spectral_clustering: unchanged. Still defines local neighborhoods for stable \widehat D estimation.
	•	03_self_edge_lower_bound: unchanged. Biases couplings to known tracks; improves \hat v and r_j quality; reduces spurious variance in \widehat D.

⸻

Benefits and knobs

Benefits
	•	Trajectory distributions; variance accumulates with elapsed time.
	•	Directional heterogeneity via eigenvectors of D_\theta.
	•	Modular: no change to UOT core; just extra head + loss + optional EM integrator.

Knobs
	•	\lambda_{\text{diff}}: strength of diffusion fit (suggest schedule: warm‑up 0→target in first N epochs).
	•	Rank: full‑rank (default); low‑rank (future flag --diffusion_low_rank true, q<d) to cut compute and capture subspace noise.
	•	\widehat D smoothing: shrinkage \widehat D \leftarrow (1-\alpha)\widehat D + \alpha\,(\operatorname{tr}(\widehat D)/d)I; neighborhood size for covariance estimation.
	•	Stability: add \varepsilon I to \widehat D; clip \operatorname{tr}(D_\theta) or \|D_\theta\|_F; gradient clipping.
	•	Simulator step h: ensure invariance by using the true frame gaps at training; at inference pick h small enough for stability.

⸻

Testing & diagnostics
	•	Local calibration: check \|D_{\text{pred}} - \widehat D\|_F distributions over validation minibatches.
	•	Global calibration: roll out multiple stochastic trajectories from the same initial z_0; compare empirical spread vs. held‑out displacement spread at future times (energy distance or sliced Wasserstein).
	•	Ablations: (i) drift‑only vs drift+diffusion, (ii) full‑rank vs diagonal, (iii) UOT‑weighted \widehat D vs pairwise \widehat D.
	•	Plots: visualize eigenvalues/vectors of D_\theta along the wild‑type backbone; heatmap heterogeneity over time.

⸻

Pseudocode (training step sketch)

for minibatch in loader:
    # sample local pairs via spectral clusters; get UOT plan Π (01), teacher v̂, gaps h
    z_t, z_tp, h, v_hat, (optional Π matches) = batch

    # forward
    h = enc(z_t, t)
    f = drift_head(h)
    if learn_diffusion:
        L = chol_head(h)                 # lower‑triangular, softplus diag
        D_pred = L @ L.T                 # per‑unit‑time diffusion

    # drift loss
    L_drift = mse_or_nll(f, v_hat, weights=row_mass)  # from UOT

    # diffusion loss
    if learn_diffusion:
        r = (z_tp - z_t) - f * h[:, None]            # residual displacements
        if use_uot_weighting:
            # r_j per matched target, weight by Π_{ij}; average rr^T/h per source i
            D_hat = uot_weighted_second_moment(...)
        else:
            D_hat = local_covariance(r / h[:, None]) * h_mean
        L_diff = frobenius(D_pred - D_hat).mean()
        loss = L_drift + lambda_diff * L_diff
    else:
        loss = L_drift

    loss.backward(); optimizer.step()


⸻

Notes on low‑rank (comment to include in code)

To reduce compute or encode subspace stochasticity, set q≪d and predict L_\theta \in \mathbb{R}^{d\times q} (rectangular). Keep the same loss with D_{\text{pred}} = L_\theta L_\theta^\top. Sampling uses Z \in \mathbb{R}^q. This often suffices if empirical \widehat D is near low‑rank.