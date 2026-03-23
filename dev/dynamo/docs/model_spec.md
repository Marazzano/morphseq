# Morphogenetic Dynamics Model: Technical Specification

## 1. Overview

This document specifies a stochastic dynamical model for forward prediction of embryo trajectories in a learned latent space. The model learns a shared developmental potential landscape modulated by a low-dimensional set of perturbation modes. Given a partial observed trajectory, the model infers per-embryo mode loadings and rate parameters via closed-form solves, then forward-simulates under the inferred dynamics.

A kernel regression baseline is specified alongside the primary model to benchmark whether the learned dynamics justify their complexity.

---

## 2. Problem Setting

- Observed data: collections of embryo trajectories in a latent space $z \in \mathbb{R}^d$ (e.g., $d = 10$, from a pretrained VAE or similar embedding model).
- Each embryo $e$ belongs to a perturbation class $p(e)$ (e.g., gene knockdown, drug treatment, wild-type).
- Observations may be irregularly spaced in time and vary in length. Some embryos may have as few as a single observed snapshot.
- Goal: given a partial trajectory $z_{t_0:t_n}$ (possibly from an unseen perturbation class), predict the forward distribution $p(z_{t_n + \tau})$.

---

## 3. Dynamical Model

### 3.1 SDE Framework

The latent state evolves under an Itô SDE:

$$dz = f(z; c_e, R_e) \, dt + \sqrt{2D} \, dW$$

where:
- $f$ is the drift (defined below)
- $c_e \in \mathbb{R}^M$ is the embryo-level mode loading vector
- $R_e > 0$ is the embryo-level rate parameter
- $D > 0$ is a global scalar diffusion coefficient (shared across all embryos)
- $W$ is a standard Wiener process

### 3.2 Drift Structure

The drift decomposes as:

$$f(z; c_e, R_e) = R_e \left[ -\beta \, \nabla_z \phi_0(z) + \sum_{m=1}^{M} c_{e,m} \, (-I + S_m) \, \nabla_z \phi_m(z) \right]$$

where:

| Symbol | Description |
|--------|-------------|
| $\phi_0(z)$ | Baseline developmental potential (scalar field, shared across all embryos) |
| $\phi_m(z)$ | Mode potentials (scalar fields, $m = 1, \dots, M$) |
| $S_m$ | Antisymmetric matrix per mode ($S_m = -S_m^\top$, $\|S_m\|_F = 1$), encoding rotational dynamics |
| $\beta > 0$ | Global drift scale for baseline potential |
| $c_e \in \mathbb{R}^M$ | Mode loadings per embryo (unconstrained reals) |
| $R_e > 0$ | Rate parameter per embryo |

### 3.3 Mode Parameterization (Helmholtz)

Each mode contributes a drift $(-I + S_m) \nabla_z \phi_m(z)$. This admits a clean interpretation:

- When $S_m = 0$: mode $m$ is a pure potential (gradient) mode. Flow descends the landscape $\phi_m$.
- When $S_m \neq 0$: the gradient is rotated before application, producing spiraling or circulation dynamics around the level sets of $\phi_m$.

**Critical property**: the fixed points of mode $m$ are determined entirely by $\nabla_z \phi_m = 0$, regardless of $S_m$. The antisymmetric matrix changes the geometry of approach (spiraling vs. direct descent) without altering the topology (locations of attractors, saddles, basins). This means each mode can be visualized as a scalar landscape even when its dynamics are non-conservative.

To toggle between potential-only and Helmholtz modes: simply freeze $S_m = 0$ for the potential-only variant. This should be a runtime flag, not an architectural change.

### 3.4 Variant: Orthogonal-to-Baseline Modes

An alternative, more constrained formulation defines modes as flux fields that are orthogonal to $\nabla \phi_0$ by construction. Each mode uses the baseline potential's own gradient, rotated by an antisymmetric matrix:

$$f_m(z) = S_m \, \nabla_z \phi_0(z)$$

The total drift becomes:

$$f(z; c_e, R_e) = R_e \left[-\beta I + \sum_m c_{e,m} \, S_m \right] \nabla_z \phi_0(z)$$

This has several notable properties:

**Exact tempo-mode separation.** The rate of developmental progression under this drift is $d\phi_0/dt = \nabla \phi_0 \cdot f = -\beta R_e \|\nabla \phi_0\|^2$, because $x^\top S x = 0$ for any antisymmetric $S$. Mode loadings have zero effect on developmental tempo — they exclusively redirect trajectories across isoclines. All tempo variation lives in $R_e$ by mathematical necessity, not regularization.

**No separate mode networks.** Each mode is an antisymmetric matrix: $d(d-1)/2$ parameters (45 for $d=10$). No MLPs, no smoothness penalties, no Hessian regularization. Total mode parameters for $M=5$: 225 scalars plus the shared $\phi_0$ network. The $\mathcal{R}_1$, $\mathcal{R}_2$, $\mathcal{R}_3$ penalties are unnecessary since mode complexity is inherently bounded.

**Limitations.** Modes cannot create new branching points that don't exist in $\phi_0$. They can redirect which branch a trajectory follows, and shape how it traverses a valley, but the topology of the landscape — where valleys exist and where they split — is determined entirely by $\phi_0$. This is appropriate for a developmental landscape with no attractors (no $\nabla \phi_0 = 0$), where perturbations are expected to redirect fate decisions rather than create novel fates. Mode drift is strongest in transit regions where $\|\nabla \phi_0\|$ is large, and weakest near flat regions of the landscape.

**Closed-form solve.** The drift remains linear in $c$, so the closed-form ridge solve (§4.1) applies identically. The design matrix columns become $H_{t,m} = R_e \, S_m \, \nabla_z \phi_0(z_t) \, \Delta t$.

**Phenotypic distance in $c$-space.** With the Frobenius normalization $\|S_m\|_F = 1$ (§5.2), all modes have comparable strength and the loading vector $c_e \in \mathbb{R}^M$ is a well-scaled dynamical phenotype. Euclidean distance $\|c_a - c_b\|$ directly measures how differently two embryos interact with the developmental landscape. Combined with the scalar stage difference $|\phi_0(z_a) - \phi_0(z_b)|$, this gives a clean two-component phenotypic comparison: developmental timing (from $\phi_0$) and dynamical character (from $c$). For the full Helmholtz variant, $c$-space distances are less interpretable because mode power also depends on the $\phi_m$ network magnitudes (see §5.2).

This variant should be implemented alongside the full Helmholtz model as a model selection flag (e.g., `mode_type: "orthogonal" | "helmholtz" | "potential_only"`). Comparing the two directly tests whether perturbation effects in the data can be fully explained as redirections of baseline flow, or whether they require genuinely novel landscape features.

### 3.5 Rate Parameters

The embryo-level rate decomposes as:

$$R_e = \lambda_e \cdot \exp\left(-\beta_T (T_{\text{ref}} - T_e)\right)$$

where:
- $T_e$ is the experimentally controlled temperature for embryo $e$ (fixed, known)
- $T_{\text{ref}}$ is a reference temperature (fixed constant)
- $\beta_T$ is a global Arrhenius-like temperature coefficient (learnable or fixed from external estimate)
- $\lambda_e > 0$ is an embryo-specific residual rate factor

**Identifiability constraint**: enforce $\frac{1}{N} \sum_e \lambda_e = 1$ across the training set. This can be implemented via reparameterization: $\lambda_e = \tilde{\lambda}_e \, / \, \bar{\tilde{\lambda}}$, where $\tilde{\lambda}_e$ are unconstrained positive parameters (e.g., $\tilde{\lambda}_e = \exp(\ell_e)$ with learnable $\ell_e$).

$R_e$ modulates the **entire** drift (baseline + modes), not just the baseline. This reflects the assumption that temperature scales developmental rate uniformly. A key consequence: $R_e$ factors out of the least-squares solve for $c_e$, so mode loadings are invariant to developmental tempo.

---

## 4. Inference (Per-Embryo)

All per-embryo quantities ($c_e$, $R_e$) are inferred via closed-form solves — no inner optimization loop, no encoder network.

### 4.1 Closed-Form Solve for Mode Loadings $c_e$

Under Euler-Maruyama discretization, for transition $(z_t, z_{t+1})$ with time step $\Delta t$:

$$z_{t+1} \sim \mathcal{N}\left(z_t + f(z_t; c_e, R_e) \, \Delta t, \;\; 2D \Delta t \, I \right)$$

With $R_e$ known (or estimated separately, see §4.2), define:

- Observed displacement: $\delta_t = z_{t+1} - z_t$
- Baseline prediction: $g_t = -R_e \, \beta \, \nabla_z \phi_0(z_t) \, \Delta t$
- Mode contribution vectors: $h_{t,m} = R_e \, (-I + S_m) \, \nabla_z \phi_m(z_t) \, \Delta t$
- Residual: $r_t = \delta_t - g_t$

**Heteroscedasticity correction**: if $\Delta t$ varies across transitions, weight each row by $1 / \sqrt{2D \Delta t_i}$.

Stack all transitions into a system $R = H c_e$ where $R \in \mathbb{R}^{Td}$, $H \in \mathbb{R}^{Td \times M}$. The regularized solution is:

$$c_e^* = (H^\top H + \lambda_c I_M)^{-1} (H^\top R + \lambda_c \, c_{0,p(e)})$$

where $\lambda_c$ is the L2 regularization strength and $c_{0,p(e)}$ is the class-level prior for embryo $e$'s perturbation class.

This is an $M \times M$ linear system (e.g., $5 \times 5$ for $M=5$). Solved via `torch.linalg.solve`. Fully differentiable — gradients flow through the solve into the network parameters.

### 4.2 Closed-Form Solve for Rate $R_e$

Given $c_e^*$, the total predicted drift direction at each transition is known. $R_e$ is a scalar multiplier on the structured component. The optimal $R_e$ is:

$$R_e^* = \frac{\sum_t \delta_t^\top \hat{f}_t \, \Delta t}{\sum_t \| \hat{f}_t \|^2 \, \Delta t^2}$$

where $\hat{f}_t$ is the total structured drift (baseline + modes, before $R_e$ scaling) at $z_t$. This is a projection of observed displacements onto predicted drift directions.

**Alternating estimation**: solve for $c_e$ given $R_e$, then $R_e$ given $c_e$. Each step is closed-form. Two to three alternations suffice. Identical procedure at train and test time (no train/test asymmetry).

### 4.3 Class-Level Priors $c_{0,p}$

The class-level prior $c_{0,p}$ for perturbation class $p$ is a learnable parameter vector. Given fixed basis functions and $\lambda_c$, the optimal $c_{0,p}$ is the solution to a linear system derived from differentiating the total class likelihood — it is the data-weighted average of embryo-level $c_e^*$ values within the class, where embryos with more transitions are weighted less toward the prior.

$c_{0,p}$ can be solved analytically (nested linear solve) or optimized via backprop alongside network weights. The latter is simpler to implement initially.

---

## 5. Network Architecture

### 5.1 Potential Networks

Each of $\phi_0, \phi_1, \dots, \phi_M$ is a small MLP: $\mathbb{R}^d \to \mathbb{R}$.

- Architecture: 2 hidden layers, width 32–64, smooth activations (softplus or ELU).
- Small capacity is an intentional inductive bias toward simple landscape topography.
- Gradients $\nabla_z \phi_m$ are computed via autodiff.

### 5.2 Antisymmetric Matrices

Each $S_m$ is a learnable constant $d \times d$ antisymmetric matrix. Parameterize by learning the $d(d-1)/2$ upper-triangular entries and antisymmetrizing. Initialize at zero (pure potential modes at start of training).

**Frobenius normalization (orthogonal modes).** For the orthogonal-modes variant (§3.4), constrain $\|S_m\|_F = 1$ for all $m$. Reparameterize as $S_m = \tilde{S}_m / \|\tilde{S}_m\|_F$ where $\tilde{S}_m$ is unconstrained. This ensures that the mode loadings $c_m$ directly measure mode strength in comparable units, making Euclidean distance in $c$-space a meaningful phenotypic distance. Without this constraint, the model can trade scale between $S_m$ and $c_m$ freely, destroying the interpretability of the loading vectors.

**Frobenius normalization (Helmholtz modes).** For the full Helmholtz variant, the situation is less clean. The effective strength of mode $m$ is $\|(-I + S_m) \nabla \phi_m(z)\|$, which varies spatially because $\nabla \phi_m$ depends on $z$. Normalizing $\|S_m\|_F = 1$ still helps (it removes the $S_m$-scaling degeneracy) but does not fully equalize mode power since the $\phi_m$ networks contribute their own scale. As a practical compromise, normalize $\|S_m\|_F = 1$ for the Helmholtz variant as well, and accept that $c$-space distances are more interpretable for orthogonal modes than for Helmholtz modes.

### 5.3 Initialization

- Mode networks $\phi_m$ ($m \geq 1$): initialize weights near zero so all embryos start on the baseline landscape. Modes are recruited as training progresses.
- Baseline network $\phi_0$: standard initialization (e.g., Xavier/He).
- $S_m$ **(orthogonal modes)**: initialize the unconstrained $\tilde{S}_m$ entries at small random values (e.g., $\mathcal{N}(0, 0.01)$). The Frobenius normalization ensures $\|S_m\|_F = 1$ from the start. Cannot initialize at exactly zero under normalization (division by zero). The random init gives random mode orientations, which is appropriate since no mode direction is privileged a priori. Mode effect is still initially negligible because mode loadings $c_m$ start near zero (via the ridge prior).
- $S_m$ **(Helmholtz modes)**: initialize at zero during the potential-only phase ($S_m$ frozen). When unfreezing $S_m$ in a later build stage, switch to the normalized reparameterization with small random $\tilde{S}_m$.

---

## 6. Regularization

### 6.1 Mode Loading Regularization

L2 penalty on $c_e$, implemented as the $\lambda_c$ term in the ridge solve (§4.1). Controls shrinkage toward the class-level prior $c_{0,p}$.

### 6.2 Baseline Potential Smoothness

Penalize rapid spatial variation in $\phi_0$:

$$\mathcal{R}_0 = \mathbb{E}_{z \sim \text{data}} \left[ \| \nabla_z^2 \phi_0(z) \|_F^2 \right]$$

This encourages $\phi_0$ to define a smooth developmental landscape with broad valleys rather than noisy, highly curved features. Computed via double backward passes. Active from stage 1 training onward — $\phi_0$ smoothness is critical because $\nabla \phi_0$ appears everywhere: it defines the drift, the isoclines, and the mode directions in the orthogonal variant.

### 6.3 Mode Gradient Magnitude (Helmholtz only)

Penalize the average drift magnitude of each mode across training data:

$$\mathcal{R}_1 = \mathbb{E}_{z \sim \text{data}} \left[ \sum_m \| (-I + S_m) \nabla_z \phi_m(z) \|^2 \right]$$

This constrains modes to be "small" relative to baseline without creating adversarial coupling between $\phi_0$ and the modes.

### 6.4 Mode Potential Power (Helmholtz only)

Penalize the total scalar power of each mode across the data manifold:

$$\mathcal{R}_2 = \mathbb{E}_{z \sim \text{data}} \left[ \sum_m \phi_m(z)^2 \right]$$

This constrains modes from dominating the baseline landscape in terms of potential value, preserving the interpretability of $\phi_0$ as the primary developmental stage coordinate.

### 6.5 Hessian Smoothness (Helmholtz only)

Penalize rapid spatial variation in each potential:

$$\mathcal{R}_3 = \mathbb{E}_{z \sim \text{data}} \left[ \sum_m \| \nabla_z^2 \phi_m(z) \|_F^2 \right]$$

Computed via double backward passes. Encourages smooth, broad landscape features per mode.

### 6.6 Rate Parameter Centering

Mean-centering constraint on $\lambda_e$ across the training set: $\bar{\lambda} = 1$. Implemented via reparameterization (§3.5).

---

## 7. Training

### 7.1 Staged Training

Training proceeds in stages to enforce the interpretive hierarchy: $\phi_0$ defines baseline development, modes capture deviations.

**Stage 1: Baseline potential only.** Train $\phi_0$ network weights, $\beta$, $D$, and $R_e$ (per embryo). No modes. Hessian smoothness penalty $\mathcal{R}_0$ active. This produces the $\phi_0$-only model, which is saved as a checkpoint and serves as a permanent evaluation baseline alongside the kernel regression.

**Stage 2: Introduce modes.** Freeze $\phi_0$. Introduce mode potentials $\phi_m$, antisymmetric matrices $S_m$ (initialized at zero), class-level priors $c_{0,p}$. Train with the closed-form $c_e$ solve active. $R_e$ continues to update. Regularization terms $\mathcal{R}_1$, $\mathcal{R}_2$, $\mathcal{R}_3$ are active.

**Stage 3 (optional): Fine-tune baseline.** Unfreeze $\phi_0$ with a heavily reduced learning rate (e.g., $10\times$ lower than stage 2). This allows the baseline to adjust slightly to accommodate the modes. Monitor whether $\phi_0$ drifts significantly from its stage 1 form — if it does, the modes aren't expressive enough. Skip this stage initially.

Within each stage, the data sampling and forward pass logic are identical.

### 7.2 Data Sampling

Each training iteration:

1. **Trajectory selection with class balancing.** Sample trajectories with probability proportional to temperature-scaled class weights. For class $p$ with $N_p$ trajectories out of $N_{\text{total}}$:

$$w_p \propto \left(\frac{N_p}{N_{\text{total}}}\right)^{1 - \gamma}$$

where $\gamma \in [0, 1]$ controls balancing strength. At $\gamma = 0$, sampling is proportional to class frequency (no rebalancing). At $\gamma = 1$, all classes are sampled equally regardless of size. Default: $\gamma = 0.5$ (square-root weighting).

2. **Context window placement.** For each selected trajectory, randomly sample a start index $t_0$ uniformly from valid positions (i.e., positions where at least $W$ consecutive points are available as context and at least 1 future point exists). The context is $z_{t_0:t_0+W}$. This means the same trajectory contributes different context/target splits across epochs, exposing the model to contexts from early, middle, and late development.

3. **Target sampling.** Sample $M$ target indices uniformly from $\{t_n+1, \ldots, \min(t_{\text{end}}, t_n + K_{\text{max}})\}$, where $t_n = t_0 + W$ is the last context point. Each target is scored as an independent single-step transition from its observed predecessor (teacher forcing — see §7.3). Default: $M = 1$. Increase to 3–5 if single-step displacements are dominated by noise in the embeddings.

**Time resolution**: within any single experiment, all embryos share a uniform time step $\Delta t$. Across experiments, $\Delta t$ may differ. The model should accept $\Delta t$ as an input per experiment. The heteroscedasticity correction (weighting rows by $1/\sqrt{2D\Delta t}$) applies when training across experiments with different resolutions.

### 7.3 Loss Function

$$\mathcal{L} = -\sum_e \frac{1}{M} \sum_{m=1}^{M} \log p(z_{t_m}^{(e)} | z_{t_m - 1}^{(e)}, c_e^*, R_e^*) + \alpha_0 \mathcal{R}_0 + \alpha_1 \mathcal{R}_1 + \alpha_2 \mathcal{R}_2 + \alpha_3 \mathcal{R}_3$$

where $t_1, \ldots, t_M$ are the randomly sampled target indices (§7.2). $\mathcal{R}_0$ (baseline smoothness) is active in all stages; $\mathcal{R}_1$, $\mathcal{R}_2$, $\mathcal{R}_3$ are active only when Helmholtz modes are present. Each target is scored as a **single-step transition from the observed preceding point** (teacher forcing). The model never chains its own predictions during training — every evaluation is one Euler step from a real data point. This avoids compounding error and Euler stability issues.

The key property: $c_e^*$ and $R_e^*$ are inferred from the context window only, but the loss is evaluated on future transitions that were not used for inference. This meta-learning structure ensures $\phi_0$ is optimized to be a potential where context-inferred loadings generalize to unseen future behavior.

When $M > 1$, averaging over multiple future transitions reduces the impact of embedding noise on gradient estimates, since each individual single-step displacement may have poor signal-to-noise ratio.

The single-step transition log-likelihood under Euler-Maruyama is:

$$\log p(z_{t+1} | z_t) = -\frac{d}{2} \log(4\pi D \Delta t) - \frac{\| z_{t+1} - z_t - f(z_t; c_e^*, R_e^*) \Delta t \|^2}{4 D \Delta t}$$

### 7.4 Parameters and Their Optimization Method

| Parameter | Method | Notes |
|-----------|--------|-------|
| $\phi_0$ network weights | Backprop | Baseline potential |
| $\phi_m$ network weights | Backprop | Mode potentials |
| $S_m$ entries | Backprop | Antisymmetric matrices; init at 0 |
| $\beta$ | Backprop | Global drift scale |
| $\beta_T$ | Fixed or backprop | Arrhenius coefficient; external estimate available |
| $D$ | Backprop | Global scalar diffusion |
| $c_{0,p}$ | Backprop (or analytic) | Class-level priors |
| $c_e$ | Closed-form (ridge) | Per-embryo, inside forward pass |
| $R_e$ | Closed-form (projection) | Per-embryo, inside forward pass |
| $\lambda_e$ | Absorbed into $R_e$ | Via reparameterization |

### 7.5 Forward Pass (Single Training Step)

1. Sample trajectories (class-balanced), context windows, and $M$ target indices per trajectory (§7.2).
2. Compute $\nabla_z \phi_0(z_t)$ and $\nabla_z \phi_m(z_t)$ at all context transitions via autodiff.
3. Assemble design matrix $H$ and residual vector $R$ per embryo (context only).
4. Solve for $c_e^*$ via ridge regression.
5. Compute $R_e^*$ via scalar projection.
6. Iterate steps 3–5 once or twice (alternating $c$ and $R$).
7. For each of the $M$ target transitions: evaluate the drift $f(z_{t_m-1}; c_e^*, R_e^*)$ at the observed predecessor and compute the single-step log-likelihood of the observed target $z_{t_m}$.
8. Average the $M$ log-likelihoods per embryo. Add regularization penalties.
9. Backpropagate through everything (including the linear solve) into network weights.

---

## 8. Interpretive Framework

### 8.1 Developmental Stage

$\phi_0(z)$ defines a scalar developmental stage for every point in latent space. This is exact and global — not a post-hoc pseudotime estimate.

### 8.2 Developmental Isoclines

Level sets $\{ z : \phi_0(z) = s \}$ are surfaces of equal developmental stage. Two embeddings $z_a$ and $z_b$ with $\phi_0(z_a) = \phi_0(z_b)$ are at the same developmental stage; the distance between them on the level set is pure morphological difference.

### 8.3 Distance Decomposition

For any two embeddings $z_a, z_b$:
- **Developmental progression difference**: $\Delta s = \phi_0(z_a) - \phi_0(z_b)$
- **Morphological difference**: displacement along the shared level set (after projecting to matching stage)

This decomposition is intrinsic to the model and requires no ODE integration, no Euclidean projection, and introduces no path-dependence.

### 8.4 Dynamical Phenotype

The loading vector $c_e \in \mathbb{R}^M$ is a compact dynamical phenotype for embryo $e$. It describes which landscape features are active and at what strength. For comparing perturbation classes, the class-level $c_{0,p}$ serves as a summary phenotype.

Each mode's contribution at a given isocline can be decomposed into:
- **Gradient-parallel component**: changes developmental tempo (accelerates or retards progression)
- **Gradient-orthogonal component**: diverts to a different fate

This provides a richer phenotypic vocabulary than raw trajectory distances.

---

## 9. Baselines (Non-Parametric)

Two non-parametric baselines of increasing sophistication. Both share the same evaluation pipeline as the learned models.

### 9.1 Simple Kernel Regression

The simplest possible baseline. Given a query point $\hat{z}(t)$, find nearby training trajectory points, look at where they went, and report the weighted distribution of outcomes.

**Method.** Compute the distance from $\hat{z}(t)$ to every point $z_s^{(i)}$ across all training trajectories. Weight by a Gaussian kernel:

$$w_{i,s} = \exp\left(-\frac{\|z_s^{(i)} - \hat{z}(t)\|^2}{2\sigma^2}\right)$$

The prediction at horizon $\tau$ (in hours) is the weighted distribution of training positions at $\Delta_s = \text{round}(\tau / \Delta t_i)$ frames past each matched point. For each training point $z_s^{(i)}$, if $z_{s + \Delta_s}^{(i)}$ exists, it contributes to the prediction with weight $w_{i,s}$. Points whose trajectories terminate before $s + \Delta_s$ are dropped.

**Outputs.** The predictive distribution is a weighted point cloud. The mean prediction is the weighted centroid. Uncertainty is the weighted covariance.

**Properties.** Dead simple to implement (one function, one hyperparameter). Makes no assumptions about direction, speed, or trajectory structure. Handles single-snapshot queries. Fails for novel perturbations, degrades at long horizons (references drop out), and cannot handle bifurcations — it will smear probability across branches. This is the floor against which all other approaches are measured.

**Hyperparameters.** One: kernel bandwidth $\sigma$ (cross-validate on held-out trajectories).

---

### 9.2 Branching Particle Filter Baseline

A more sophisticated non-parametric baseline that follows matched reference trajectories forward, dynamically recruits new references, and preserves multimodal predictions through bifurcations.

#### 9.2.1 Reference Selection

Given a query point $\hat{z}(t)$ (the latest observed point):

**Step 1: Spatial filtering.** For each training trajectory, keep only points within radius $R$ of $\hat{z}(t)$. Discard any trajectory with fewer than $N$ points within $R$.

**Step 2: Local linear fit.** For each surviving trajectory, fit a linear trend through its points within $R$ (including the point closest to $\hat{z}(t)$). This yields a midpoint $p_i$ and direction vector $v_i$ per reference.

**Step 3: Fit the query window.** Fit a linear trend through the query context window (the last $W$ observed points up to and including $\hat{z}(t)$). This yields a query midpoint $p_q$ and direction vector $v_q$.

**Step 4: Weighting.** Combine spatial proximity and directional alignment into a single weight:

$$w_i = \exp\left(-\frac{\|p_i - p_q\|^2}{2\sigma_{\text{pos}}^2}\right) \cdot \left(\frac{v_i \cdot v_q}{\|v_i\| \|v_q\|}\right)^\alpha_+$$

where $(\cdot)_+$ denotes clamping negative dot products to zero (opposite-direction trajectories get zero weight) and $\alpha$ controls directional sensitivity.

**Step 5: Anchor assignment.** For each reference trajectory $i$, identify its point closest to $\hat{z}(t)$ as the anchor $t^{(i)}_{\text{anchor}}$.

**Step 6: Speed ratio.** Compute average speeds in units of latent-space-distance per hour. For the query:

$$\bar{v}_{\text{query}} = \frac{1}{W-1} \sum_{j=0}^{W-2} \frac{\|z_{j+1} - z_j\|}{\Delta t_{\text{query}}}$$

For reference $i$, over its points within $R$:

$$\bar{v}_i = \frac{1}{|P_i|-1} \sum_{t \in P_i} \frac{\|z_{t+1}^{(i)} - z_t^{(i)}\|}{\Delta t_i}$$

The speed ratio $\rho_i = \bar{v}_{\text{query}} / \bar{v}_i$ is dimensionless. This matches **developmental progress** (same latent distance covered) rather than real time elapsed, so a slow reference gets stretched to keep pace with a fast query.

#### 9.2.2 Forward Prediction

**Time stepping.** The algorithm advances in query-time steps: each step is $\Delta t_{\text{query}}$ hours. At step $k$, each active reference $i$ is at position $z_i^{(k)}$, which is the point $\rho_i \cdot k \cdot \Delta t_{\text{query}} / \Delta t_i$ frames past its anchor in reference $i$'s own timeline. Interpolate between frames when this is non-integer.

**Example.** Query at 5-minute intervals developing at speed 2.0 (latent units/hr). Reference at 10-minute intervals developing at speed 1.0. $\rho_i = 2$. Per query step (5 min), advance reference by $2 \times 5/10 = 1$ frame. The reference is slower and more coarsely sampled — both factors are accounted for.

**Initialization.** The active particle set is $\{(w_i, \text{traj}_i, t^{(i)}_{\text{anchor}}, \rho_i)\}$ from the selection step.

**Predictive distribution at step $k$.** The weighted cloud of active particle positions:

$$p(\hat{z}(t + k\Delta t_{\text{query}})) \approx \sum_i w_i^{(k)} \, \delta\left(z - z_i^{(k)}\right)$$

**Particle death.** When reference trajectory $i$ runs out of observed points, remove it from the active set and renormalize remaining weights. This naturally increases uncertainty at long horizons.

**Particle recruitment.** At each step, each active reference $i$ at position $z_i^{(k)}$ can recruit new training trajectories. A new trajectory $j$ is eligible if it has a point within radius $R$ of $z_i^{(k)}$ and extends further into the future. Its inherited weight is:

$$w_j^{\text{new}} = w_i^{(k)} \cdot \exp\left(-\frac{\|z_j - z_i^{(k)}\|^2}{2\sigma^2}\right)$$

If the same new trajectory $j$ is recruited by multiple active references, sum the inherited weights. Speed ratios for recruited trajectories are computed from their local velocity near the recruitment point relative to the recruiting reference's local velocity. This is how the reference set evolves — dying references are replaced by locally recruited successors, and new trajectories that begin midway through the prediction horizon are incorporated as they appear.

**Branch preservation.** Recruitment is local to each reference, not relative to any global centroid. At a bifurcation, references on branch A recruit near branch A; references on branch B recruit near branch B. The two branches maintain independent weight pools with no crosstalk. The predictive distribution is naturally multimodal.

**Particle cap.** If the active set exceeds $N_{\text{max}}$ particles at any step, keep only the top $N_{\text{max}}$ by weight and renormalize. For a first pass, pure top-$N$ pruning by weight is sufficient. Branches with substantial weight naturally retain particles.

**Underflow prevention.** Periodically renormalize all active weights to sum to 1. This does not affect the relative weighting structure.

#### 9.2.3 Prediction Outputs

- **Mean prediction** at horizon $k$: weighted centroid of the particle cloud (useful for metrics, but misleading near bifurcations).
- **Full predictive distribution**: the weighted point cloud itself. Compare to the dynamical model's SDE-sampled distribution via earth mover's distance.
- **Uncertainty**: weighted covariance of the particle cloud. Large eigenvalues flag bifurcation regions.
- **Multimodality detection**: cluster the particle cloud (e.g., by splitting on the first principal component of the weighted covariance); if two clusters have substantial weight, the prediction is bimodal.

#### 9.2.4 Hyperparameters

| Hyperparameter | Description | Suggested Starting Point |
|----------------|-------------|--------------------------|
| $R$ | Spatial filter radius | Scale of typical trajectory spacing |
| $N$ | Minimum points within $R$ per reference | 3–5 |
| $W$ | Query context window length (points) | 5–11 |
| $\sigma_{\text{pos}}$ | Positional kernel bandwidth | Cross-validate |
| $\alpha$ | Directional alignment exponent | 2–4 |
| $\sigma$ | Recruitment kernel bandwidth | Same as $\sigma_{\text{pos}}$ initially |
| $N_{\text{max}}$ | Maximum active particles | 200–500 |

#### 9.2.5 Properties

- Returns a full multimodal predictive distribution (weighted point cloud).
- Handles bifurcations naturally — branches maintain independent reference pools.
- Handles fragmented reference trajectories — dying references are replaced by local recruitment; new trajectories entering the prediction region are incorporated.
- Handles rate variation across embryos and experiments via per-reference speed ratios matched on developmental progress.
- No index alignment, no notion of developmental stage, no arc-length parameterization.
- No modeling assumptions. Degrades gracefully: in sparse regions, few references are recruited and uncertainty is wide; in dense regions, many references provide tight predictions.
- Fails for novel perturbations with no nearby training trajectories (by design — this is where the dynamical model should win).

---

## 10. Test Set Design

### Tier 1: Novel Perturbation (Must-Have)

Hold out one or more entire perturbation classes believed to be within the span of the mode basis. This is the primary test of whether the model learns generalizable dynamics. The kernel baseline is blind here by construction.

### Tier 2: Within-Class Generalization (Calibration)

Hold out random embryos from known classes, stratified by trajectory length. Compare performance as a function of number of observed time points. The potential model's advantage should grow as context shrinks.

### Tier 3: Out-of-Span Perturbation (Stress Test)

Hold out a perturbation class suspected to be outside the mode basis span. The model should produce poor predictions but with appropriately high uncertainty (large residuals). Confident wrong predictions indicate poor calibration.

### Tier 4: Horizon Sweep

For a well-represented class, predict at increasing horizons $\tau$. Compare degradation curves between the model and baseline. The baseline should degrade faster; if it doesn't, the learned dynamics aren't capturing long-range flow structure.

---

## 11. Evaluation Stack

All evaluation is logged to Weights & Biases (W&B). Every model checkpoint — including the kernel baseline and the $\phi_0$-only model from stage 1 — is evaluated on the same test sets to enable direct comparison.

### 11.1 Quantitative Metrics

The following are computed and logged at each evaluation checkpoint:

- **Held-out NLL**: negative log-likelihood on test transitions. Primary metric. Reported separately for each test tier (§10).
- **Per-horizon NLL**: at evaluation time, score targets at fixed horizons $k \in \{1, 2, 4, 8, 16\}$ steps from the context endpoint (teacher-forced single-step likelihoods). Reveals how prediction quality degrades with distance from the context window.
- **Earth mover's distance (EMD)**: between predicted forward distributions (sampled from the SDE) and observed positions at each horizon. Captures distributional accuracy beyond mean prediction.
- **Baseline comparisons**: all five models (§11.2) evaluated on the same test sets, always visible as reference lines on all plots. The gaps between successive models in the complexity hierarchy isolate the contribution of each component.
- **Mode utilization diagnostics**:
  - Distribution of residual norms after the $c_e$ solve (are the modes a good basis?)
  - Per-mode average $|c_{e,m}|$ (which modes are being used? With $\|S_m\|_F = 1$, these are directly comparable across modes)
  - Per-mode $c_m$ variance across embryos (modes with low variance are effectively constant and could be absorbed into $\phi_0$)

### 11.2 Five Models Always Compared

Every evaluation report shows five models in order of increasing complexity:

1. **Simple kernel regression** — nearest-point lookup, single $\sigma$ (§9.1)
2. **Branching particle filter** — direction-aware matching, local recruitment, multimodal predictions (§9.2)
3. **$\phi_0$-only** — learned baseline potential, no modes (stage 1 checkpoint)
4. **Orthogonal modes** — modes as pure redirections of baseline flow ($S_m \nabla \phi_0$, §3.4)
5. **Full Helmholtz model** — independent mode potentials with optional non-conservative dynamics (three sub-variants: potential-only with $S_m=0$, Helmholtz with learned $S_m$, and pure potential with separate $\phi_m$)

This hierarchy answers a sequence of questions: does direction-aware matching help beyond naive proximity (1→2)? Does learning dynamics help beyond non-parametric methods (2→3)? Do modes help beyond baseline dynamics (3→4)? Does allowing modes to reshape the landscape add value beyond redirecting flow (4→5)?

---

## 12. Visualization

### 12.1 Panel 1: Trajectory View

- X-axis: $\phi_0(z)$ (developmental stage)
- Y-axis: first residual PC (PCA on displacements orthogonal to $\nabla\phi_0$, computed per isocline)
- Observed trajectories as solid colored lines (by perturbation class)
- Model predictions as dashed lines with shaded uncertainty envelopes
- Baseline kernel predictions shown as point clouds

This is the primary diagnostic for whether the model captures overall flow structure. Developmental progression reads left-to-right by construction; vertical spread is morphological variation at fixed stage.

### 12.2 Panel 2: Prediction Fan

For a single selected embryo:

- Context fragment as a solid line
- Fan of 50–100 forward-simulated SDE trajectories from the model (in the same $\phi_0$ vs residual-PC space)
- Kernel baseline's weighted point cloud overlaid
- Observed future (if available) marked

This shows whether the model's uncertainty is calibrated and whether it outperforms the baseline for individual cases. Should be generated for representative embryos from each test tier.

### 12.3 Panel 3: Phenotype Space

- Plot $c_e$ vectors for all embryos, projected into their first 2 PCs
- Colored by perturbation class
- Held-out novel perturbation embryos highlighted distinctly

This reveals whether the dynamical phenotype captures meaningful biological structure: do perturbation classes cluster? Are there subpopulations? Are novel perturbations placed sensibly relative to training classes?

### 12.4 Panel 4: Mode Deflection Fields (Orthogonal Modes Variant)

For the orthogonal-modes model, the lateral effect of perturbations can be visualized directly on isocline cross-sections.

**Streamlines on isoclines.** Pick a developmental stage $s = \phi_0(z)$. On the level set, the baseline drift is purely normal (descending $\phi_0$), so all tangential flow comes from the modes: $\sum c_m S_m \nabla \phi_0$ projected onto the level set. Plot this as a 2D vector field (in residual PC coordinates) showing the direction and magnitude of lateral deflection. Sweep $s$ through developmental time to produce a movie of how the perturbation's redirecting effect evolves along development.

**Trajectory bundle comparisons.** In the $(\phi_0, \text{PC}_1, \text{PC}_2)$ space, simulate bundles of trajectories from a shared initial region under different $c$ vectors (e.g., using class-level $c_{0,A}$ vs $c_{0,B}$). Overlay bundles in different colors. This directly shows: same starting point, different mode loadings, different valley choices. The developmental stage at which bundles separate reveals when the perturbation's effect kicks in; the direction of separation reveals which fates are promoted or suppressed.

**Effective deflection potential (optional).** On each isocline, perform a Helmholtz decomposition of the tangential mode field into a gradient part and a curl part. If the gradient part dominates, extract a scalar $\psi_{\text{eff}}(z; c)$ whose level sets define "valleys" within the isocline — trajectories follow the valleys of $\psi_{\text{eff}}$ when choosing between fates. If the curl part is large, the full vector field visualization (streamlines) is necessary.

### 12.5 Uncertainty Reporting

High-dimensional uncertainty is reported in two complementary ways:

- **Visual (2D projections)**: marginal uncertainty along the $\phi_0$ axis (temporal uncertainty: "when does it reach stage X?") shown separately from uncertainty orthogonal to $\phi_0$ (fate uncertainty: "which fate does it adopt?"). Displayed in the prediction fan panel.
- **Numerical (full 10D)**: log-determinant of predicted covariance at each horizon, broken out by test tier. Tracked over training as a scalar summary of model confidence. Compared across the three model tiers (kernel, $\phi_0$-only, full).

---

## 13. Hyperparameters Summary

| Hyperparameter | Description | Suggested Starting Point |
|----------------|-------------|--------------------------|
| $M$ | Number of modes | 3–5 |
| Hidden width | MLP width per layer | 32–64 |
| Hidden depth | MLP layers per potential | 2 |
| Activation | Smooth nonlinearity | Softplus |
| $\lambda_c$ | L2 penalty on mode loadings | Cross-validate |
| $\alpha_0$ | Baseline Hessian smoothness penalty weight | Tune |
| $\alpha_1$ | Mode gradient penalty weight | Tune |
| $\alpha_2$ | Mode potential power penalty weight | Tune |
| $\alpha_3$ | Hessian smoothness penalty weight | Tune |
| $W$ | Context window length (points) | 5–11 |
| $M_{\text{targets}}$ | Future transitions scored per fragment | 1 (increase to 3–5 if noisy) |
| $K_{\text{max}}$ | Max horizon for target sampling (steps) | All remaining points |
| $\gamma$ | Class balancing temperature | 0.5 |

---

## 14. Future Extensions (Tabled)

The following were discussed and deemed valuable but deferred in the interest of starting lean:

- **Embryo-specific local drift correction ($v_e$)**: add a per-embryo constant drift vector $v_e \in \mathbb{R}^d$ with a spatial decay kernel $w_e(z) = \exp(-\|z - \bar{z}_e\|^2 / 2\sigma_w^2)$ centered on the embryo's observed data. The term $v_e \cdot w_e(z)$ would handle OOD trajectories that the mode basis cannot adequately represent, applying near the observed data and decaying to zero for forward prediction. $v_e$ would enter the closed-form ridge solve as additional columns in the design matrix (expanding $c_e$ to $[c_e; v_e]$), with a separate regularization weight $\lambda_v > \lambda_c$ to ensure modes get first priority. Pursue if residual analysis after mode fitting shows systematic per-embryo biases that the mode basis cannot capture.
- **Underdamped Langevin (momentum)**: augment state with velocity to capture ballistic arcs. Doubles state dimension, adds damping parameter $\gamma$. Breaks closed-form $c$ solve. Pursue if first-order model fails to capture curved trajectories.
- **Diagonal diffusion**: replace scalar $D$ with per-dimension $D_i$. Minimal complexity increase, likely worthwhile if residual analysis shows anisotropic noise.
- **Time-varying potential**: allow landscape to deform over developmental time. Handles "same state, different stage, different future." Partially preserves closed-form structure if time dependence is simple.
- **Optimal transport predictive distance**: $d(z_a, z_b) = W_p(\mu_{z_a}, \mu_{z_b})$ for dynamically meaningful embryo comparison near bifurcations.

---

## 15. Implementation Notes for Claude Code

### 15.1 Framework

Plain PyTorch. No PyTorch Lightning initially — the non-standard forward pass (inner linear solves, alternating estimation, multi-step simulation) benefits from full transparency during debugging. Lightning can be introduced as a later refactor once core mechanics are validated.

### 15.2 Build Sequence

Implement and test in this order. Each step produces a runnable, evaluable artifact.

1. **Data loading and fragment sampling.** Define the data interface: per embryo, provide trajectory tensor, experiment-level $\Delta t$, temperature $T_e$, perturbation class label. Implement class-balanced sampling ($\gamma$), random context window placement, and $M$-target sampling (§7.2). These are required infrastructure for all subsequent model training.
2. **Evaluation and visualization stack.** Build the W&B logging, metric computation (§11), and visualization panels (§12) against dummy/random predictions. This infrastructure must exist before any model is trained.
3. **Simple kernel regression (§9.1).** Nearest-point lookup baseline. One hyperparameter. Implement in an afternoon. This is the absolute floor.
4. **Branching particle filter (§9.2).** Direction-aware matching, local recruitment, multimodal predictions. More complex but substantially more powerful. Implement in layers: first get selection and single-shot prediction working (follow references forward, no recruitment), then add recruitment logic.
5. **$\phi_0$-only model (Stage 1).** Single baseline potential, $\beta$, $D$, $R_e$. No modes. Includes: Hessian smoothness penalty $\mathcal{R}_0$ on $\phi_0$ (§6.2), rate identifiability constraint mean($\lambda_e$) = 1 (§3.5), teacher-forced multi-target loss (§7.3). Train and evaluate. Save checkpoint — this becomes the permanent $\phi_0$-only reference.
6. **Orthogonal modes (§3.4).** Freeze $\phi_0$. Introduce antisymmetric matrices $S_m$ with Frobenius normalization $\|S_m\|_F = 1$ (§5.2), closed-form $c_e$ solve, class-level priors $c_{0,p}$. Train and evaluate. This is the simplest mode variant and tests whether redirecting baseline flow is sufficient.
7. **Full Helmholtz modes.** Introduce independent mode potentials $\phi_m$ with $S_m = 0$ (potential-only modes). Activate Helmholtz-specific regularization $\mathcal{R}_1$, $\mathcal{R}_2$, $\mathcal{R}_3$ (§6.3–6.5). Train and evaluate. Compare to orthogonal modes to assess whether independent landscape features add value.
8. **Unfreeze $S_m$ on full model.** Allow non-conservative dynamics in the Helmholtz modes. Compare to potential-only modes.

### 15.3 Key Implementation Constraints

- All per-embryo inference ($c_e$, $R_e$) must be batched and differentiable. Use `torch.linalg.solve` for the ridge systems.
- Mode type selection should be a single config flag: `mode_type: "orthogonal" | "helmholtz" | "potential_only"`. The orthogonal variant uses $S_m \nabla \phi_0$ (no $\phi_m$ networks). The Helmholtz variant uses $(-I + S_m) \nabla \phi_m$. The potential-only variant is Helmholtz with $S_m$ frozen at zero.
- The kernel baselines (simple and particle filter) should be standalone modules sharing the same data loading and evaluation pipeline.
- The five-model comparison (simple kernel, particle filter, $\phi_0$-only, orthogonal modes, full Helmholtz) must be trivially reproducible at any point during development.
