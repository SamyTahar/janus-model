# Janus (JCM) - toy model used in this project

This project implements a **simplified Newtonian toy model** inspired by the "Janus" idea (two coupled sectors).
It is **not** a full relativistic implementation and should be treated as a sandbox for qualitative behavior.

## Two populations

Each particle has:

- a position vector `r_i = (x_i, y_i, z_i)`
- a velocity `v_i`
- a **sign** `s_i` in `{+1, -1}`
  - `s_i = +1` : population `M+` (visible/ordinary matter)
  - `s_i = -1` : population `M-` (gemellar / "negative" sector)
- a **mass magnitude** `m_i > 0` (always positive)

We define a signed "charge":

`q_i = s_i * m_i`

## Interaction law (Newtonian approximation)

We use a softened pair interaction:

`F_i<-j = G * q_i * q_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)`

The acceleration is computed with a **positive inertial mass**:

`a_i = F_i / m_i`

So:

`a_i = G * (q_i / m_i) * sum_{j!=i} q_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)`

Since `q_i / m_i = s_i`, the rule becomes:

`a_i = G * s_i * sum_{j!=i} q_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)`

This gives the requested behavior:

- same sign: `(+,+)` and `(-,-)` => attraction
- opposite signs: `(+,-)` => repulsion
- action/reaction: the pair forces have the same magnitude and opposite direction
- no "runaway" from negative inertial mass (because `m_i` is always positive here)

In the code:

- `G` is `params.janus_g`
- `eps` is `params.softening`
- `q_j` is built as `q_j = s_j * m_j`

## Barnes-Hut octree acceleration

The total acceleration uses an octree (Barnes-Hut):

- a cell stores a total charge `Q_cell = sum q_j`
- and a center of charge `R_cell = (sum q_j * r_j) / Q_cell`

If `size / distance < theta`, the cell is approximated as a monopole:

`a_i += G * s_i * Q_cell * (R_cell - r_i) / (|R_cell - r_i|^2 + eps^2)^(3/2)`

`theta` is controlled by `params.theta`.

## Initial conditions: galaxy in a void (lacune)

Mode `init_mode = "janus_galaxy"` builds:

- `M+` : an exponential disk in the `(x,y)` plane, with small thickness in `z`, inside `void_radius`
- `M-` : a clumpy distribution outside the void, biased toward the inner shell to form a "potential barrier"

The dissymmetry is controlled by:

- `negative_fraction` (fraction of particles in `M-`)
- `mass_positive` (mass magnitude per particle in `M+`)
- `mass_negative` (mass magnitude per particle in `M-`, magnitude only)

A simple mass dominance indicator is shown in the window caption:

`(~ratio) = (N_- * mass_negative) / (N_+ * mass_positive)`

## Measured outputs (HUD)

The HUD (key `V`) displays:

- a crude `m=2` mode indicator for `M+` (bar/spiral proxy)
- a simple rotation curve estimate for `M+`:
  - bins in radius `r`
  - mean `|v_phi|` per bin

These are qualitative indicators only.

## Visual deliverable

- Two populations are colored differently:
  - `M+` : blue-ish
  - `M-` : orange-ish
- Use `X` to cycle filters: `all -> pos -> neg -> all`
- Use `P` to save a screenshot to `particle_sim3d/output/`

## TODO

- Add a PM/FFT (Particle-Mesh) solver and validate k=0 handling plus force reconstruction (mass assignment + gradient).
