# Simulation 3D (Janus A) - rendu OpenGL

Cette version rend des particules en 3D via **OpenGL** (acceleration GPU) et calcule les forces sur CPU avec un **octree Barnes-Hut** (~ `O(N log N)`).

Le mode par defaut cree une galaxie `M+` dans une **lacune** entouree par une distribution `M-` (mode `init_mode="janus_galaxy"`).

## Prerequis

- Python 3.10+
- Dependence pour le rendu : `pyglet`
- Optionnel (GPU Metal): `pyobjc-framework-Metal`, `numpy`

Installation :

```powershell
pip install pyglet
```

Metal (optionnel):

```powershell
pip install pyobjc-core pyobjc-framework-Metal numpy
```

## Lancer

```powershell
python -m particle_sim3d.main
```

## Parametrage

Les parametres sont dans `particle_sim3d/params.json`.

- **Auto-reload** : si tu modifies le JSON et l'enregistres, l'app recharge automatiquement (2 fois/sec max).
- **Vitesse** : `time_scale` accelere le temps simule (ex: `10.0` = 10x plus vite).
- **Menu overlay** : `F1` (ou `Tab`) affiche un menu pour ajuster des parametres (fleches + `+`/`-`).
- Dans le menu: tape directement une valeur numerique (ou `ENTER` pour editer), et `PGUP/PGDN` change le pas d'increment.
- **Camera** : clic gauche + glisser pour tourner, molette pour zoomer, `T` follow, `G` focus, `H` fit. Limites via `camera_min_distance` / `camera_max_distance`.
- **Fil de fer bounds** : en mode `bound_mode="sphere"`, tu peux afficher la limite en fil de fer (`bound_wire_visible`) et ajuster son opacite (`bound_wire_opacity`) via le menu.
- **M- sur la bound sphere** : active `negative_on_boundary` (menu) pour coller toutes les particules M- sur la surface de la bound sphere.
- **Masses** : `mass_positive` et `mass_negative` sont independantes (magnitude par particule pour `M+` et `M-`).
- **Fusion** : `merge_enabled`, `merge_radius`, `merge_min_count`, `merge_mode`, `merge_max_cells`, `merge_temp_threshold`, `merge_blob_scale` pour fusionner les zones tres denses (nouvelle particule plus grosse).
- **Comptage** : `population_mode="total"` utilise `particle_count` + `negative_fraction`, `population_mode="explicit"` utilise `positive_count` + `negative_count`.
- **Couleurs** : `color_gradient=true` active un degrade sur M+. `color_gradient_mode` choisit la mesure (mix/speed/force/density/proximity/temperature).
- **Sprites** : `sprite_enabled=true` + `sprite_scale` pour des points plus doux.
- **Grille** : `grid_enabled=true` + `grid_step` pour afficher une grille XY.
- **Trainées** : `trails_enabled=true`, `trails_length`, `trails_stride`, `trails_alpha`, `trails_pos_only`, `trails_blur`, `trails_width`.
- **Split** : `split_screen=true` affiche M+ et M- cote a cote (2 vues).
- **Multi-vues** : `multi_view=true` et `multi_view_count=3` pour 3 cameras independantes.
- **GPU Metal** : `force_backend="metal"` active le calcul des forces sur GPU (calcul direct ~O(N^2)).
- **Comparaison** : `force_backend="cpu_direct"` fait le meme calcul direct sur CPU pour comparer.
- **Tuiles** : `force_tile_size` ajuste la taille des blocs pour `metal`/`cpu_direct`.
- **Debug** : `force_debug=true` affiche des diagnostics en console (metal/cpu_direct).
- **Doc parametres** : voir `particle_sim3d/PARAMETERS.md`.
- **macOS** : `mac_compat=true` (defaut) force un contexte OpenGL sans MSAA/point smoothing (moins de fill-rate) et coupe la shadow window pyglet pour limiter les soucis OpenGL/Metal. Mets `false` si tu veux forcer le MSAA.

Touches :
- `Espace` : pause
- `+ / -` : accelerer / ralentir (vitesse du temps)
- `1..5` : presets (1x, 2x, 5x, 10x, 20x)
- `F` : toggle fast-forward (1x <-> 10x)
- `V` : afficher/masquer HUD (rotation curve, m2, etc.)
- `X` : filtre d'affichage (all -> pos -> neg)
- `B` : afficher/masquer le fil de fer de la bound sphere (si `bound_mode="sphere"`)
- `P` : screenshot PNG dans `particle_sim3d/output/`
- `O` : demarre/stoppe l'enregistrement video (frames PNG + MP4 si `ffmpeg` est present)
- `C` : reset camera
- `R` : reset (recree la distribution)
- `S` : sauver JSON
- `L` : charger JSON
- `E` : exporter CSV (positions/vitesses dans `particle_sim3d/output/`)

## Modele Janus (equations)

Voir `particle_sim3d/MODEL_JANUS.md` pour les equations simplifiees (approximation newtonienne) et le sens des parametres.

## Architecture

Le projet est organise en modules specialises :

```
particle_sim3d/
├── main.py              # Point d'entree
├── app.py               # Application principale
├── sim.py               # Simulation physique
├── renderer_pyglet.py   # Rendu OpenGL/Pyglet
├── params.py            # Parametres et persistance
│
├── physics.py           # Calcul des forces (CPU/Barnes-Hut)
├── octree.py            # Structure octree pour Barnes-Hut
├── init_conditions.py   # Distribution initiale des particules
├── metal_backend.py     # Backend GPU Metal (macOS)
│
├── camera.py            # Etat et utilitaires camera
├── camera_controller.py # Controleur multi-vues
├── drawing.py           # Utilitaires de dessin OpenGL
├── ui_overlay.py        # Interface 2D (legende, menu)
├── recording.py         # Enregistrement video
│
├── config_groups.py     # Configuration menus et parametres
├── callbacks.py         # Handlers clavier
├── trail_manager.py     # Gestion des trainees
├── color_mapper.py      # Gradients de couleur
└── menu.py              # Constantes menu/UI
```

### Modules principaux

| Module | Lignes | Description |
|--------|--------|-------------|
| `sim.py` | ~1250 | Simulation N-corps avec octree |
| `app.py` | ~1295 | Application et callbacks |
| `renderer_pyglet.py` | ~1430 | Rendu 3D OpenGL |

### Modules extraits (refactoring)

| Module | Source | Description |
|--------|--------|-------------|
| `physics.py` | sim.py | Solveurs Barnes-Hut et direct |
| `init_conditions.py` | sim.py | Generateurs de distributions |
| `camera.py` | renderer | Etat camera orbitale |
| `drawing.py` | renderer | Grille, wireframe, trails |
| `recording.py` | renderer | Capture video ffmpeg |
| `ui_overlay.py` | renderer | Legende, menu, FPS |
| `config_groups.py` | app.py | MENU_GROUPS, PARAM_HINTS |
| `callbacks.py` | app.py | KeyHandler, TextInputHandler |
| `trail_manager.py` | app.py | TrailManager |
| `color_mapper.py` | app.py | ColorMapper, gradients |
