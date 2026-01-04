# Parametres de la simulation (params.json)

Tous les parametres se configurent dans `particle_sim3d/params.json`.
L'application recharge automatiquement le JSON quand tu l'enregistres (auto-reload).

Astuce (menu overlay):
- Ouvre le menu avec `F1` ou `Tab`.
- Tape directement une valeur numerique (ou `ENTER` sur un parametre numerique) => edition directe.
- `PGUP/PGDN` => change le pas d'increment pour `LEFT/RIGHT` et `+/-`.

Important:
- Les unites sont **arbitraires** (ceci est une "toy simulation"). Les valeurs sont coherentes entre elles, mais pas calibre-es en SI.
- Certains parametres ne servent **que pour l'initialisation** (ils ne changent rien si tu modifies en cours de simulation, sauf si tu fais `R` reset).
- Les limites ("bounds") sont des **parois rigides** avec rebond (`bounce`).

## Fenetre / rendu

- `width`, `height` (int): taille de la fenetre au demarrage.
- `background` ([r,g,b]): couleur de fond (0..255).
- `point_size` (float): taille des points OpenGL.
- `camera_min_distance` (float): distance minimale de zoom.
- `camera_max_distance` (float): distance maximale de zoom.
- `color_gradient` (bool): active un degrade de couleur sur M+ selon force/vitesse/densite/proximite.
- `color_gradient_mode` (str): selectionne ce que le degrade represente.
  - `mix` (defaut): mix vitesse + force + densite + proximite.
  - `speed`: vitesse (||v||).
  - `force`: norme de l'acceleration (||a||).
  - `density`: densite locale (cellules 3D autour de M+).
  - `proximity`: proximite du centre (1.0 = proche, 0.0 = loin).
  - `temperature`: proxy de temperature = energie cinetique relative (m * ||v||^2).
- `split_screen` (bool): ecran partage M+ / M- (2 vues).
- `multi_view` (bool): multi-vues (meme scene, cameras independantes).
- `multi_view_count` (int): nombre de vues (1..3).
- `sprite_enabled` (bool): rend les particules en "sprites" doux (points avec halo).
- `sprite_scale` (float): multiplicateur de taille pour les sprites.
- `grid_enabled` (bool): affiche une grille dans le plan XY.
- `grid_step` (float): pas de la grille (espacement).
- `trails_enabled` (bool): affiche des trainées (trails).
- `trails_length` (int): nombre de frames conservees pour les trainées.
- `trails_stride` (int): dessine 1 particule sur N pour les trainées.
- `trails_alpha` (float 0..1): opacite max des trainées.
- `trails_pos_only` (bool): ne garder que les trainées M+.
- `trails_blur` (bool): active un effet de flou (line smoothing si supporté).
- `trails_width` (float): largeur des trainées.
- `target_fps` (int): cadence de rendu (le pas de temps est pilote par cette cadence).
- `mac_compat` (bool): si `true`, passe sur un contexte OpenGL plus tolerant sur macOS (pas de MSAA, point smoothing coupe, shadow window desactivee) pour reduire la charge GPU/Metal.

## Temps et integration

- `time_scale` (float): facteur d'acceleration du temps simule.
  - Exemple: `10.0` = 10x plus vite.
  - Raccourcis: `+/-`, `1..5`, `F` (1x <-> 10x).

- `damping` (float 0..1): multiplie les vitesses a chaque tick (perte d'energie numerique / frottement).
  - Plus proche de `1.0` => moins de dissipation.
  - Attention: comme c'est applique "par frame", l'effet depend du FPS et de `time_scale`.

- `max_speed` (float): limite la norme de vitesse (stabilite numerique).
  - Mets `0` pour desactiver le clamp.

- `bounce` (float 0..1): coefficient de restitution sur les parois (rebond).
  - `1.0` = rebond elastique
  - `0.0` = pas de rebond (la composante normale est annulee)

## Limites spatiales (bounds)

Tu peux choisir entre une **boite** (bound box) et une **sphere** (en pratique: un ellipsoide si on l'aplatit).

### Mode "box"

- `bound_mode`: `"box"`
- `bounds` (float): demi-taille de la boite. Domaine: `x,y,z` dans `[-bounds, +bounds]`.

Collision: si une particule depasse, elle est remise sur le plan et la composante de vitesse correspondante est inversee avec `bounce`.

### Mode "sphere" (sphere applatie)

- `bound_mode`: `"sphere"`
- `bound_sphere_radius` (float): rayon `R`.
- `bound_sphere_flatten_z` (float): facteur d'aplatissement sur l'axe Z.
  - `1.0` = sphere
  - `< 1.0` = sphere "applatie" (oblate), type "galette"
  - `> 1.0` = sphere "etiree" (prolate)
- `bound_wire_visible` (bool): affiche la limite en fil de fer.
- `bound_wire_opacity` (float 0..1): opacite du fil de fer.

La condition "a l'interieur" est:

`x^2 + y^2 + (z^2)/(flatZ^2) <= R^2`

Donc l'amplitude max en Z est environ `R * flatZ`.

Collision: projection sur la surface + reflection de la vitesse sur la normale (avec `bounce` uniquement sur la composante normale).

## Population Janus (M+ / M-)

Le modele utilise deux populations:
- `M+` (signe +1): particules visibles / "galaxie"
- `M-` (signe -1): environnement "gemellaire"

Parametres:

- `population_mode` (str): choix du mode de comptage.
  - `"total"`: utilise `particle_count` + `negative_fraction`.
  - `"explicit"`: utilise `positive_count` + `negative_count`.

- `negative_fraction` (float 0..1): fraction de particules `M-` (le reste est `M+`).
  - Exemple: `0.67` => ~67% `M-`.
  - Ignore si `population_mode="explicit"`.
- `merge_enabled` (bool): fusionne les particules dans les zones de forte densite.
- `merge_radius` (float): taille de cellule (monde) pour detecter la densite.
- `merge_min_count` (int): nombre minimum de particules dans une cellule pour fusionner.
- `merge_mode` (str): `"all"` pour fusionner M+/M-, `"M+"` pour ne fusionner que la galaxie.
- `merge_max_cells` (int): limite le nombre de cellules fusionnees par frame (0 = illimite).
- `merge_temp_threshold` (float): temperature minimale (proxy = moyenne de `m * |v|^2` par cellule) pour autoriser la fusion.
  - `0` = desactive.
  - La fusion cree une nouvelle particule plus grosse (masse totale, vitesse moyenne ponderee).
- `merge_blob_scale` (float): multiplicateur de taille pour les particules fusionnees (blob).

- `mass_positive` (float > 0): masse (magnitude) d'une particule `M+`.
- `mass_negative` (float > 0): masse (magnitude) d'une particule `M-`.

Ces masses sont **independantes**.
Dans les interactions, on utilise une "charge signee" `q = s * m` (voir `particle_sim3d/MODEL_JANUS.md`).

## Gravite (approximation Newtonienne)

- `janus_enabled` (bool): active/desactive les forces.
- `janus_g` (float): constante de gravite effective `G` (plus grand => forces plus fortes).
- `force_backend` (str): methode de calcul des forces.
  - `cpu` (defaut): Barnes-Hut sur CPU (~O(N log N)).
  - `metal`: calcul direct sur GPU (~O(N^2)), necessite `pyobjc-framework-Metal` + `numpy`.
  - `cpu_direct`: calcul direct sur CPU (~O(N^2)) pour comparer au backend Metal.
- `force_tile_size` (int): taille des blocs pour `metal`/`cpu_direct` (influence la perf).
- `force_debug` (bool): affiche des diagnostics en console pour depanner les backends forces.
- `softening` (float): adoucissement `eps` pour eviter les singularites a tres courte distance.
  - Plus petit => forces plus "dures" (mais plus instable numeriquement).
  - Plus grand => interactions plus lisses.
- `theta` (float): parametre Barnes-Hut (precision vs vitesse).
  - Plus petit => plus precis, plus lent.
  - Plus grand => plus rapide, moins precis.

## Conditions initiales (init)

- `init_mode`:
  - `"janus_galaxy"` (defaut): galaxie `M+` dans une lacune, et `M-` en amas autour.
  - `"random"`: distribution aleatoire uniforme, vitesses aleatoires.

### Parametres communs a l'init

- `population_mode` (str): choix du mode de comptage.
  - `"total"`: utilise `particle_count` + `negative_fraction`.
  - `"explicit"`: utilise `positive_count` + `negative_count`.
- `particle_count` (int): nombre total de particules (utilise si `population_mode="total"`).
- `positive_count` (int): nombre de particules `M+` (utilise si `population_mode="explicit"`).
- `negative_count` (int): nombre de particules `M-` (utilise si `population_mode="explicit"`).
- `seed` (int): graine aleatoire (reproductibilite).

### Init "random"

- `initial_speed` (float): vitesse initiale max pour les particules.

### Init "janus_galaxy"

#### Lacune / echelle globale

- `void_radius` (float): rayon de la lacune au centre (zone vide en `M-`).

#### Galaxie M+ (disque)

Le disque est dans le plan XY, avec une petite epaisseur en Z.

- `galaxy_radius` (float): rayon max du disque.
- `galaxy_scale_length` (float): longueur d'echelle (profil exponentiel).
- `galaxy_thickness` (float): epaisseur (sigma) sur Z.

Rotation initiale (courbe de rotation parametree, purement phenomenologique):
- `galaxy_vmax` (float): vitesse azimutale asymptotique.
- `galaxy_turnover` (float): echelle radiale de montee de la rotation.
- `galaxy_sigma_v` (float): dispersion de vitesse (bruit).

#### Environnement M- (amas)

- `negative_clump_count` (int): nombre d'amas `M-`.
- `negative_clump_sigma` (float): taille d'un amas (dispersion spatiale).
- `negative_sigma_v` (float): dispersion de vitesse initiale des particules `M-`.
- `negative_vphi_scale` (float): facteur de rotation initiale pour `M-`.
  - `0.0` = pas de rotation, `1.0` = meme rotation que `M+`, `< 0` = contre-rotation.
- `negative_on_boundary` (bool, mode sphere uniquement): place toutes les particules `M-` sur la surface de la bound sphere.

## Conseils pratiques

- Pour voir evoluer plus vite: augmente `time_scale` (et/ou `janus_g`).
- Si "tout explose" / ca devient instable: augmente `softening`, baisse `janus_g`, baisse `time_scale`, et/ou baisse `max_speed`.
- Pour isoler la galaxie: passe en mode sphere et aplatis en Z (`bound_mode="sphere"`, `bound_sphere_flatten_z < 1.0`).
