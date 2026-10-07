# Inverter_teaching
few dynamic representation of inverters to understand behavior of PMW / PLL / Control etc


# Simulateur d'onduleur triphasé : grid-following vs grid-forming

Simulateurs interactifs (Python / Tkinter / Matplotlib) d'un onduleur triphasé
connecté à un **réseau infini à travers une impédance réglable**, avec un schéma
animé et les principaux signaux en temps réel.

Deux stratégies de commande à comparer sur le même réseau :

| | **Grid-following** | **Grid-forming (VSM)** |
|---|---|---|
| Synchronisation | PLL synchrone (SRF, un seul PI) | Machine synchrone virtuelle (équation d'oscillation) |
| Grandeur commandée | Courant (régulation PI dans le repère dq) | Tension (angle et amplitude de la référence) |
| Consignes | P*, Q* | P*, Q*, inertie H, amortissement D, droop Q–V |
| Point faible illustré | Réseau faible (PLL qui oscille) | Perte de synchronisme si P* > Pmax |

## Fichiers

| Fichier | Contenu |
|---|---|
| `onduleur_gfl_50hz.py` | Grid-following, **échelle de temps réelle** (50 Hz, MLI 5 kHz) |
| `onduleur_vsm_50hz.py` | Grid-forming (VSM), **échelle de temps réelle** |
| `onduleur_gfl_schema.py` | Grid-following, temps **ralenti** (réseau à 2 Hz, MLI 50 Hz) |
| `onduleur_vsm_schema.py` | Grid-forming (VSM), temps **ralenti** |
| `onduleur_sim.py`, `onduleur_sim_triphase.py`, `onduleur_sim_schema.py` | Premières versions pédagogiques (PLL seule, sans réseau ni impédance) |

Les versions « 50 Hz » ont les mêmes paramètres qu'en pratique ; les versions
« ralenties » sont plus simples à observer sans réglage de vitesse.

## Installation et lancement

```bash
pip install numpy matplotlib      # Tkinter est fourni avec Python (sous Linux : sudo apt install python3-tk)
python onduleur_gfl_50hz.py
python onduleur_vsm_50hz.py
```

## Ce qui est simulé

Chaîne : **bus DC → onduleur 3 bras (MLI sinus-triangle) → filtre LC → impédance Zg → réseau infini**.

- Les 3 bras sont réellement commutés (porteuse triangulaire commune).
- Modèle électrique en repère αβ (réseau équilibré, 3 fils) :
  `Lf·di_L/dt = v_inv − v_pcc`, `C·dv_C/dt = i_L − i_g` (avec une résistance d'amortissement Rd en série avec C),
  `Lg·di_g/dt = v_pcc − v_g − Rg·i_g`.
- Système en per-unit : 10 kVA, 325 V crête, 15,8 Ω. `Xf = 0,15 pu` ; `Rg` et `Xg` réglables.
- Index de modulation `m = E / (Vdc/2)` **limité à 1** : si Vdc est trop faible, la MLI sature
  (signalé en rouge sur le schéma).

### Grid-following
PLL SRF sur la tension au point de raccordement → consignes `i_d* = P*/(1,5·v_d)`,
`i_q* = −Q*/(1,5·v_d)` (+ compensation du condensateur) → PI de courant avec découplage et
prédiction de tension, mis à jour aux sommets de la porteuse → limitation de courant et anti-windup.

### Grid-forming (VSM)
`2H·dΔω/dt = P* − P − D·Δω`, `θ = ∫ ω0(1+Δω) dt`, `E = E0·(1 + nq·(Q* − Q))`.
Pas de PLL ni de boucle de courant : la tension de référence est directement `E·sin(θ)`.

## Interface

- **Schéma animé** : interrupteurs du bras A qui commutent, voyant de verrouillage / de synchronisme,
  cadran (angle réseau vs angle PLL ou VSM), cadre rouge en cas de saturation.
- **Graphiques** : tension DC, sortie MLI, tension au point de raccordement, courant dans Zg, réseau,
  modulante / porteuse, P et Q (avec consignes), courants dq ou fréquence.
- **Onglets de commande** :
  - *Réseau* : amplitude, phase, **saut de phase** (valeur à saisir + bouton), fréquence,
    harmoniques 5 et 7, impédance `Rg` / `Xg` (réseau faible ou fort, SCR ≈ 1/Xg).
  - *Points* / *VSM* : consignes P*, Q* et réglages de la commande.
  - *PLL/I* (grid-following) : bande passante du courant, ωn, ζ et fréquence nominale de la PLL.
  - *Onduleur* : Vdc, fréquence de MLI (log), coupure du filtre LC.
  - *Affich.* : durée affichée (0,5 ms à 1 s) et **vitesse de simulation** (×0,001 à ×1 du temps réel).
- Boutons : pause, réinitialisation de la PLL, resynchronisation.

### Conseils d'utilisation
- Pour voir la commutation (MLI 5 kHz), réduis la durée affichée à 1–2 ms et ralentis la simulation.
- Pour voir les transitoires de régulation, monte la durée à 200–1000 ms.
- Grid-following : garde la bande passante du courant et ωn nettement sous la fréquence de MLI,
  sinon la boucle devient instable (un avertissement s'affiche).
- Si le PC ne suit pas, la simulation ralentit d'elle-même (budget de calcul par image).

## Limites

- Réseau équilibré uniquement, 3 fils, une seule phase tracée.
- MLI idéale (pas de temps mort, pertes de commutation ni limitation de courant matérielle côté VSM).
- Le VSM n'a pas de boucle de courant interne : un saut de phase ou un réseau très faible
  provoque de fortes surintensités (signalées dans l'interface).
- Outil pédagogique : les réglages par défaut sont choisis pour être stables et lisibles, pas optimisés.

## Licence

TO DO !!
