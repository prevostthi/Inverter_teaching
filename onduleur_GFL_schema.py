#!/usr/bin/env python3
"""
Onduleur TRIPHASÉ "grid-following" (PLL + régulation de courant dq)
connecté à un RÉSEAU INFINI à travers une impédance Zg = Rg + jXg.

Même présentation que la version grid-forming (VSM) : schéma animé dessiné par
le script, mêmes graphiques, mêmes réglages du réseau / de Zg / de la MLI.

    BUS DC --> ONDULEUR (3 bras) --> FILTRE LC --> Zg (Rg, Lg) --> RÉSEAU INFINI
                   ^                       |
                   |                       | mesure de v_pcc et i_L
             MODULATEUR MLI <-- v_ref -- GRID-FOLLOWING : PLL -> consignes i_dq* -> PI dq -> abc

Principe de la commande (l'onduleur "suit" le réseau, il ne le forme pas) :
    * PLL synchrone (SRF) à un seul PI, alimentée par la tension au point de
      raccordement v_pcc -> angle θ̂ ;
    * consignes de courant dans le repère dq (d aligné sur v_pcc) :
          i_d* = P* / (1,5·v_d)        i_q* = −Q* / (1,5·v_d)  (+ compensation du filtre C)
    * régulateurs PI de courant (découplage + prédiction de v_pcc), mise à jour
      synchronisée avec la porteuse (aux sommets du triangle) ;
    * limitation de courant et anti-windup quand la MLI sature (m > 1).

Modèle physique (repère αβ, réseau équilibré, 3 fils) :
    Lf·di_L/dt = v_inv − v_pcc − Rf·i_L
    C·dv_C/dt  = i_L − i_g                  (avec Rd en série avec C : amortissement)
    Lg·di_g/dt = v_pcc − v_g − Rg·i_g
Les 3 bras sont réellement commutés (MLI sinus-triangle).

Échelle de temps RÉELLE : réseau à 50 Hz, MLI à 5 kHz, filtre LC à 1 kHz.
Pour voir les phénomènes, ralentis la simulation avec le curseur de vitesse
(jusqu'à ×0,001) et règle la durée affichée (de 0,5 ms à 1 s).

Dépendances :  pip install numpy matplotlib
"""

import math
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arc
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ----------------------------------------------------------------------------
# Paramètres de simulation
# ----------------------------------------------------------------------------
DT = 5e-6                 # pas de calcul (s) -> 200 kHz (MLI et filtre à l'échelle réelle)
N_MAX = 4000              # nombre maximal de points affichés par courbe
WINDOW = 0.04             # durée affichée par défaut (s) : 2 périodes à 50 Hz
FRAME_MS = 33             # période de rafraîchissement (ms)
FRAME_BUDGET = 0.026      # temps de calcul maximal par image (s)

DEF_FCAR = 5000.0          # fréquence de porteuse MLI par défaut (Hz)
DEF_FC = 1000.0            # fréquence de coupure du filtre LC par défaut (Hz)
DEF_VDC = 800.0           # tension DC par défaut (marge pour injecter P et Q)

# Taille de la figure (pouces) : calculée automatiquement d'après la taille de
# l'écran au lancement (voir App.__init__). Les polices suivent (FS).
FIG_W = 11.0
FIG_H = FIG_W / 1.5
FS = FIG_W / 11.25
FIG_W_MAX = 13.0          # taille maximale du dessin (pouces)
PANEL_PX = 300            # place réservée au panneau de commande (pixels)

# Panneau de commande compact
FONT_B = ("TkDefaultFont", 8, "bold")
FONT_S = ("TkDefaultFont", 8)
SCALE_LEN = 170

# ----------------------------------------------------------------------------
# Système en "per-unit" (bases choisies pour un onduleur de 10 kVA)
# ----------------------------------------------------------------------------
SN = 10e3                         # puissance de base (VA)
VB = 325.0                        # tension de base (crête phase-neutre)
IB = 2 * SN / (3 * VB)            # courant de base (crête)
ZB = VB / IB                      # impédance de base
F_BASE = 50.0                     # fréquence nominale du réseau (Hz)
W_BASE = 2 * math.pi * F_BASE

XF_PU = 0.15                      # inductance du filtre côté onduleur (pu)
LF = XF_PU * ZB / W_BASE
RF = 0.005 * ZB                   # résistance série de Lf
RD_FACTOR = 0.5                   # Rd = RD_FACTOR * sqrt(Lf/C) (amortissement)

# Réglages par défaut de la commande
DEF_ALPHA = 1500.0                # bande passante de la boucle de courant (rad/s)
DEF_WN = 100.0                    # pulsation propre de la PLL (rad/s)

TWO_PI_3 = 2 * math.pi / 3
SQRT3 = math.sqrt(3.0)
HALF_SQRT3 = SQRT3 / 2



def window_layout(win):
    """Décimation et nombre de points pour afficher `win` secondes."""
    dec = max(1, math.ceil(win / (N_MAX * DT)))
    n = max(50, min(N_MAX, int(round(win / (dec * DT)))))
    return dec, n

# 0 vg_a, 1 P, 2 ref_a, 3 tri, 4 vdc, 5 vpole_a, 6 vpcc_a, 7 Q,
# 8 i_d, 9 i_q, 10 ig_a(pu), 11 P*, 12 Q*, 13 i_d*, 14 i_q*, 15 sinus PLL
N_SIG = 16

# Couleurs
C_GRID = "#1f77b4"
C_PLL = "#ff7f0e"
C_MOD = "#d62728"
C_DC = "#9467bd"
C_PWM = "#8c564b"
C_OUT = "#2ca02c"
C_ZG = "#17becf"
C_WIRE = "#444444"

# Dimensions du dessin (unités arbitraires, 150 x 100)
W, H = 150.0, 100.0


def phase_voltage(theta, amp, harm):
    """Tension d'une phase (fondamental + éventuellement harmoniques 5 et 7)."""
    v = math.sin(theta)
    if harm:
        v += 0.06 * math.sin(5 * theta) + 0.04 * math.sin(7 * theta)
    return amp * v


class GFLModel:
    """Modèle numérique complet (un pas = DT) : réseau, PLL, courant dq, MLI, filtre, Zg."""

    def __init__(self):
        # --- réseau ---
        self.th_g = 0.0          # angle du réseau (sans le saut de phase)
        self.th_grid = 0.0       # angle total du réseau
        # --- PLL ---
        self.th_p = 0.0
        self.wi = 0.0
        self.w = W_BASE
        self.err = 0.0           # sin(erreur de phase), échantillon tenu
        self.a_est = 0.0         # amplitude de v_pcc (lissée)
        self.e_s = 0.0
        self.e_prev = 0.0
        self.vd_prev = 0.0
        self.vq_prev = 0.0
        self.enabled = True      # injection autorisée (PLL accrochée)
        self.lock_cnt = 0
        # --- régulation de courant ---
        self.ts_cnt = 0
        self.xi_d = self.xi_q = 0.0
        self.vd_ref = self.vq_ref = 0.0
        self.v_hold = 0.0
        self.id_s = self.iq_s = 0.0      # courants mesurés (échantillonnés)
        self.id_r = self.iq_r = 0.0      # consignes
        self.ilimited = False
        self.sat = False
        # --- MLI ---
        self.carrier_ph = 0.0
        self.m_raw = 0.0
        self.m = 0.0
        # --- mesures de puissance ---
        self.p_f = 0.0
        self.q_f = 0.0
        # --- électrique (repère αβ) ---
        self.iL_a = self.iL_b = 0.0
        self.ig_a = self.ig_b = 0.0
        self.vc_a = self.vc_b = 0.0
        self.i_mag = 0.0
        self.vp_mag = 0.0
        self.synced = False

        # --- paramètres (modifiés par l'interface) ---
        self.cf = 1.0
        self.rd = 1.0
        self.set_filter(DEF_FC)
        self.lg = 0.15 * ZB / W_BASE
        self.rg = 0.03 * ZB
        self.p_ref = 0.5
        self.q_ref = 0.0
        self.i_lim = 1.2
        self.alpha = DEF_ALPHA
        self.kp_pll = 2 * 0.7 * DEF_WN
        self.ki_pll = DEF_WN ** 2
        self.w_nom = W_BASE

    # ------------------------------------------------------------------
    def set_filter(self, fc):
        """Condensateur du filtre pour une coupure fc (Lf fixée)."""
        self.cf = 1.0 / ((2 * math.pi * fc) ** 2 * LF)
        self.rd = RD_FACTOR * math.sqrt(LF / self.cf)

    def set_params(self, fc, rg_pu, xg_pu, p_ref, q_ref, i_lim, alpha, wn, zeta, f0):
        self.set_filter(fc)
        self.rg = rg_pu * ZB
        self.lg = xg_pu * ZB / W_BASE
        self.p_ref = p_ref
        self.q_ref = q_ref
        self.i_lim = i_lim
        self.alpha = alpha
        self.kp_pll = 2.0 * zeta * wn
        self.ki_pll = wn * wn
        self.w_nom = 2 * math.pi * f0

    def resync(self):
        """Resynchronisation complète : PLL calée sur le réseau, courants nuls."""
        self.synced = False

    def reset_pll(self):
        """Réinitialise la PLL (angle 0, fréquence nominale) : l'injection est
        coupée jusqu'à ce que la PLL s'accroche de nouveau."""
        self.th_p = 0.0
        self.wi = 0.0
        self.w = self.w_nom
        self.a_est = 0.0
        self.enabled = False
        self.lock_cnt = 0
        self.xi_d = self.xi_q = 0.0

    # ------------------------------------------------------------------
    def _control(self, vd, vq, sp, cp, vdc):
        """Régulation de courant, exécutée aux sommets de la porteuse."""
        ts = self.ts_cnt * DT
        self.ts_cnt = 0
        w = self.w
        id_ = self.iL_a * sp - self.iL_b * cp
        iq_ = self.iL_a * cp + self.iL_b * sp

        # --- consignes de courant à partir de P*, Q* ---
        self.ilimited = False
        if self.enabled:
            vdm = max(vd, 50.0)
            id_r = self.p_ref * SN / (1.5 * vdm)
            iq_r = -self.q_ref * SN / (1.5 * vdm) + w * self.cf * vd   # + courant du condensateur
            lim = self.i_lim * IB
            mag = math.hypot(id_r, iq_r)
            if mag > lim:
                k = lim / mag
                id_r *= k
                iq_r *= k
                self.ilimited = True
        else:
            id_r = iq_r = 0.0
            self.xi_d = self.xi_q = 0.0

        # --- PI + découplage + prédiction de la tension ---
        kp = self.alpha * LF
        ki = kp * self.alpha / 4.0
        ed = id_r - id_
        eq = iq_r - iq_
        vdr = vd + kp * ed + self.xi_d - w * LF * iq_
        vqr = vq + kp * eq + self.xi_q + w * LF * id_
        mag_v = math.hypot(vdr, vqr)
        self.sat = mag_v > vdc / 2.0
        if not self.sat:                       # anti-windup
            self.xi_d += ki * ed * ts
            self.xi_q += ki * eq * ts
        self.vd_ref = vdr
        self.vq_ref = vqr
        self.v_hold = mag_v
        self.id_s = id_
        self.iq_s = iq_
        self.id_r = id_r
        self.iq_r = iq_r

    # ------------------------------------------------------------------
    def step(self, vac, phi_deg, freq, harm, vdc, fcar):
        # ---------------- Réseau infini (αβ) ----------------
        self.th_g = (self.th_g + 2 * math.pi * freq * DT) % (2 * math.pi)
        th = self.th_g + math.radians(phi_deg)
        self.th_grid = th
        if harm:
            va = phase_voltage(th, vac, True)
            vb = phase_voltage(th - TWO_PI_3, vac, True)
            vc = phase_voltage(th + TWO_PI_3, vac, True)
            vg_a = (2.0 / 3.0) * (va - 0.5 * vb - 0.5 * vc)
            vg_b = (vb - vc) / SQRT3
        else:
            vg_a = vac * math.sin(th)
            vg_b = -vac * math.cos(th)

        # ---------------- Synchronisation initiale ----------------
        if not self.synced:
            self.th_p = th
            self.w = 2 * math.pi * freq
            self.wi = self.w - self.w_nom
            self.a_est = math.hypot(vg_a, vg_b)
            self.iL_a = self.iL_b = self.ig_a = self.ig_b = 0.0
            self.vc_a, self.vc_b = vg_a, vg_b
            self.xi_d = self.xi_q = 0.0
            self.vd_ref = self.a_est
            self.vq_ref = 0.0
            self.v_hold = self.a_est
            self.id_s = self.iq_s = self.id_r = self.iq_r = 0.0
            self.p_f = self.q_f = 0.0
            self.enabled = True
            self.lock_cnt = 0
            self.ts_cnt = 0
            self.e_s = self.e_prev = 0.0
            self.vd_prev = self.a_est
            self.vq_prev = 0.0
            self.synced = True

        # ---------------- Point de raccordement ----------------
        ic_a = self.iL_a - self.ig_a
        ic_b = self.iL_b - self.ig_b
        vp_a = self.vc_a + self.rd * ic_a
        vp_b = self.vc_b + self.rd * ic_b

        # ---------------- Mesure de P et Q (au PCC, côté réseau) ----------------
        p_inst = 1.5 * (vp_a * self.ig_a + vp_b * self.ig_b) / SN
        q_inst = 1.5 * (vp_b * self.ig_a - vp_a * self.ig_b) / SN
        k = DT / 0.002
        self.p_f += (p_inst - self.p_f) * k
        self.q_f += (q_inst - self.q_f) * k

        # ---------------- PLL (SRF, un seul PI) ----------------
        sp = math.sin(self.th_p)
        cp = math.cos(self.th_p)
        vd = vp_a * sp - vp_b * cp
        vq = vp_a * cp + vp_b * sp
        amp = math.hypot(vp_a, vp_b)
        self.a_est += (amp - self.a_est) * DT / 0.005
        e = self.e_s                       # détecteur de phase échantillonné (tenu)
        self.err = e
        self.wi += self.ki_pll * e * DT
        lim = 2 * math.pi * 10
        self.wi = max(-lim, min(lim, self.wi))
        self.w = self.w_nom + self.kp_pll * e + self.wi
        if not self.enabled:               # attente de l'accrochage de la PLL
            if abs(e) < 0.05:
                self.lock_cnt += 1
            else:
                self.lock_cnt = 0
            if self.lock_cnt * DT > 0.02:
                self.enabled = True

        # ---------------- Porteuse + échantillonnage synchrone ----------------
        ph_old = self.carrier_ph
        self.carrier_ph = (ph_old + fcar * DT) % 1.0
        ph = self.carrier_ph
        self.ts_cnt += 1
        if (ph_old < 0.5 <= ph) or (ph < ph_old):     # sommet ou creux du triangle
            e_n = vq / max(amp, 5.0)
            self.e_s = 0.5 * (e_n + self.e_prev)      # moyenne de 2 échantillons :
            self.e_prev = e_n                         # annule l'ondulation alternée
            vd_a = 0.5 * (vd + self.vd_prev)
            vq_a = 0.5 * (vq + self.vq_prev)
            self.vd_prev = vd
            self.vq_prev = vq
            self._control(vd_a, vq_a, sp, cp, vdc)
        tri = 4.0 * abs(ph - 0.5) - 1.0

        # ---------------- Modulantes (dq -> abc), index m ≤ 1 ----------------
        hv = vdc / 2.0
        self.m_raw = self.v_hold / max(hv, 1.0)
        self.m = min(self.m_raw, 1.0)
        sc = 1.0 / self.m_raw if self.m_raw > 1.0 else 1.0
        va_ = self.vd_ref * sp + self.vq_ref * cp
        vb_ = -self.vd_ref * cp + self.vq_ref * sp
        kk = sc / max(hv, 1.0)
        ra = va_ * kk
        rb = (-0.5 * va_ + HALF_SQRT3 * vb_) * kk
        rc = (-0.5 * va_ - HALF_SQRT3 * vb_) * kk

        # ---------------- MLI (3 bras) ----------------
        pa = hv if ra > tri else -hv
        pb = hv if rb > tri else -hv
        pc = hv if rc > tri else -hv
        vi_a = (2.0 / 3.0) * (pa - 0.5 * pb - 0.5 * pc)
        vi_b = (pb - pc) / SQRT3

        self.th_p = (self.th_p + self.w * DT) % (2 * math.pi)

        # ---------------- Filtre LC + impédance réseau ----------------
        self.iL_a += DT * (vi_a - vp_a - RF * self.iL_a) / LF
        self.iL_b += DT * (vi_b - vp_b - RF * self.iL_b) / LF
        self.ig_a += DT * (vp_a - vg_a - self.rg * self.ig_a) / self.lg
        self.ig_b += DT * (vp_b - vg_b - self.rg * self.ig_b) / self.lg
        self.vc_a += DT * (self.iL_a - self.ig_a) / self.cf
        self.vc_b += DT * (self.iL_b - self.ig_b) / self.cf

        # ---------------- Grandeurs lissées pour l'affichage ----------------
        kk2 = DT / 0.01
        self.i_mag += (math.hypot(self.ig_a, self.ig_b) / IB - self.i_mag) * kk2
        self.vp_mag += (math.hypot(vp_a, vp_b) - self.vp_mag) * kk2

        return (vg_a, self.p_f, ra, tri, vdc, pa, vp_a, self.q_f,
                self.id_s / IB, self.iq_s / IB, self.ig_a / IB,
                self.p_ref, self.q_ref, self.id_r / IB, self.iq_r / IB,
                self.a_est * math.sin(self.th_p))


class App:
    def __init__(self, root):
        global FIG_W, FIG_H, FS
        self.root = root
        # dessin le plus grand possible compte tenu de l'écran
        avail_w = root.winfo_screenwidth() - PANEL_PX
        avail_h = root.winfo_screenheight() - 170
        FIG_W = max(8.0, min(FIG_W_MAX, min(avail_w, 1.5 * avail_h) / 100.0))
        FIG_H = FIG_W / 1.5
        FS = FIG_W / 11.25
        root.title("Onduleur grid-following (PLL + courant dq) connecté au réseau infini via Zg")
        self.model = GFLModel()
        self.dec, self.n_pts = window_layout(WINDOW)
        self.buf = np.zeros((N_SIG, self.n_pts))
        self.cnt = 0
        self.step_acc = 0.0
        self.prefill = True            # remplit la fenêtre dès le premier affichage
        self.f_hat = F_BASE            # fréquence estimée lissée (affichage)
        self.x = np.linspace(-self.n_pts * self.dec * DT, 0, self.n_pts)
        self.plot_axes = []
        self.paused = False

        # variables de contrôle
        self.v_vac = tk.DoubleVar(value=325.0)
        self.v_phi = tk.DoubleVar(value=0.0)
        self.v_jump = tk.DoubleVar(value=10.0)
        self.v_freq = tk.DoubleVar(value=50.0)
        self.v_harm = tk.BooleanVar(value=False)
        self.v_rg = tk.DoubleVar(value=0.03)
        self.v_xg = tk.DoubleVar(value=0.15)
        self.v_pref = tk.DoubleVar(value=0.5)
        self.v_qref = tk.DoubleVar(value=0.0)
        self.v_ilim = tk.DoubleVar(value=1.2)
        self.v_alpha = tk.DoubleVar(value=DEF_ALPHA)
        self.v_wn = tk.DoubleVar(value=DEF_WN)
        self.v_zeta = tk.DoubleVar(value=0.7)
        self.v_f0 = tk.DoubleVar(value=50.0)
        self.v_vdc = tk.DoubleVar(value=DEF_VDC)
        self.v_fcar = tk.DoubleVar(value=math.log10(DEF_FCAR))   # log10 de la fréquence
        self.v_fc = tk.DoubleVar(value=DEF_FC)
        self.v_win = tk.DoubleVar(value=math.log10(WINDOW * 1000.0))   # log10 de la durée en ms
        self.v_speed = tk.DoubleVar(value=math.log10(0.01))    # log10 de la vitesse

        self._build_figure()
        self._build_controls()
        self.root.after(FRAME_MS, self.update)

    # ==================================================================
    # Construction de la figure : schéma + graphiques
    # ==================================================================
    def _build_figure(self):
        left = ttk.Frame(self.root)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(FIG_W, FIG_H), dpi=100)
        bg = self.fig.add_axes([0, 0, 1, 1], zorder=0)
        bg.set_xlim(0, W)
        bg.set_ylim(0, H)
        bg.axis("off")
        bg.set_autoscale_on(False)
        self.bg = bg

        self._draw_schematic(bg)
        self._build_plots()

        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    def _draw_schematic(self, bg):
        # ---------- petits utilitaires de dessin ----------
        def line(xs, ys, color=C_WIRE, lw=1.6, **kw):
            bg.plot(xs, ys, color=color, lw=lw, solid_capstyle="round",
                    zorder=kw.pop("zorder", 1), **kw)

        def dot(x, y, color=C_WIRE):
            bg.plot([x], [y], "o", color=color, ms=3.8, zorder=3)

        def box(x0, y0, x1, y1, ec, fc="#fbfbfd", lw=1.8, ls="-"):
            p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                               boxstyle="round,pad=0,rounding_size=1.5",
                               fc=fc, ec=ec, lw=lw, ls=ls, zorder=0.5)
            bg.add_patch(p)
            return p

        def txt(x, y, s, fs=7, color="#222222", ha="center", va="center",
                weight="normal", **kw):
            return bg.text(x, y, s, fontsize=fs * FS, color=color, ha=ha, va=va,
                           fontweight=weight, zorder=6, **kw)

        def arrow(p0, p1, color=C_WIRE, lw=1.6):
            bg.annotate("", xy=p1, xytext=p0, zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                        shrinkA=0, shrinkB=0, mutation_scale=9))

        def coil(x0, y, n=4, w=2.5, h=3.2):
            for k in range(n):
                bg.add_patch(Arc((x0 + w / 2 + w * k, y), w, h, theta1=0,
                                 theta2=180, ec=C_WIRE, lw=1.7, zorder=2))

        txt(75, 97.3, "ONDULEUR GRID-FOLLOWING (PLL + courant dq) connecté à un réseau infini via Zg",
            fs=10, weight="bold")

        self.t_time = txt(1, 97.3, "", fs=7, ha="left", color="#555555")

        # ======================= BUS DC =======================
        box(1, 38, 18, 72, C_DC)
        txt(9.5, 69.5, "BUS DC", fs=8, color=C_DC, weight="bold")
        line([5.5, 13.5], [58, 58], lw=1.6)
        line([7.5, 11.5], [56.5, 56.5], lw=3.4)
        line([5.5, 13.5], [53, 53], lw=1.6)
        line([7.5, 11.5], [51.5, 51.5], lw=3.4)
        line([9.5, 9.5], [58, 65.5])
        line([9.5, 9.5], [56.5, 53])
        line([9.5, 9.5], [51.5, 44.5])
        txt(13.2, 61, "+", fs=9, weight="bold")
        txt(13.2, 47, "−", fs=11, weight="bold")
        self.t_vdc = txt(9.5, 41, "", fs=7.5, color=C_DC, weight="bold")

        # ======================= ONDULEUR =======================
        box(22, 38, 50, 72, C_PWM)
        txt(36, 69.5, "ONDULEUR – 3 bras", fs=8, color=C_PWM, weight="bold")
        line([9.5, 45], [65.5, 65.5])               # rail +
        line([9.5, 45], [44.5, 44.5])               # rail −
        self.sw_off = "#e6e6e6"
        self.sw_on = "#2ca02c"
        self.sw_top = self.sw_bot = None
        for x, name in ((29, "C"), (36, "B"), (43, "A")):
            active = (name == "A")
            line([x, x], [65.5, 61.7])
            line([x, x], [56.3, 53.7])
            line([x, x], [48.3, 44.5])
            top = Rectangle((x - 2, 56.3), 4, 5.4, fc=self.sw_off, ec=C_WIRE,
                            lw=1.4, zorder=2)
            bot = Rectangle((x - 2, 48.3), 4, 5.4, fc=self.sw_off, ec=C_WIRE,
                            lw=1.4, zorder=2)
            bg.add_patch(top)
            bg.add_patch(bot)
            dot(x, 55)
            if active:
                self.sw_top, self.sw_bot = top, bot
            else:
                line([x, x + 3.5], [55, 55], color="#999999")
                txt(x + 4.6, 55, name, fs=7, color="#888888")
        txt(47.2, 57.3, "A", fs=8, weight="bold")
        txt(36, 40.3, "bras A : vert = interrupteur fermé", fs=6, color="#555555")
        line([43, 57], [55, 55])                    # sortie A vers le filtre

        # ======================= FILTRE LC =======================
        box(54, 38, 78, 72, C_OUT)
        txt(66, 69.5, "FILTRE LC", fs=8, color=C_OUT, weight="bold")
        self.t_fc = txt(66, 65.5, "", fs=7, color=C_OUT)
        coil(57, 55)
        txt(62, 59.2, "L", fs=8, weight="bold")
        line([67, 85], [55, 55])                    # vers l'impédance
        dot(71.5, 55)
        dot(76, 55)
        # condensateur C en série avec Rd
        line([71.5, 71.5], [55, 51.2])
        line([69, 74], [51.2, 51.2], lw=2.2)
        line([69, 74], [49.8, 49.8], lw=2.2)
        line([71.5, 71.5], [49.8, 47.8])
        bg.add_patch(Rectangle((70.3, 43.4), 2.4, 4.4, fc="white", ec=C_WIRE,
                               lw=1.5, zorder=2))
        line([71.5, 71.5], [43.4, 42.6])
        line([69.6, 73.4], [42.6, 42.6], lw=1.6)
        line([70.3, 72.7], [41.8, 41.8], lw=1.6)
        line([70.9, 72.1], [41.0, 41.0], lw=1.6)
        txt(67.2, 50.5, "C", fs=8, weight="bold")
        txt(68.4, 45.6, "Rd", fs=6.5, color="#555555")
        txt(60.5, 41.4, "neutre = milieu\ndu bus DC", fs=5.8, color="#555555")
        txt(73.4, 58.0, "v_pcc", fs=6.5, color="#555555")

        # ======================= IMPÉDANCE RÉSEAU =======================
        box(82, 38, 110, 72, C_ZG)
        txt(96, 69.5, "IMPÉDANCE RÉSEAU", fs=8, color=C_ZG, weight="bold")
        arrow((78.6, 55), (81.8, 55))
        txt(80.2, 57.4, "i_g", fs=6, color="#555555")
        bg.add_patch(Rectangle((85, 53.7), 6, 2.6, fc="white", ec=C_WIRE,
                               lw=1.5, zorder=2))
        line([91, 94], [55, 55])
        coil(94, 55)
        line([104, 128], [55, 55])
        txt(88, 59.0, "Rg", fs=8, weight="bold")
        txt(99, 59.0, "Lg", fs=8, weight="bold")
        self.t_zg = txt(96, 45, "", fs=6.8, color="#0b7f8c")

        # ======================= RÉSEAU INFINI =======================
        box(114, 38, 148, 72, C_GRID)
        txt(131, 69.5, "RÉSEAU INFINI", fs=8, color=C_GRID, weight="bold")
        bg.add_patch(Circle((136, 55), 8, fc="white", ec=C_WIRE, lw=1.8, zorder=2))
        t = np.linspace(-5, 5, 60)
        line(136 + t, 55 + 3 * np.sin(t * math.pi / 5), color=C_GRID, lw=2,
             zorder=3)
        txt(121, 57.6, "A B C", fs=5.5, color="#555555")
        self.t_grid = txt(131, 43.2, "", fs=7, color=C_GRID)

        # ======================= MODULATEUR MLI =======================
        self.mod_box = box(22, 1, 50, 33, C_MOD)
        txt(36, 31.2, "MODULATEUR MLI", fs=8, color=C_MOD, weight="bold")
        self.t_m = txt(36, 27.6, "", fs=7, weight="bold")
        self.t_fcar = txt(36, 2.9, "", fs=6.5, color=C_MOD)
        arrow((36, 33), (36, 38))
        txt(43.8, 35.5, "ordres de gâchette", fs=6, color="#555555")
        line([9.5, 9.5], [38, 18])
        arrow((9.5, 18), (21.6, 18))
        txt(15.8, 20, "Vdc", fs=6.5, color=C_DC)

        # ======================= CONTRÔLE GRID-FOLLOWING =======================
        box(58, 1, 148, 35, C_PLL)
        txt(59.5, 33.6, "GRID-FOLLOWING", fs=8, color=C_PLL, weight="bold", ha="left")
        self.led = Circle((131, 33.6), 1.0, fc="#999999", ec=C_WIRE, lw=1, zorder=3)
        bg.add_patch(self.led)
        self.t_lock = txt(133, 33.6, "", fs=6.5, ha="left")

        def sub(x0, x1, label, fs=6.3):
            box(x0, 26.5, x1, 32, C_PLL, fc="#fff7ee", lw=1.3)
            txt((x0 + x1) / 2, 29.25, label, fs=fs)

        sub(63, 77, "PLL (SRF)\nv_pcc → θ̂")
        sub(82, 96, "(P*, Q*) →\ni_d*, i_q*")
        sub(101, 115, "régulation PI\ncourant dq")
        sub(120, 134, "dq → abc")
        arrow((77, 29.25), (82, 29.25))
        dot(79.5, 29.25)
        txt(79.5, 27.2, "θ̂", fs=7, color=C_PLL, weight="bold")
        arrow((96, 29.25), (101, 29.25))
        arrow((115, 29.25), (120, 29.25))
        # θ̂ vers la transformation dq -> abc
        line([79.5, 79.5, 127], [29.25, 33.4, 33.4])
        arrow((127, 33.4), (127, 32.1))
        # sortie vers le modulateur
        line([127, 127], [26.5, 25])
        line([127, 58], [25, 25])
        arrow((58, 25), (50.4, 25), color=C_PLL, lw=2)
        txt(54.2, 27.3, "v_ref", fs=7, color=C_PLL, weight="bold")
        # mesure (v_pcc, i_L) -> PLL
        line([76, 76], [55, 36.8])
        arrow((76, 36.8), (76, 32.1))
        txt(77.6, 36.4, "mesure v_pcc, i_L", fs=6, ha="left", color="#555555")

        # lecture PLL (au-dessus du cadran) et légende du cadran
        self.t_pll = txt(135, 22.6, "", fs=6.5)
        txt(130, 2.6, "réseau", fs=6.3, color=C_GRID)
        txt(141, 2.6, "PLL", fs=6.3, color=C_PLL, weight="bold")

    # ------------------------------------------------------------------
    def _mkax(self, x0, y0, x1, y1, title, color, ylim):
        ax = self.fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H],
                               zorder=2)
        self.plot_axes.append(ax)
        ax.set_title(title, fontsize=7 * FS, color=color, pad=2, loc="left")
        ax.set_xlim(-self.n_pts * self.dec * DT, 0)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=6 * FS, length=2, pad=1, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_color(color)
            sp.set_linewidth(1.2)
        return ax

    def _build_plots(self):
        b = self.buf
        x = self.x

        # --- Bus DC ---
        ax = self._mkax(5, 76, 18, 92, "Tension DC (V)", C_DC, (0, 1100))
        self.l_vdc, = ax.plot(x, b[4], color=C_DC, lw=2)

        # --- Sortie MLI ---
        ax = self._mkax(26, 76, 50, 92, "Sortie MLI, bras A (±Vdc/2)", C_PWM,
                        (-560, 560))
        self.l_pwm, = ax.plot(x, b[5], color=C_PWM, lw=0.7)

        # --- Tension au PCC (sortie du filtre) + sinus reconstruit par la PLL ---
        ax = self._mkax(58, 76, 78, 92, "Sortie filtre (V)", C_OUT, (-560, 560))
        self.l_grid2, = ax.plot(x, b[0], color=C_GRID, lw=1, alpha=0.35)
        self.l_out, = ax.plot(x, b[6], color=C_OUT, lw=1.4)
        self.l_pllsin, = ax.plot(x, b[15], color=C_PLL, lw=1.1, ls="--")

        # --- Courant dans Zg ---
        ax = self._mkax(87, 76, 110, 92, "Courant i_g, phase A (pu)", C_ZG, (-2.5, 2.5))
        self.l_ig, = ax.plot(x, b[10], color=C_ZG, lw=1.4)

        # --- Réseau infini phase A ---
        ax = self._mkax(118, 76, 148, 92, "Réseau infini, phase A (V)", C_GRID,
                        (-450, 450))
        self.l_grid, = ax.plot(x, b[0], color=C_GRID, lw=1.6)

        # --- Modulante / porteuse ---
        ax = self._mkax(27, 5, 48, 21, "modulante / porteuse", C_MOD, (-1.25, 1.25))
        self.l_ref, = ax.plot(x, b[2], color=C_MOD, lw=1.5)
        self.l_car, = ax.plot(x, b[3], color="gray", lw=0.7)

        # --- P et Q (avec consignes en pointillés) ---
        ax = self._mkax(65, 5, 90, 19, "P (orange) / Q (bleu) en pu", C_PLL, (-2, 2))
        self.l_pref, = ax.plot(x, b[11], color=C_PLL, lw=0.9, ls=":")
        self.l_qref, = ax.plot(x, b[12], color=C_GRID, lw=0.9, ls=":")
        self.l_p, = ax.plot(x, b[1], color=C_PLL, lw=1.7)
        self.l_q, = ax.plot(x, b[7], color=C_GRID, lw=1.3)

        # --- Courants dq (consignes en pointillés) ---
        ax = self._mkax(99, 5, 119, 19, "i_d (rouge) / i_q (bleu), pu", C_MOD, (-1.5, 1.5))
        self.l_idr, = ax.plot(x, b[13], color=C_MOD, lw=0.9, ls=":")
        self.l_iqr, = ax.plot(x, b[14], color=C_GRID, lw=0.9, ls=":")
        self.l_id, = ax.plot(x, b[8], color=C_MOD, lw=1.6)
        self.l_iq, = ax.plot(x, b[9], color=C_GRID, lw=1.3)

        # --- Cadran : angle du réseau vs angle de la PLL ---
        ax = self.fig.add_axes([124 / W, 3 / H, 22 / W, 17 / H], zorder=2)
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Circle((0, 0), 1.0, fc="white", ec="#555555", lw=1.2))
        for a in range(0, 360, 30):
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            ax.plot([0.88 * ca, ca], [0.88 * sa, sa], color="#888888", lw=0.8)
        for a in (0, 90, 180, 270):
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            ax.text(1.22 * ca, 1.22 * sa, f"{a}°", fontsize=5.5 * FS, ha="center",
                    va="center", color="#666666")
        self.n_real, = ax.plot([0, 1], [0, 0], color=C_GRID, lw=1.4,
                               solid_capstyle="round")
        self.n_pll, = ax.plot([0, 0.92], [0, 0], color=C_PLL, lw=3,
                              solid_capstyle="round", alpha=0.9)
        ax.plot([0], [0], "o", color="#333333", ms=3)

        # courbes dont l'axe des temps change avec la durée affichée
        # (courbe, indice du signal) : sert à la mise à jour et au changement de fenêtre
        self.line_idx = [(self.l_vdc, 4),
                         (self.l_pwm, 5),
                         (self.l_grid2, 0),
                         (self.l_out, 6),
                         (self.l_pllsin, 15),
                         (self.l_ig, 10),
                         (self.l_grid, 0),
                         (self.l_ref, 2),
                         (self.l_car, 3),
                         (self.l_pref, 11),
                         (self.l_qref, 12),
                         (self.l_p, 1),
                         (self.l_q, 7),
                         (self.l_idr, 13),
                         (self.l_iqr, 14),
                         (self.l_id, 8),
                         (self.l_iq, 9)]

    # ==================================================================
    # Panneau de commande (compact)
    # ==================================================================
    def _slider(self, parent, label, var, lo, hi, res, unit):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=1)
        head = ttk.Frame(f)
        head.pack(fill=tk.X)
        ttk.Label(head, text=label, font=FONT_B).pack(side=tk.LEFT)
        val = ttk.Label(head, text="", foreground="#0a58ca", font=FONT_B)
        val.pack(side=tk.RIGHT)
        txt = str(res)
        nd = len(txt.split(".")[1]) if "." in txt else 0

        def refresh(*_):
            try:
                v = var.get()
            except tk.TclError:
                return
            val.config(text=f"{v:.{nd}f} {unit}".strip())

        var.trace_add("write", refresh)
        refresh()
        tk.Scale(f, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                 variable=var, showvalue=False, length=SCALE_LEN, width=10,
                 sliderlength=16, bd=0, highlightthickness=0).pack(fill=tk.X)

    def _slider_log(self, parent, label, var, lo, hi, unit):
        """Curseur logarithmique : `var` contient log10(valeur)."""
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=1)
        head = ttk.Frame(f)
        head.pack(fill=tk.X)
        ttk.Label(head, text=label, font=FONT_B).pack(side=tk.LEFT)
        val = ttk.Label(head, text="", foreground="#0a58ca", font=FONT_B)
        val.pack(side=tk.RIGHT)

        def refresh(*_):
            try:
                v = 10 ** var.get()
            except tk.TclError:
                return
            t = f"{v:.0f}" if v >= 1000 else f"{v:.3g}"
            val.config(text=f"{t} {unit}")

        var.trace_add("write", refresh)
        refresh()
        tk.Scale(f, from_=math.log10(lo), to=math.log10(hi), resolution=0.01,
                 orient=tk.HORIZONTAL, variable=var, showvalue=False,
                 length=SCALE_LEN, width=10, sliderlength=16, bd=0,
                 highlightthickness=0).pack(fill=tk.X)

    def _note(self, parent, text):
        ttk.Label(parent, text=text, foreground="gray", justify="left",
                  wraplength=SCALE_LEN + 20, font=FONT_S).pack(anchor="w", pady=3)

    def _build_controls(self):
        ttk.Style().configure("TNotebook.Tab", padding=(5, 2), font=FONT_S)
        p = ttk.LabelFrame(self.root, text="Commandes", padding=4)
        p.pack(side=tk.RIGHT, fill=tk.Y, padx=4, pady=4)

        nb = ttk.Notebook(p)
        nb.pack(fill=tk.X)
        t_grid, t_pts, t_reg, t_inv, t_disp = (ttk.Frame(nb, padding=4) for _ in range(5))
        nb.add(t_grid, text="Réseau")
        nb.add(t_pts, text="Points")
        nb.add(t_reg, text="PLL/I")
        nb.add(t_inv, text="Onduleur")
        nb.add(t_disp, text="Affich.")

        # ---------------- Onglet Réseau (source + impédance Zg) ----------------
        self._slider(t_grid, "Amplitude crête", self.v_vac, 0, 400, 1, "V")
        self._slider(t_grid, "Phase", self.v_phi, -180, 180, 1, "°")
        fj = ttk.Frame(t_grid)
        fj.pack(fill=tk.X, pady=2)
        ttk.Label(fj, text="Saut :", font=FONT_B).pack(side=tk.LEFT)
        ttk.Spinbox(fj, from_=-180, to=180, increment=5, width=5,
                    textvariable=self.v_jump).pack(side=tk.LEFT, padx=3)
        ttk.Label(fj, text="°", font=FONT_B).pack(side=tk.LEFT)
        ttk.Button(fj, text="⚡ Appliquer", command=self.phase_jump).pack(side=tk.RIGHT)
        self._slider(t_grid, "Fréquence", self.v_freq, 40, 60, 0.1, "Hz")
        ttk.Checkbutton(t_grid, text="Harmoniques 5 et 7",
                        variable=self.v_harm).pack(anchor="w")
        ttk.Separator(t_grid).pack(fill=tk.X, pady=3)
        self._slider(t_grid, "Rg (impédance)", self.v_rg, 0.0, 0.5, 0.005, "pu")
        self._slider(t_grid, "Xg (impédance)", self.v_xg, 0.02, 1.0, 0.01, "pu")
        self.lbl_zg = ttk.Label(t_grid, text="", justify="left", font=("Courier", 8))
        self.lbl_zg.pack(anchor="w")

        # ---------------- Onglet Points de fonctionnement ----------------
        self._slider(t_pts, "P* (puissance active)", self.v_pref, -1.2, 1.2, 0.05, "pu")
        self._slider(t_pts, "Q* (puissance réactive)", self.v_qref, -0.6, 0.6, 0.05, "pu")
        self._slider(t_pts, "Limite de courant", self.v_ilim, 0.5, 2.0, 0.05, "pu")
        self.lbl_pts = ttk.Label(t_pts, text="", justify="left", font=("Courier", 8))
        self.lbl_pts.pack(anchor="w", pady=2)
        self._note(t_pts, "P* > 0 : l'onduleur injecte dans le réseau. "
                          "1 pu = 10 kW. Q* > 0 : il fournit du réactif "
                          "(comportement inductif). Si Vdc est trop faible, la MLI "
                          "sature et le point demandé n'est plus atteint.")

        # ---------------- Onglet PLL / régulation de courant ----------------
        self._slider(t_reg, "Bande passante courant α", self.v_alpha, 50, 5000, 10, "rad/s")
        self._slider(t_reg, "PLL : pulsation propre ωn", self.v_wn, 10, 600, 1, "rad/s")
        self._slider(t_reg, "PLL : amortissement ζ", self.v_zeta, 0.2, 2.0, 0.05, "")
        self._slider(t_reg, "PLL : fréquence nominale", self.v_f0, 40, 60, 0.1, "Hz")
        self.lbl_reg = ttk.Label(t_reg, text="", justify="left", font=("Courier", 8))
        self.lbl_reg.pack(anchor="w", pady=2)
        self._note(t_reg, "α et ωn doivent rester nettement sous la fréquence de MLI "
                          "(en rad/s), sinon la boucle devient instable. En réseau "
                          "faible (Xg grand), une PLL rapide s'instabilise aussi. "
                          "Astuce : change la fréquence nominale puis clique sur "
                          "« Réinitialiser la PLL » pour voir l'accrochage.")

        # ---------------- Onglet Onduleur ----------------
        self._slider(t_inv, "Tension DC Vdc", self.v_vdc, 20, 1000, 1, "V")
        self._slider_log(t_inv, "Fréquence MLI", self.v_fcar, 500, 20000, "Hz")
        self._slider(t_inv, "Coupure filtre LC", self.v_fc, 200, 3000, 10, "Hz")
        self._note(t_inv, "Index de modulation m = |v_ref| / (Vdc/2), limité à 1.")

        # ---------------- Onglet Affichage ----------------
        self._slider_log(t_disp, "Durée affichée", self.v_win, 0.5, 1000, "ms")
        self._slider_log(t_disp, "Vitesse simulation", self.v_speed, 0.001, 1, "× temps réel")
        self._note(t_disp, "Ralentis la simulation (jusqu'à ×0,001) pour suivre la commutation ; réduis la durée affichée (1 à 2 ms) pour voir les impulsions MLI, ou augmente-la (200 à 1000 ms) pour voir les transitoires de régulation. Si le PC ne suit pas, la simulation ralentit d'elle-même.")

        ttk.Separator(p).pack(fill=tk.X, pady=3)
        self.lbl_status = ttk.Label(p, text="", justify="left", font=("Courier", 8))
        self.lbl_status.pack(anchor="w")

        self.btn_pause = ttk.Button(p, text="⏸ Pause", command=self.toggle_pause)
        self.btn_pause.pack(fill=tk.X, pady=1)
        ttk.Button(p, text="↺ Réinitialiser la PLL",
                   command=self.model.reset_pll).pack(fill=tk.X, pady=1)
        ttk.Button(p, text="⟲ Resynchroniser tout",
                   command=self.model.resync).pack(fill=tk.X, pady=1)

    # ------------------------------------------------------------------
    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.config(text="▶ Reprendre" if self.paused else "⏸ Pause")

    def phase_jump(self):
        """Ajoute instantanément le saut saisi à la phase du réseau."""
        try:
            jump = self.v_jump.get()
        except tk.TclError:          # saisie invalide (texte, case vide...)
            return
        new = self.v_phi.get() + jump
        new = (new + 180.0) % 360.0 - 180.0     # ramené dans [-180°, +180°]
        self.v_phi.set(new)

    def _apply_window(self, win):
        """Adapte la durée affichée : décimation, nombre de points, axe des temps."""
        dec, n = window_layout(win)
        if dec == self.dec and n == self.n_pts:
            return
        self.dec, self.n_pts, self.cnt = dec, n, 0
        span = n * dec * DT
        self.x = np.linspace(-span, 0, n)
        for ax in self.plot_axes:
            ax.set_xlim(-span, 0)
        # historique remis à l'échelle : rempli avec la dernière valeur
        self.buf = np.repeat(self.buf[:, -1:], n, axis=1)
        for ln, i in self.line_idx:
            ln.set_data(self.x, self.buf[i])

    # ==================================================================
    # Boucle d'animation
    # ==================================================================
    def update(self):
        if not self.paused:
            try:
                vac = self.v_vac.get()
                phi = self.v_phi.get()
                freq = self.v_freq.get()
                harm = self.v_harm.get()
                rg = self.v_rg.get()
                xg = self.v_xg.get()
                pref = self.v_pref.get()
                qref = self.v_qref.get()
                ilim = self.v_ilim.get()
                alpha = self.v_alpha.get()
                wn = self.v_wn.get()
                zeta = self.v_zeta.get()
                f0 = self.v_f0.get()
                vdc = self.v_vdc.get()
                fcar = 10 ** self.v_fcar.get()
                fc = self.v_fc.get()
                win = 10 ** self.v_win.get() / 1000.0      # secondes
                speed = 10 ** self.v_speed.get()           # fraction du temps réel
            except tk.TclError:          # valeur en cours de saisie : on saute l'image
                self.root.after(FRAME_MS, self.update)
                return

            m = self.model
            m.set_params(fc, rg, xg, pref, qref, ilim, alpha, wn, zeta, f0)

            self._apply_window(win)
            dec = self.dec

            # ---- simulation (avec un budget de temps de calcul par image) ----
            self.step_acc += speed * FRAME_MS * 1e-3 / DT
            n_steps = int(self.step_acc)
            self.step_acc -= n_steps
            budget = FRAME_BUDGET
            if self.prefill:
                n_steps = self.n_pts * self.dec
                budget = 5.0
                self.prefill = False
            samples = []
            step = m.step
            cnt = self.cnt
            t0 = time.perf_counter()
            for i in range(n_steps):
                out = step(vac, phi, freq, harm, vdc, fcar)
                cnt += 1
                if cnt >= dec:
                    cnt = 0
                    samples.append(out)
                if (i & 255) == 255 and time.perf_counter() - t0 > budget:
                    self.step_acc = 0.0          # le PC ne suit pas : on décroche du temps réel
                    break
            self.cnt = cnt
            if samples:
                chunk = np.array(samples).T
                nk = min(chunk.shape[1], self.n_pts)
                self.buf = np.roll(self.buf, -nk, axis=1)
                self.buf[:, -nk:] = chunk[:, -nk:]
            b = self.buf

            # ---- courbes ----
            for ln, i in self.line_idx:
                ln.set_ydata(b[i])

            # ---- interrupteurs du bras A ----
            top_on = b[5][-1] > 0
            self.sw_top.set_facecolor(self.sw_on if top_on else self.sw_off)
            self.sw_bot.set_facecolor(self.sw_off if top_on else self.sw_on)

            # ---- cadran : angle réseau vs angle de la PLL ----
            tg = m.th_grid
            tp = m.th_p
            self.n_real.set_data([0, math.cos(tg)], [0, math.sin(tg)])
            self.n_pll.set_data([0, 0.92 * math.cos(tp)], [0, 0.92 * math.sin(tp)])

            # ---- lecture PLL, voyant de verrouillage ----
            self.f_hat += 0.15 * (m.w / (2 * math.pi) - self.f_hat)
            err_deg = math.degrees(math.asin(max(-1.0, min(1.0, m.err))))
            locked = m.enabled and abs(err_deg) < 3.0
            self.led.set_facecolor("#2ca02c" if locked else C_PLL)
            self.t_lock.set_text("PLL verrouillée" if locked else "accrochage…")
            self.t_pll.set_text(
                f"f̂ = {self.f_hat:.2f} Hz   erreur = {err_deg:+.1f}°\n"
                f"Â = {m.a_est:.0f} V   θ̂ = {math.degrees(tp):.0f}°")

            # ---- modulateur ----
            sat = m.m_raw > 1.0
            self.t_m.set_text(
                f"m = |v_ref|/(Vdc/2) = {m.m_raw:.2f}"
                + ("\n⚠ SATURÉ → m = 1" if sat else ""))
            self.t_m.set_color("red" if sat else "#222222")
            self.mod_box.set_edgecolor("red" if sat else C_MOD)
            self.mod_box.set_linewidth(3.2 if sat else 1.8)

            # ---- textes des blocs ----
            scr = 1.0 / max(xg, 1e-3)
            self.t_vdc.set_text(f"Vdc = {vdc:.0f} V")
            self.t_time.set_text(f"fenêtre : {win * 1000:.3g} ms\nvitesse : ×{speed:.3g}")
            self.t_fc.set_text(f"fc = {fc:.0f} Hz")
            self.t_fcar.set_text(
                f"porteuse : {fcar:.0f} Hz")
            self.t_zg.set_text(f"Rg = {rg:.3f} pu    Xg = {xg:.2f} pu\nSCR ≈ {scr:.1f}")
            self.t_grid.set_text(
                f"Vcrête = {vac:.0f} V\nφ = {phi:+.0f}°\nf = {freq:.2f} Hz")

            # ---- infos dans le panneau ----
            self.lbl_zg.config(text=f"Zg = {rg * ZB:.2f} + j{xg * ZB:.2f} Ω\nSCR ≈ 1/Xg = {scr:.1f}")
            self.lbl_pts.config(text=f"P* = {pref * SN / 1e3:+.1f} kW   Q* = {qref * SN / 1e3:+.1f} kvar\n"
                                     f"|i*| = {math.hypot(m.id_r, m.iq_r) / IB:.2f} pu")
            kp_i = alpha * LF
            too_fast = alpha > 0.35 * fcar or wn > 0.35 * fcar
            self.lbl_reg.config(
                text=(f"Courant : Kp = {kp_i:.2f} V/A  Ki = {kp_i * alpha / 4:.1f}\n"
                      f"PLL     : Kp = {m.kp_pll:.1f}  Ki = {m.ki_pll:.0f}\n"
                      f"réponse PLL ≈ {4000.0 / max(zeta * wn, 1e-3):.0f} ms"
                      + ("\n⚠ α ou ωn trop proche de fMLI" if too_fast else "")),
                foreground="#b45309" if too_fast else "black")

            i_over = m.i_mag > 1.2
            state = ("SATURATION MLI" if sat else "PLL EN ACCROCHAGE" if not m.enabled
                     else "SURINTENSITÉ" if i_over else "LIMITE DE COURANT" if m.ilimited
                     else "normal")
            self.lbl_status.config(
                text=(f"P = {m.p_f:+5.2f} pu ({m.p_f * SN / 1e3:+5.1f} kW)\n"
                      f"Q = {m.q_f:+5.2f} pu ({m.q_f * SN / 1e3:+5.1f} kvar)\n"
                      f"i_d = {m.id_s / IB:+5.2f}  i_q = {m.iq_s / IB:+5.2f} pu\n"
                      f"f̂ = {self.f_hat:.2f} Hz  err = {err_deg:+.1f}°\n"
                      f"|i_g| = {m.i_mag:4.2f} pu\n"
                      f"|v_pcc| = {m.vp_mag:4.0f} V\n"
                      f"État : {state}"),
                foreground="red" if state not in ("normal",) else "black")

            self.canvas.draw_idle()

        self.root.after(FRAME_MS, self.update)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
