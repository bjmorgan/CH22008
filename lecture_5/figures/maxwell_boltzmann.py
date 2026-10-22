"""Figure 5.2: where the exponential factor in collision theory comes from.

Distribution of head-on collision energy (the part of the relative kinetic energy along the
line between the molecules' centres) across all collisions, at two temperatures. This
distribution is exactly exponential, so the shaded tail above Ea is exactly exp(-Ea/RT).
See the appendix "Derivation of the Exponential Energy Factor".

Writes maxwell_boltzmann.png (lecture notes) and maxwell_boltzmann.pdf (slides) next to
this script. Run with: python3 maxwell_boltzmann.py
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Helvetica",
    "mathtext.fontset": "stix",
    "font.size": 16,
    "pdf.fonttype": 42,
})

R = 8.314e-3            # kJ K^-1 mol^-1
EA = 7.5                # kJ mol^-1; illustrative, small enough for the tails to be visible
TEMPS = (300, 600)      # K
COLOURS = {300: "#3b75b4", 600: "#ee8833"}
SHADES = {300: "#b0b0b0", 600: "#f8d0b0"}
E = np.linspace(0, 25, 2000)
OUT = Path(__file__).resolve().parent


def head_on(E, T):
    """Head-on collision energy across all collisions: p(E) = exp(-E/RT) / RT."""
    return np.exp(-E / (R * T)) / (R * T)


fig, ax = plt.subplots(figsize=(6, 4.4))
tail = E >= EA
for T in (600, 300):    # draw the larger (600 K) tail first so the 300 K tail sits on top
    ax.fill_between(E[tail], head_on(E[tail], T), color=SHADES[T], lw=0)
for T in TEMPS:
    ax.plot(E, head_on(E, T), color=COLOURS[T], lw=2.0, label=f"T = {T} K")
top = head_on(0, 300)
ax.plot([EA, EA], [0, 0.55 * top], "k--", lw=1.5)
ax.text(EA, 0.57 * top, r"$E_\mathrm{a}$", ha="center", va="bottom", fontsize=20)
ax.text(0.97, 0.42, r"$P(E \geq E_\mathrm{a}) = \exp\left(\dfrac{-E_\mathrm{a}}{RT}\right)$",
        transform=ax.transAxes, ha="right", fontsize=21)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.set_yticks([])
ax.set_ylim(0, 1.05 * top)
ax.set_xlim(-0.5, 25)
ax.set_ylabel("probability density")
ax.set_xlabel(r"head-on collision energy / kJ mol$^{-1}$")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "maxwell_boltzmann.png", dpi=200)
fig.savefig(OUT / "maxwell_boltzmann.pdf")
