#!/usr/bin/env python3
"""Run every experiment in the orbital simulator project.

    python run_experiments.py              # everything (about 2 minutes)
    python run_experiments.py --quick      # shorter runs, for a fast check

The experiments, each of which writes a figure to figures/:

  1. Simulate the whole solar system for 340 years and check it conserves energy and
     angular momentum (fig1_solar_system.png).
  2. Compare three integrators on a two-body orbit: energy conservation over 1000 orbits
     (fig2_integrators.png) and the order of convergence against the exact Kepler
     solution (fig3_convergence.png).
  3. Kepler's second law: equal areas in equal times, for Mercury and for Halley's Comet
     (fig4_kepler_second_law.png, halley_second_law.gif).
  4. Kepler's first and third laws, measured from the simulation (fig5_kepler_third_law.png).
  5. Mercury's perihelion precession: Newtonian gravity alone, then with the first
     post-Newtonian (general relativity) correction (fig6_mercury_precession.png).

All the numbers are also saved to results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

import orbital_simulator as sim  # noqa: E402

# --------------------------------------------------------------------------- plot style

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"

# One fixed colour per planet, used in every figure (never re-assigned by rank).
PLANET_COLOUR = dict(zip([p.name for p in sim.PLANETS], [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]))
# One fixed colour per integrator.
METHOD_COLOUR = {"euler": ORANGE, "symplectic_euler": VIOLET, "leapfrog": BLUE}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "font.size": 10,
        }
    )


# --------------------------------------------------------------------------- experiment 1 and 4: the solar system


def run_solar_system(years: float = 340.0, dt_days: float = 0.5, save_every: int = 2):
    """Leapfrog simulation of the Sun and all eight planets from J2000."""
    system = sim.solar_system()
    dt = dt_days * sim.DAY
    traj = sim.run(system, "leapfrog", dt, int(round(years / dt)), save_every=save_every)
    energy = sim.total_energy(traj.pos, traj.vel, traj.mass)
    ang = sim.angular_momentum(traj.pos, traj.vel, traj.mass)
    mom = sim.linear_momentum(traj.vel, traj.mass)
    conservation = {
        "years": years,
        "dt_days": dt_days,
        "steps": int(round(years / dt)),
        "energy_relative_range": float((energy.max() - energy.min()) / abs(energy[0])),
        "angular_momentum_relative_change": float(np.abs(ang - ang[0]).max() / np.linalg.norm(ang[0])),
        "momentum_max": float(np.abs(mom).max()),
    }
    return traj, conservation


def analyse_planets(traj) -> list:
    """Measure each planet's orbit in the simulation: period, mean semi-major axis, the shape of
    the orbit (Kepler's first law), and how constant the areal velocity is (second law)."""
    rows = []
    for planet in sim.PLANETS:
        if 2 * planet.reference_period > traj.t[-1]:
            continue  # fewer than two orbits in this run: no period to average
        i = traj.names.index(planet.name)
        r, v, mu = sim.jacobi_orbit(traj.pos, traj.vel, traj.mass, i)
        periods = sim.measure_periods(traj.t, r)
        a_osc, _, _ = sim.osculating_elements(r, v, mu)
        first = traj.t <= periods[0] + 1e-9
        fit = sim.fit_orbit_equation(r[first])
        dadt = sim.areal_velocity(r[first], v[first])
        a_mean = float(a_osc.mean())
        period = float(periods.mean())
        rows.append(
            {
                "name": planet.name,
                "n_orbits": int(len(periods)),
                "period_sim_yr": period,
                "period_reference_yr": planet.reference_period,
                "period_difference_percent": 100.0 * (period / planet.reference_period - 1.0),
                "a_sim_au": a_mean,
                "a_jpl_au": planet.a,
                "kepler3_ratio": float(period**2 / a_mean**3 / (4 * np.pi**2 / mu)),
                "t2_over_a3": float(period**2 / a_mean**3),
                "e_fit": fit["e"],
                "e_jpl": planet.e,
                "first_law_max_residual": fit["max_relative_residual"],
                "areal_velocity_spread": float((dadt.max() - dadt.min()) / dadt.mean()),
            }
        )
    return rows


def plot_solar_system(traj, rows, out: Path) -> None:
    by_name = {r["name"]: r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.9))
    panels = (
        (axes[0], ["Mercury", "Venus", "Earth", "Mars"], 1.85, "Inner planets (one orbit each)"),
        (axes[1], ["Jupiter", "Saturn", "Uranus", "Neptune"], 34.0, "Outer planets (one orbit each)"),
    )
    for ax, names, lim, title in panels:
        for name in names:
            if name not in by_name:
                continue
            i = traj.names.index(name)
            path = traj.pos[traj.t <= by_name[name]["period_sim_yr"] + 1e-9, i, :2]
            colour = PLANET_COLOUR[name]
            ax.plot(path[:, 0], path[:, 1], color=colour, lw=1.6)
            start = traj.pos[0, i, :2]
            ax.scatter(*start, s=34, color=colour, edgecolor=SURFACE, linewidth=1.3, zorder=4)
            ax.annotate(name, start, xytext=(6, 5), textcoords="offset points", fontsize=9, color=INK2)
        ax.scatter(*traj.pos[0, 0, :2], s=55, color=INK, edgecolor=SURFACE, linewidth=1.3, zorder=5)
        ax.annotate("Sun", traj.pos[0, 0, :2], xytext=(6, -12), textcoords="offset points", fontsize=9, color=INK2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x (AU)")
        ax.set_ylabel("y (AU)")
        ax.set_title(title)
    fig.suptitle("The simulated solar system from J2000, seen from above the ecliptic (dots: positions at J2000)", x=0.06, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_third_law(rows, fit, out: Path) -> None:
    a = np.array([r["a_sim_au"] for r in rows])
    t = np.array([r["period_sim_yr"] for r in rows])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    aa = np.geomspace(0.3, 40, 50)
    left.plot(aa, 10.0 ** fit["intercept"] * aa ** fit["slope"], color=MUTED, lw=1.3, label=f"Fit: T ∝ a^{fit['slope']:.4f} ± {fit['slope_error']:.4f}")
    for r in rows:
        left.scatter(r["a_sim_au"], r["period_sim_yr"], s=46, color=PLANET_COLOUR[r["name"]], edgecolor=SURFACE, linewidth=1.3, zorder=3)
        left.annotate(r["name"], (r["a_sim_au"], r["period_sim_yr"]), xytext=(-6, 7), textcoords="offset points", ha="right", fontsize=9, color=INK2)
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("Semi-major axis a (AU), measured in the simulation")
    left.set_ylabel("Orbital period T (years), measured in the simulation")
    left.set_title("Kepler's third law from the simulation")
    left.legend(loc="lower right")

    right.axhline(1.0, color=AXIS, lw=1)
    right.axhspan(0.999, 1.001, color=BLUE, alpha=0.08, lw=0)
    for k, r in enumerate(rows):
        right.scatter(k, r["kepler3_ratio"], s=46, color=PLANET_COLOUR[r["name"]], edgecolor=SURFACE, linewidth=1.3, zorder=3)
    right.set_xticks(range(len(rows)))
    right.set_xticklabels([r["name"] for r in rows], rotation=35, ha="right")
    right.set_ylim(0.997, 1.003)
    right.grid(axis="x", visible=False)
    right.set_ylabel("(T² / a³) / (4π² / G(M + m))")
    right.set_title("Measured against theory (shaded: ±0.1%)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- experiment 2: integrators


def integrator_energy_experiment(n_orbits: int = 1000, steps_per_orbit: int = 100) -> dict:
    """Energy error of each integrator on Earth's orbit, at 100 steps per orbit."""
    earth = sim.PLANET_BY_NAME["Earth"]
    system = sim.two_body(earth.a, earth.e, earth.mass, name="Earth")
    period = sim.orbital_period(earth.a, earth.mass)
    out = {}
    for method in sim.INTEGRATORS:
        traj = sim.run(system, method, period / steps_per_orbit, n_orbits * steps_per_orbit, save_every=10)
        energy = sim.total_energy(traj.pos, traj.vel, traj.mass)
        out[method] = {
            "orbits": traj.t / period,
            "energy_error": np.abs(energy / energy[0] - 1.0),
            "relative_position": traj.pos[:, 1] - traj.pos[:, 0],
        }
    return out


def convergence_experiment(steps=(50, 100, 200, 400, 800, 1600, 3200), fraction: float = 0.6) -> dict:
    """Position error of each integrator against the exact Kepler solution, versus step size.

    Mercury's orbit is used (e = 0.206). Each run goes for 0.6 of an orbit, not a whole one: at
    exactly one period the symplectic Euler method happens to be more accurate than its order
    suggests, which would hide the fact that it is only first order.
    """
    mercury = sim.PLANET_BY_NAME["Mercury"]
    system = sim.two_body(mercury.a, mercury.e, mercury.mass, name="Mercury")
    period = sim.orbital_period(mercury.a, mercury.mass)
    mu = sim.G * (1.0 + mercury.mass)
    t_final = fraction * period
    exact = sim.kepler_positions(mercury.a, mercury.e, np.sqrt(mu / mercury.a**3) * t_final)[0]
    dts = np.array([period / n for n in steps])
    result = {"dt_days": dts / sim.DAY, "methods": {}}
    for method in sim.INTEGRATORS:
        errors = []
        for n in steps:
            traj = sim.run(system, method, period / n, int(round(fraction * n)))
            rel = traj.pos[-1, 1] - traj.pos[-1, 0]
            errors.append(np.linalg.norm(rel[:2] - exact) / mercury.a)
        errors = np.array(errors)
        use = (np.array(steps) >= 200) & (errors > 1e-11)  # asymptotic regime, above round-off
        slope = float(np.polyfit(np.log10(dts[use]), np.log10(errors[use]), 1)[0])
        result["methods"][method] = {"errors": errors, "measured_order": slope, "expected_order": sim.INTEGRATOR_ORDER[method]}
    return result


def plot_integrators(energy: dict, out: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.8, 4.9), gridspec_kw={"width_ratios": [1.35, 1]})
    for method, data in energy.items():
        # Symplectic methods wobble in energy once per orbit. Plotting every sample would draw a solid
        # band, so show the largest error inside each orbit: that is the number that matters.
        orbit_index = np.floor(data["orbits"] + 1e-9).astype(int)
        n_bins = int(orbit_index.max())
        worst = np.array([data["energy_error"][orbit_index == k].max() for k in range(n_bins)])
        a.plot(np.arange(n_bins) + 0.5, np.maximum(worst, 1e-16), color=METHOD_COLOUR[method], lw=1.6, label=sim.INTEGRATOR_LABEL[method])
    a.set_yscale("log")
    a.set_ylim(1e-5, 3)
    a.set_xlim(0, energy["leapfrog"]["orbits"][-1])
    a.set_xlabel("Orbits completed (100 steps per orbit)")
    a.set_ylabel("Largest relative energy error |ΔE / E| in each orbit")
    a.set_title(f"Energy conservation over {energy['leapfrog']['orbits'][-1]:.0f} orbits")
    a.legend(loc="lower right", ncol=2)

    orbits = energy["euler"]["orbits"]
    keep = orbits <= 20
    for method, colour_label in (("euler", "Euler"), ("leapfrog", "Leapfrog")):
        rel = energy[method]["relative_position"][keep]
        b.plot(rel[:, 0], rel[:, 1], color=METHOD_COLOUR[method], lw=1.1, label=colour_label)
    b.scatter([0], [0], s=50, color=INK, zorder=4)
    b.set_aspect("equal")
    b.set_xlabel("x (AU)")
    b.set_ylabel("y (AU)")
    b.set_title("The same orbit, first 20 orbits")
    b.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_convergence(conv: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    dts = conv["dt_days"]
    for method, data in conv["methods"].items():
        ax.plot(
            dts,
            data["errors"],
            color=METHOD_COLOUR[method],
            marker="o",
            ms=4.5,
            label=f"{sim.INTEGRATOR_LABEL[method]}: slope {data['measured_order']:.2f}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time step (days)")
    ax.set_ylabel("Position error / semi-major axis")
    ax.set_title("Convergence to the exact Kepler orbit (Mercury, 0.6 orbit)")
    ax.legend(loc="lower right", title="Slope of the line = order of the method", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- experiment 3: Kepler's second law


def second_law_experiment(n_wedges: int = 12) -> dict:
    """Equal areas in equal times, measured geometrically.

    Mercury: split one orbit into n equal time intervals (equal steps in the mean anomaly), place
    the planet using the exact solution of Kepler's equation, and measure the area of each wedge
    from positions alone (sum of triangle areas). Halley: integrate the two-body orbit and
    compute the speed and the areal velocity.
    """
    mercury = sim.PLANET_BY_NAME["Mercury"]
    edges = np.linspace(0.0, 2 * np.pi, n_wedges + 1)
    paths = [sim.kepler_positions(mercury.a, mercury.e, np.linspace(edges[k], edges[k + 1], 4001)) for k in range(n_wedges)]
    areas = np.array([sim.swept_area(p) for p in paths])
    ellipse_area = np.pi * mercury.a**2 * np.sqrt(1.0 - mercury.e**2)

    system = sim.two_body(sim.HALLEY_A, sim.HALLEY_E, sim.HALLEY_MASS, name="Halley")
    period = sim.orbital_period(sim.HALLEY_A, sim.HALLEY_MASS)
    t_eval = np.linspace(0.0, period, 3001)
    t, pos, vel = sim.integrate_adaptive(system, period, t_eval=t_eval)
    r, v = sim.relative_orbit(pos, vel, 1)
    speed = np.linalg.norm(v, axis=1) * sim.KMS_PER_AU_YR
    dadt = sim.areal_velocity(r, v)
    return {
        "mercury": {"paths": paths, "areas": areas, "ellipse_area": ellipse_area, "n_wedges": n_wedges,
                    "area_spread": float((areas.max() - areas.min()) / areas.mean()),
                    "sum_over_ellipse": float(areas.sum() / ellipse_area)},
        "halley": {"t": t, "speed_kms": speed, "areal_velocity": dadt, "period_yr": period,
                   "perihelion_au": float(np.linalg.norm(r, axis=1).min()),
                   "aphelion_au": float(np.linalg.norm(r, axis=1).max()),
                   "max_speed_kms": float(speed.max()), "min_speed_kms": float(speed.min()),
                   "areal_velocity_spread": float((dadt.max() - dadt.min()) / dadt.mean()),
                   "closure_error_au": float(np.linalg.norm(r[-1] - r[0]))},
    }


def plot_second_law(data: dict, out: Path) -> None:
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15.2, 4.7), gridspec_kw={"width_ratios": [1.05, 1, 1]})
    mer = data["mercury"]
    n = mer["n_wedges"]
    for k, path in enumerate(mer["paths"]):
        poly = np.vstack([[0.0, 0.0], path])
        a.fill(poly[:, 0], poly[:, 1], color=BLUE, alpha=0.16 if k % 2 == 0 else 0.42, lw=0)
        a.plot(path[:, 0], path[:, 1], color=BLUE, lw=1.2)
        a.plot([0, path[0, 0]], [0, path[0, 1]], color=SURFACE, lw=1.2)
    a.scatter([0], [0], s=55, color=INK, zorder=4)
    a.annotate("Sun", (0, 0), xytext=(6, -13), textcoords="offset points", fontsize=9, color=INK2)
    a.annotate("perihelion", (mer["paths"][0][0, 0], 0), xytext=(-8, 8), textcoords="offset points", fontsize=9, color=INK2, ha="right")
    a.set_aspect("equal")
    a.set_xlabel("x (AU)")
    a.set_ylabel("y (AU)")
    a.set_title(f"Mercury: {n} equal times, {n} equal areas")
    a.text(0.5, 0.03, f"each wedge {mer['areas'].mean():.5f} AU², spread {mer['area_spread']:.1e}", transform=a.transAxes, ha="center", fontsize=9, color=INK2)

    hal = data["halley"]
    b.plot(hal["t"], hal["speed_kms"], color=BLUE)
    b.set_yscale("log")
    b.set_xlabel("Years since perihelion")
    b.set_ylabel("Speed (km/s)")
    b.set_title("Halley's Comet: speed varies 60-fold")
    b.annotate(f"{hal['max_speed_kms']:.1f} km/s at {hal['perihelion_au']:.2f} AU", (hal["t"][0], hal["max_speed_kms"]), xytext=(10, -14), textcoords="offset points", fontsize=9, color=INK2)
    idx = int(np.argmin(hal["speed_kms"]))
    b.annotate(f"{hal['min_speed_kms']:.2f} km/s at {hal['aphelion_au']:.1f} AU", (hal["t"][idx], hal["min_speed_kms"]), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9, color=INK2)

    mean = hal["areal_velocity"].mean()
    c.plot(hal["t"], hal["areal_velocity"] / mean, color=BLUE)
    c.set_ylim(0.9, 1.1)
    c.set_xlabel("Years since perihelion")
    c.set_ylabel("Areal velocity dA/dt, divided by its mean")
    c.set_title("...but dA/dt does not change")
    c.text(0.5, 0.12, f"largest variation: {hal['areal_velocity_spread']:.0e} of its value", transform=c.transAxes, ha="center", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def animate_halley(out: Path, window_yr: float = 2.0, frames: int = 150) -> None:
    """Halley's Comet round one orbit; the shaded wedge is the area swept in the last 2 years."""
    system = sim.two_body(sim.HALLEY_A, sim.HALLEY_E, sim.HALLEY_MASS, name="Halley")
    period = sim.orbital_period(sim.HALLEY_A, sim.HALLEY_MASS)
    step = 0.01
    t_eval = np.arange(0.0, period + window_yr + step, step)
    t, pos, vel = sim.integrate_adaptive(system, t_eval[-1], t_eval=t_eval)
    r, v = sim.relative_orbit(pos, vel, 1)
    xy = r[:, :2]
    speed = np.linalg.norm(v, axis=1) * sim.KMS_PER_AU_YR
    per_window = int(round(window_yr / step))
    frame_index = np.linspace(per_window, len(t) - 1, frames).astype(int)

    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.plot(xy[:, 0], xy[:, 1], color=AXIS, lw=1)
    ax.scatter([0], [0], s=55, color=INK, zorder=4)
    ax.set_xlim(-38, 5)
    ax.set_ylim(-6.5, 6.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    wedge = ax.fill([0, 0], [0, 0], color=BLUE, alpha=0.4, lw=0)[0]
    comet = ax.scatter([], [], s=42, color=ORANGE, edgecolor=SURFACE, linewidth=1.2, zorder=5)
    text = ax.text(0.99, 0.94, "", transform=ax.transAxes, ha="right", va="top", fontsize=9.5)
    ax.set_title("Halley's Comet: the area swept in any 2 years is the same")

    def update(k):
        i = frame_index[k]
        seg = xy[i - per_window : i + 1]
        poly = np.vstack([[0.0, 0.0], seg])
        wedge.set_xy(poly)
        comet.set_offsets(xy[i : i + 1])
        area = sim.swept_area(seg)
        text.set_text(f"t = {t[i] - window_yr:5.1f} yr    speed {speed[i]:5.2f} km/s\narea swept in last {window_yr:.0f} yr = {area:.2f} AU²")

    anim = FuncAnimation(fig, update, frames=frames, interval=80)
    anim.save(out, writer=PillowWriter(fps=14), dpi=70)
    plt.close(fig)


def animate_inner_planets(traj, out: Path, years: float = 2.0, frames: int = 100, trail_yr: float = 0.3) -> None:
    names = ["Mercury", "Venus", "Earth", "Mars"]
    idx = [traj.names.index(n) for n in names]
    keep = traj.t <= years
    t = traj.t[keep]
    pos = traj.pos[keep]
    dt_sample = t[1] - t[0]
    trail = int(round(trail_yr / dt_sample))
    frame_index = np.linspace(0, len(t) - 1, frames).astype(int)

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.75, 1.75)
    ax.set_aspect("equal")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.set_title("Inner planets from J2000")
    ax.scatter([0], [0], s=60, color=INK, zorder=4)
    lines = [ax.plot([], [], color=PLANET_COLOUR[n], lw=1.5)[0] for n in names]
    dots = [ax.scatter([], [], s=38, color=PLANET_COLOUR[n], edgecolor=SURFACE, linewidth=1.2, zorder=5) for n in names]
    labels = [ax.text(0, 0, n, fontsize=8.5, color=INK2) for n in names]
    clock = ax.text(0.03, 0.96, "", transform=ax.transAxes, va="top", fontsize=10)

    def update(k):
        i = frame_index[k]
        lo = max(0, i - trail)
        for line, dot, label, j in zip(lines, dots, labels, idx):
            line.set_data(pos[lo : i + 1, j, 0], pos[lo : i + 1, j, 1])
            dot.set_offsets(pos[i : i + 1, j, :2])
            label.set_position((pos[i, j, 0] + 0.05, pos[i, j, 1] + 0.05))
        clock.set_text(f"t = {t[i]:.2f} yr")

    anim = FuncAnimation(fig, update, frames=frames, interval=70)
    anim.save(out, writer=PillowWriter(fps=15), dpi=70)
    plt.close(fig)


# --------------------------------------------------------------------------- experiment 5: Mercury's precession


def measure_precession(system, years: float, gr: bool, sample_dt: float = 0.02) -> dict:
    """Integrate with the adaptive DOP853 method and fit the rate of advance of Mercury's perihelion."""
    t_eval = np.arange(0.0, years, sample_dt)
    t, pos, vel = sim.integrate_adaptive(system, t_eval[-1], t_eval=t_eval, gr_body=1 if gr else None)
    r, v, mu = sim.jacobi_orbit(pos, vel, system.mass, 1)  # only the Sun is interior: heliocentric
    _, e_vec, h = sim.osculating_elements(r, v, mu)
    varpi = sim.perihelion_longitude(e_vec, h)
    rate, err = sim.perihelion_precession(t, varpi)
    return {"rate": rate, "error": err, "t": t, "varpi": varpi}


def precession_experiment(years: float = 100.0, contributions: bool = True) -> dict:
    mercury = sim.PLANET_BY_NAME["Mercury"]
    two = sim.two_body(mercury.a, mercury.e, mercury.mass, name="Mercury")
    full = sim.solar_system()
    runs = {
        "two_body_newton": measure_precession(two, years, False),
        "two_body_gr": measure_precession(two, years, True),
        "n_body_newton": measure_precession(full, years, False),
        "n_body_gr": measure_precession(full, years, True),
    }
    out = {
        "years": years,
        "runs": runs,
        "gr_analytic": sim.gr_precession_arcsec_per_century(mercury.a, mercury.e, mercury.mass),
        "contributions": {},
    }
    if contributions:
        for planet in sim.PLANETS[1:]:
            pair = sim.solar_system(["Mercury", planet.name])
            out["contributions"][planet.name] = measure_precession(pair, years, False)["rate"]
    return out


def plot_precession(data: dict, out: Path) -> None:
    runs = data["runs"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.2, 5.0), gridspec_kw={"width_ratios": [1.25, 1]})
    styles = (
        ("n_body_gr", BLUE, "All planets + general relativity"),
        ("n_body_newton", ORANGE, "All planets, Newtonian gravity only"),
        ("two_body_gr", VIOLET, "Sun and Mercury only + general relativity"),
        ("two_body_newton", MUTED, "Sun and Mercury only, Newtonian"),
    )
    for key, colour, label in styles:
        run = runs[key]
        a.plot(run["t"], (run["varpi"] - run["varpi"][0]) / sim.ARCSEC, color=colour, lw=1.4, label=f"{label}: {run['rate']:.1f} ″/cy")
    a.set_xlabel("Time (years)")
    a.set_ylabel("Advance of perihelion (arcseconds)")
    a.set_title("Mercury's perihelion advances; the four runs differ only in the physics")
    a.legend(loc="upper left")

    parts = dict(data["contributions"])
    ordered = sorted(parts.items(), key=lambda kv: kv[1])
    labels = [f"{n}" for n, _ in ordered]
    values = [v for _, v in ordered]
    labels.append("General relativity")
    values.append(runs["two_body_gr"]["rate"])
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    colours = [ORANGE if l == "General relativity" else BLUE for l in labels]
    y = np.arange(len(values))
    b.barh(y, values, color=colours, height=0.62)
    b.set_yticks(y)
    b.set_yticklabels(labels)
    b.grid(axis="y", visible=False)
    for yi, val in zip(y, values):
        b.annotate(f"{val:.1f}" if val >= 1 else f"{val:.2f}", (val, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    b.set_xlim(0, max(values) * 1.16)
    b.set_xlabel("Contribution to the perihelion advance (arcsec / century)")
    total = sum(values)
    obs, obs_err = sim.MERCURY_OBSERVED_PRECESSION
    b.set_title("Where the advance comes from")
    b.text(0.97, 0.16, f"sum of these: {total:.1f}\nobserved: {obs:.2f} ± {obs_err:.2f}", transform=b.transAxes, ha="right", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- driver


def report(results: dict) -> None:
    c = results["conservation"]
    print(f"\nWhole solar system, leapfrog, {c['years']:.0f} years, dt = {c['dt_days']} d ({c['steps']:,} steps)")
    print(f"  energy varies by {c['energy_relative_range']:.1e} (relative), angular momentum by {c['angular_momentum_relative_change']:.1e}")
    print("\nPlanet      T_sim (yr)  T_JPL (yr)   diff    a_sim (AU)  (T²/a³)/theory  e_fit    1/r fit residual  dA/dt spread")
    for r in results["planets"]:
        print(f"  {r['name']:8s} {r['period_sim_yr']:10.4f} {r['period_reference_yr']:10.4f} {r['period_difference_percent']:+6.2f}% {r['a_sim_au']:10.4f} {r['kepler3_ratio']:12.5f} {r['e_fit']:9.5f} {r['first_law_max_residual']:12.1e} {r['areal_velocity_spread']:14.1e}")
    k = results["third_law_fit"]
    print(f"\nKepler's third law: T ∝ a^k with k = {k['slope']:.5f} ± {k['slope_error']:.5f} (expected 1.5)")
    print("\nIntegrator convergence (measured order of accuracy)")
    for method, d in results["convergence"]["methods"].items():
        print(f"  {sim.INTEGRATOR_LABEL[method]:18s} measured {d['measured_order']:.2f}, expected {d['expected_order']}")
    print(f"\nEnergy error after {results['energy_orbits']:.0f} orbits at 100 steps/orbit")
    for method, d in results["energy_final"].items():
        print(f"  {sim.INTEGRATOR_LABEL[method]:18s} final {d['final']:.1e}   worst {d['max']:.1e}")
    s = results["second_law"]
    print(f"\nKepler's second law: Mercury's {s['mercury']['n_wedges']} equal-time wedges differ by {s['mercury']['area_spread']:.1e}; wedges sum to {s['mercury']['sum_over_ellipse']:.6f} of the ellipse")
    h = s["halley"]
    print(f"  Halley: {h['perihelion_au']:.3f} to {h['aphelion_au']:.2f} AU, speed {h['min_speed_kms']:.2f} to {h['max_speed_kms']:.1f} km/s, dA/dt varies by {h['areal_velocity_spread']:.1e}, orbit closes to {h['closure_error_au']:.1e} AU")
    p = results["precession"]
    print(f"\nMercury's perihelion precession over {p['years']:.0f} years (arcsec/century)")
    print(f"  Sun + Mercury, Newtonian      {p['two_body_newton']['rate']:9.3f}  (expect 0)")
    print(f"  Sun + Mercury, + GR           {p['two_body_gr']['rate']:9.3f} ± {p['two_body_gr']['error']:.3f}  (analytic {p['gr_analytic']:.3f}; Park et al. {sim.MERCURY_GR_PRECESSION:.3f})")
    print(f"  All planets, Newtonian        {p['n_body_newton']['rate']:9.3f} ± {p['n_body_newton']['error']:.3f}  (secular theory: 532.3)")
    print(f"  All planets + GR              {p['n_body_gr']['rate']:9.3f} ± {p['n_body_gr']['error']:.3f}  (observed {sim.MERCURY_OBSERVED_PRECESSION[0]:.2f} ± {sim.MERCURY_OBSERVED_PRECESSION[1]:.2f})")
    if p["contributions"]:
        parts = ", ".join(f"{k} {v:.2f}" for k, v in p["contributions"].items())
        print(f"  by planet: {parts}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="shorter runs for a fast check (less accurate precession)")
    parser.add_argument("--no-animation", action="store_true", help="skip the GIFs")
    parser.add_argument("--out-dir", default="figures", help="where figures are written (default: figures)")
    parser.add_argument("--results", default="results.json", help="where the numbers are saved (default: results.json)")
    args = parser.parse_args(argv)

    apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    def stage(name):
        print(f"[{time.time() - start:5.0f}s] {name}", flush=True)

    stage("simulating the solar system")
    traj, conservation = run_solar_system(years=200.0 if args.quick else 340.0)
    rows = analyse_planets(traj)
    fit_slope, fit_err, fit_c = sim.fit_kepler_third_law([r["a_sim_au"] for r in rows], [r["period_sim_yr"] for r in rows])
    third_law = {"slope": fit_slope, "slope_error": fit_err, "intercept": fit_c}
    plot_solar_system(traj, rows, out_dir / "fig1_solar_system.png")
    plot_third_law(rows, third_law, out_dir / "fig5_kepler_third_law.png")

    stage("comparing integrators")
    energy = integrator_energy_experiment(n_orbits=200 if args.quick else 1000)
    conv = convergence_experiment()
    plot_integrators(energy, out_dir / "fig2_integrators.png")
    plot_convergence(conv, out_dir / "fig3_convergence.png")

    stage("Kepler's second law")
    second = second_law_experiment()
    plot_second_law(second, out_dir / "fig4_kepler_second_law.png")

    stage("Mercury's precession (the slow one)")
    prec = precession_experiment(years=25.0 if args.quick else 100.0, contributions=not args.quick)
    if not args.quick:
        plot_precession(prec, out_dir / "fig6_mercury_precession.png")

    if not args.no_animation:
        stage("animations")
        animate_halley(out_dir / "halley_second_law.gif")
        animate_inner_planets(traj, out_dir / "inner_planets.gif")

    results = {
        "conservation": conservation,
        "planets": rows,
        "third_law_fit": third_law,
        "convergence": {"dt_days": conv["dt_days"], "methods": conv["methods"]},
        "energy_orbits": float(energy["leapfrog"]["orbits"][-1]),
        "energy_final": {m: {"final": float(d["energy_error"][-1]), "max": float(d["energy_error"].max())} for m, d in energy.items()},
        "second_law": {
            "mercury": {k: v for k, v in second["mercury"].items() if k not in ("paths", "areas")},
            "halley": {k: v for k, v in second["halley"].items() if not isinstance(v, np.ndarray)},
        },
        "precession": {
            "years": prec["years"],
            "gr_analytic": prec["gr_analytic"],
            "contributions": prec["contributions"],
            **{k: {"rate": v["rate"], "error": v["error"]} for k, v in prec["runs"].items()},
        },
    }
    report(results)
    Path(args.results).write_text(json.dumps(results, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)) + "\n")
    print(f"\nDone in {time.time() - start:.0f} s. Figures in {out_dir}/, numbers in {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
