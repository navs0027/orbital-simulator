"""Tests for orbital_simulator.py and run_experiments.py. Run with:  python test_orbital_simulator.py   (or: pytest)

None of these need the internet. They check the maths against cases that can be worked out by
hand or against exact results (the Kepler solution of the two-body problem, conservation laws,
the analytic general-relativity precession), then check that the experiments run and draw
their figures. They test the code, not the astrophysics; the comparison with the real planets
in README.md is what tests the astrophysics.
"""
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import orbital_simulator as sim  # noqa: E402
import run_experiments as rx  # noqa: E402

MERCURY = sim.PLANET_BY_NAME["Mercury"]
EARTH = sim.PLANET_BY_NAME["Earth"]


# ---------------------------------------------------------------- constants and data


def test_units_are_consistent():
    # In AU / year / solar-mass units G M_sun is 4 pi^2 to within the ~4e-5 that the Earth's
    # orbit is not exactly 1 AU (Earth-Moon mass, Julian vs sidereal year)
    assert abs(sim.G / (4 * np.pi**2) - 1.0) < 1e-4
    assert abs(sim.KMS_PER_AU_YR - 4.74047) < 1e-4
    assert abs(sim.C_AU_YR - 63241.077) < 0.01
    assert abs(sim.ARCSEC - 4.848137e-6) < 1e-11
    assert abs(sim.DAY * 365.25 - 1.0) < 1e-15


def test_planet_table():
    assert [p.name for p in sim.PLANETS] == ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    assert all(p.a > q.a for p, q in zip(sim.PLANETS[1:], sim.PLANETS[:-1])), "listed outwards"
    assert abs(sim.PLANET_BY_NAME["Jupiter"].mass - 1 / 1047.3486) < 1e-12
    # sidereal periods implied by the JPL mean-longitude rates
    assert abs(EARTH.reference_period - 1.00002) < 1e-5
    assert abs(sim.PLANET_BY_NAME["Jupiter"].reference_period - 11.8626) < 1e-3
    assert abs(MERCURY.reference_period - 0.24085) < 1e-4


# ---------------------------------------------------------------- Kepler's equation and initial conditions


def test_kepler_equation_is_solved():
    for e in (0.0, 0.1, 0.5, 0.9, 0.96714, 0.99):
        m = np.linspace(-9.0, 9.0, 721)
        big_e = sim.solve_kepler(m, e)
        reduced = np.mod(m + np.pi, 2 * np.pi) - np.pi
        assert np.abs(big_e - e * np.sin(big_e) - reduced).max() < 1e-12, f"e = {e}"
        assert np.all(np.abs(big_e) <= np.pi + 1e-12)


def test_circular_orbit_has_equal_anomalies():
    m = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
    assert np.allclose(sim.solve_kepler(m, 0.0), m)


def test_perifocal_state_has_the_right_conserved_quantities():
    a, e, mu = 2.5, 0.6, sim.G * 1.001
    for m in (0.0, 0.4, 1.7, np.pi, -2.2):
        r, v = sim.perifocal_state(a, e, m, mu)
        energy = 0.5 * v @ v - mu / np.linalg.norm(r)
        assert abs(energy / (-mu / (2 * a)) - 1.0) < 1e-12
        h = np.cross(r, v)[2]
        assert abs(h / np.sqrt(mu * a * (1 - e**2)) - 1.0) < 1e-12
    # perihelion and aphelion
    r0, v0 = sim.perifocal_state(a, e, 0.0, mu)
    assert np.allclose(r0, [a * (1 - e), 0, 0])
    assert abs(np.linalg.norm(v0) - np.sqrt(mu * (1 + e) / (a * (1 - e)))) < 1e-12
    r_pi, _ = sim.perifocal_state(a, e, np.pi, mu)
    assert np.allclose(r_pi, [-a * (1 + e), 0, 0], atol=1e-12)


def test_rotation_matrix_tilts_the_orbit_by_the_inclination():
    incl, node, arg = np.radians(23.0), np.radians(70.0), np.radians(140.0)
    rot = sim.rotation_to_ecliptic(incl, node, arg)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-14)
    assert abs(np.linalg.det(rot) - 1.0) < 1e-14
    normal = rot @ [0.0, 0.0, 1.0]  # orbit normal
    assert abs(np.degrees(np.arccos(normal[2])) - 23.0) < 1e-10
    node_line = rot @ [np.cos(arg), -np.sin(arg), 0.0]  # perifocal ascending node direction, in the ecliptic
    assert abs(node_line[2]) < 1e-14
    assert abs(np.degrees(np.arctan2(node_line[1], node_line[0])) - 70.0) < 1e-10


def test_earth_at_j2000():
    r, v = sim.heliocentric_state(EARTH)
    assert abs(np.linalg.norm(r) - 0.98331) < 2e-4, "two days before perihelion on 1 Jan 2000"
    assert abs(np.linalg.norm(v) * sim.KMS_PER_AU_YR - 30.29) < 0.02
    assert abs(r[2]) < 1e-4, "Earth is in the ecliptic"


def test_solar_system_starts_at_rest_in_the_barycentric_frame():
    s = sim.solar_system()
    assert len(s.names) == 9 and s.names[0] == "Sun"
    assert np.abs((s.mass[:, None] * s.pos).sum(axis=0)).max() < 1e-15, "centre of mass at the origin"
    assert np.abs(sim.linear_momentum(s.vel, s.mass)).max() < 1e-15
    sun = np.linalg.norm(s.pos[0])
    assert 0.0005 < sun < 0.011, "the Sun sits within about a solar radius (0.005 AU) of the barycentre"
    assert s.index("Mars") == 4


def test_subset_of_planets():
    s = sim.solar_system(["Mercury", "Jupiter"])
    assert s.names == ["Sun", "Mercury", "Jupiter"]


def test_two_body_sets_up_the_requested_orbit():
    s = sim.two_body(MERCURY.a, MERCURY.e, MERCURY.mass, name="Mercury")
    rel = s.pos[1] - s.pos[0]
    assert abs(np.linalg.norm(rel) - MERCURY.a * (1 - MERCURY.e)) < 1e-12, "starts at perihelion"
    assert np.abs(sim.linear_momentum(s.vel, s.mass)).max() < 1e-16
    period = sim.orbital_period(MERCURY.a, MERCURY.mass)
    assert abs(period - 0.24084) < 1e-4
    assert abs(sim.orbital_period(1.0) - 1.0) < 1e-4, "Kepler: a = 1 AU gives one year"


# ---------------------------------------------------------------- the physics


def test_two_body_acceleration_is_newtons_law():
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    mass = np.array([1.0, 0.5])
    acc = sim.accelerations(pos, mass)
    assert np.allclose(acc[1], [-sim.G * 1.0 / 4.0, 0, 0])
    assert np.allclose(acc[0], [sim.G * 0.5 / 4.0, 0, 0])
    far = sim.accelerations(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]), mass)
    assert abs(far[1, 0] / acc[1, 0] - 0.25) < 1e-14, "inverse-square law"


def test_forces_obey_newtons_third_law():
    rng = np.random.default_rng(1)
    pos = rng.normal(size=(6, 3))
    mass = rng.uniform(0.1, 2.0, size=6)
    acc = sim.accelerations(pos, mass)
    assert np.abs((mass[:, None] * acc).sum(axis=0)).max() < 1e-12, "total force is zero"


def test_potential_energy_of_a_pair_and_of_a_series():
    pos = np.array([[0.0, 0, 0], [3.0, 4.0, 0]])
    mass = np.array([2.0, 0.5])
    assert abs(sim.potential_energy(pos, mass) - (-sim.G * 2.0 * 0.5 / 5.0)) < 1e-14
    series = np.stack([pos, 2 * pos])
    assert np.allclose(sim.potential_energy(series, mass), [-sim.G / 5.0, -sim.G / 10.0])


def test_total_energy_of_a_kepler_orbit_is_minus_gmm_over_2a():
    s = sim.two_body(1.7, 0.4, 3e-6)
    expected = -sim.G * 1.0 * 3e-6 / (2 * 1.7)
    assert abs(sim.total_energy(s.pos, s.vel, s.mass) / expected - 1.0) < 1e-9


def test_run_shapes_and_times():
    s = sim.two_body(1.0, 0.1, 3e-6)
    traj = sim.run(s, "leapfrog", 0.01, 100, save_every=10)
    assert traj.pos.shape == (11, 2, 3) and traj.vel.shape == (11, 2, 3)
    assert np.allclose(traj.t, 0.1 * np.arange(11))
    assert np.array_equal(traj.pos[0], s.pos), "the first frame is the initial state"
    assert np.array_equal(s.pos, sim.two_body(1.0, 0.1, 3e-6).pos), "run() does not modify the input"


# ---------------------------------------------------------------- integrators


def test_symplectic_methods_conserve_angular_momentum_exactly_and_euler_does_not():
    s = sim.two_body(EARTH.a, EARTH.e, EARTH.mass)
    period = sim.orbital_period(EARTH.a, EARTH.mass)
    change = {}
    for method in sim.INTEGRATORS:
        traj = sim.run(s, method, period / 100, 2000, save_every=20)
        ang = sim.angular_momentum(traj.pos, traj.vel, traj.mass)
        change[method] = np.abs(ang - ang[0]).max() / np.linalg.norm(ang[0])
    assert change["leapfrog"] < 1e-12
    assert change["symplectic_euler"] < 1e-12
    assert change["euler"] > 1e-2, "explicit Euler adds angular momentum every step"


def test_energy_error_is_bounded_for_leapfrog_but_grows_for_euler():
    s = sim.two_body(EARTH.a, EARTH.e, EARTH.mass)
    period = sim.orbital_period(EARTH.a, EARTH.mass)
    err = {}
    for method in ("euler", "leapfrog"):
        traj = sim.run(s, method, period / 100, 10000, save_every=10)  # 100 orbits
        energy = sim.total_energy(traj.pos, traj.vel, traj.mass)
        err[method] = np.abs(energy / energy[0] - 1.0)
    first, last = err["leapfrog"][:500], err["leapfrog"][500:]
    assert err["leapfrog"].max() < 2e-4
    assert last.max() < 1.05 * first.max(), "no drift: the second half is no worse than the first"
    assert err["euler"][-1] > 0.1, "Euler spirals outwards and gains energy"


def _kepler_error(method, steps_per_orbit, fraction=0.6):
    """Position error against the exact Kepler orbit, as a fraction of a, after `fraction` of an orbit."""
    s = sim.two_body(MERCURY.a, MERCURY.e, MERCURY.mass)
    period = sim.orbital_period(MERCURY.a, MERCURY.mass)
    n = int(round(fraction * steps_per_orbit))
    traj = sim.run(s, method, period / steps_per_orbit, n)
    mu = sim.G * (1 + MERCURY.mass)
    exact = sim.kepler_positions(MERCURY.a, MERCURY.e, np.sqrt(mu / MERCURY.a**3) * n * period / steps_per_orbit)[0]
    rel = traj.pos[-1, 1] - traj.pos[-1, 0]
    return np.linalg.norm(rel[:2] - exact) / MERCURY.a


def test_integrators_converge_at_their_stated_orders():
    plans = {
        "euler": (400, 800, 1600),
        "symplectic_euler": (400, 800, 1600),
        "leapfrog": (200, 400, 800),
    }
    for method, steps in plans.items():
        errors = np.array([_kepler_error(method, n) for n in steps])
        slope = np.polyfit(np.log(1.0 / np.array(steps)), np.log(errors), 1)[0]
        expected = sim.INTEGRATOR_ORDER[method]
        assert abs(slope - expected) < 0.3, f"{method}: measured order {slope:.2f}, expected {expected}"


def test_reference_solver_closes_a_kepler_orbit_to_high_accuracy():
    s = sim.two_body(MERCURY.a, MERCURY.e, MERCURY.mass)
    period = sim.orbital_period(MERCURY.a, MERCURY.mass)
    t, pos, vel = sim.integrate_adaptive(s, 10 * period, t_eval=np.array([0.0, 10 * period]))
    assert np.linalg.norm(pos[-1] - pos[0]) < 1e-8, "back at perihelion after ten orbits"
    e0, e1 = sim.total_energy(pos[0], vel[0], s.mass), sim.total_energy(pos[-1], vel[-1], s.mass)
    assert abs(e1 / e0 - 1.0) < 1e-10


# ---------------------------------------------------------------- measuring Kepler's laws


def test_measure_periods_on_a_circle():
    t = np.linspace(0.0, 7.0, 7001)
    for sign in (+1, -1):
        r = np.column_stack([np.cos(np.pi * t), sign * np.sin(np.pi * t), 0 * t])  # period 2
        periods = sim.measure_periods(t, r)
        assert len(periods) == 3
        assert np.allclose(periods, 2.0, atol=1e-6)


def test_measure_periods_rejects_a_short_run():
    t = np.linspace(0.0, 0.5, 50)
    r = np.column_stack([np.cos(t), np.sin(t), 0 * t])
    try:
        sim.measure_periods(t, r)
    except ValueError:
        return
    raise AssertionError("a run shorter than one orbit should raise ValueError")


def test_orbit_equation_fit_recovers_an_exact_ellipse_in_a_tilted_plane():
    a, e = 2.0, 0.3
    xy = sim.kepler_positions(a, e, np.linspace(0, 2 * np.pi, 400, endpoint=False))
    rot = sim.rotation_to_ecliptic(np.radians(40), np.radians(25), np.radians(80))
    r = (rot @ np.column_stack([xy, np.zeros(len(xy))]).T).T
    fit = sim.fit_orbit_equation(r)
    assert abs(fit["a"] - a) < 1e-10 and abs(fit["e"] - e) < 1e-10
    assert fit["max_relative_residual"] < 1e-10


def test_orbit_equation_fit_detects_a_non_kepler_path():
    """Under a force law that is not 1/r^2 the orbit is not a closed ellipse with the Sun at
    the focus, and the fit residual must show it: the test that the residual can fail."""
    a, e = 2.0, 0.3
    phi = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    p = a * (1 - e**2)
    r_true = p / (1 + e * np.cos(phi))
    r = r_true * (1 + 0.05 * np.cos(3 * phi))  # a 5% third-harmonic distortion
    path = np.column_stack([r * np.cos(phi), r * np.sin(phi), 0 * phi])
    assert sim.fit_orbit_equation(path)["max_relative_residual"] > 0.02


def test_third_law_fit_on_synthetic_planets():
    a = np.array([0.4, 1.0, 1.5, 5.2, 9.5, 19.2, 30.1])
    k, err, c = sim.fit_kepler_third_law(a, a**1.5)
    assert abs(k - 1.5) < 1e-12 and abs(c) < 1e-12 and err < 1e-12
    rng = np.random.default_rng(0)
    k, err, _ = sim.fit_kepler_third_law(a, a**1.4 * np.exp(rng.normal(0, 0.005, a.size)))
    assert abs(k - 1.4) < 4 * err + 0.01


def test_swept_area_of_a_quarter_circle():
    path = sim.kepler_positions(2.0, 0.0, np.linspace(0.0, np.pi / 2, 4001))
    assert abs(sim.swept_area(path) - np.pi * 4.0 / 4) < 1e-6


def test_areal_velocity_is_constant_and_equals_half_sqrt_mu_p():
    s = sim.two_body(MERCURY.a, MERCURY.e, MERCURY.mass)
    period = sim.orbital_period(MERCURY.a, MERCURY.mass)
    traj = sim.run(s, "leapfrog", period / 500, 1000, save_every=5)
    r, v = sim.relative_orbit(traj.pos, traj.vel, 1)
    dadt = sim.areal_velocity(r, v)
    assert (dadt.max() - dadt.min()) / dadt.mean() < 1e-12
    mu = sim.G * (1 + MERCURY.mass)
    assert abs(dadt.mean() / (0.5 * np.sqrt(mu * MERCURY.a * (1 - MERCURY.e**2))) - 1.0) < 1e-9


def test_osculating_elements_recover_the_orbit():
    mu = sim.G * (1 + MERCURY.mass)
    r, v = sim.perifocal_state(MERCURY.a, MERCURY.e, 1.234, mu)
    a, e_vec, h = sim.osculating_elements(r[None, :], v[None, :], mu)
    assert abs(a[0] - MERCURY.a) < 1e-12
    assert abs(np.linalg.norm(e_vec[0]) - MERCURY.e) < 1e-12
    assert np.allclose(e_vec[0] / np.linalg.norm(e_vec[0]), [1, 0, 0], atol=1e-12), "e-vector points at perihelion"
    assert h[0, 2] > 0 and abs(h[0, 0]) + abs(h[0, 1]) < 1e-15


def test_jacobi_orbit_for_the_innermost_planet_is_the_heliocentric_orbit():
    s = sim.solar_system(["Mercury", "Jupiter"])
    traj = sim.run(s, "leapfrog", 0.01, 20)
    r_j, v_j, mu = sim.jacobi_orbit(traj.pos, traj.vel, traj.mass, 1)
    r_h, v_h = sim.relative_orbit(traj.pos, traj.vel, 1)
    assert np.allclose(r_j, r_h) and np.allclose(v_j, v_h)
    assert abs(mu - sim.G * (1 + MERCURY.mass)) < 1e-15


def test_jacobi_orbit_uses_the_interior_centre_of_mass():
    s = sim.solar_system(["Mercury", "Jupiter"])
    traj = sim.run(s, "leapfrog", 0.01, 5)
    r, v, mu = sim.jacobi_orbit(traj.pos, traj.vel, traj.mass, 2)
    m_in = traj.mass[:2].sum()
    centre = (traj.mass[0] * traj.pos[:, 0] + traj.mass[1] * traj.pos[:, 1]) / m_in
    assert np.allclose(r, traj.pos[:, 2] - centre)
    assert abs(mu - sim.G * (m_in + traj.mass[2])) < 1e-15


# ---------------------------------------------------------------- general relativity and precession


def test_gr_precession_formula_for_mercury():
    """6 pi G M / (c^2 a (1 - e^2)) per orbit is 42.98 arcsec per century for Mercury."""
    rate = sim.gr_precession_arcsec_per_century(MERCURY.a, MERCURY.e, MERCURY.mass)
    assert abs(rate - sim.MERCURY_GR_PRECESSION) < 0.01
    per_orbit = sim.gr_precession_per_orbit(MERCURY.a, MERCURY.e) / sim.ARCSEC
    assert abs(per_orbit - 0.1035) < 5e-4, "about a tenth of an arcsecond per orbit"


def test_gr_correction_for_a_circular_orbit_is_a_small_outward_push():
    gm = sim.G
    r = np.array([0.5, 0.0, 0.0])
    v = np.array([0.0, np.sqrt(gm / 0.5), 0.0])
    acc = sim.gr_acceleration(r, v, gm)
    expected = 3 * gm**2 / (sim.C_AU_YR**2 * 0.5**3)
    assert abs(acc[0] / expected - 1.0) < 1e-12 and abs(acc[1]) < 1e-15
    newton = gm / 0.5**2
    assert 1e-8 < expected / newton < 1e-7, "of order (v/c)^2 = 1e-8"


def test_newtonian_two_body_orbit_does_not_precess_and_gr_does():
    s = sim.two_body(MERCURY.a, MERCURY.e, MERCURY.mass, name="Mercury")
    newton = rx.measure_precession(s, 25.0, gr=False)
    assert abs(newton["rate"]) < 0.01, "an inverse-square force gives a closed ellipse"
    gr = rx.measure_precession(s, 25.0, gr=True)
    analytic = sim.gr_precession_arcsec_per_century(MERCURY.a, MERCURY.e, MERCURY.mass)
    assert abs(gr["rate"] / analytic - 1.0) < 0.02, f"measured {gr['rate']:.2f}, analytic {analytic:.2f}"


def _orbit_with_known_angles(incl_deg, node_deg, arg_deg):
    mu = sim.G * (1 + MERCURY.mass)
    r, v = sim.perifocal_state(MERCURY.a, MERCURY.e, 0.7, mu)
    rot = sim.rotation_to_ecliptic(np.radians(incl_deg), np.radians(node_deg), np.radians(arg_deg))
    return sim.osculating_elements((rot @ r)[None, :], (rot @ v)[None, :], mu)


def test_longitude_of_perihelion_is_node_plus_argument_of_perihelion():
    _, e_vec, h = _orbit_with_known_angles(7.0, 48.3, 29.1)
    assert abs(np.degrees(sim.perihelion_longitude(e_vec, h))[0] - (48.3 + 29.1)) < 1e-9
    # an orbit in the ecliptic has no node: the angle is then measured from +x
    _, e_vec, h = _orbit_with_known_angles(0.0, 0.0, 77.0)
    assert abs(np.degrees(sim.perihelion_longitude(e_vec, h))[0] - 77.0) < 1e-9


def test_precession_rate_fit_on_a_straight_line():
    t = np.linspace(0.0, 50.0, 200)
    varpi = 1.0 + 2.0e-5 * t  # radians per year
    rate, err = sim.perihelion_precession(t, varpi)
    assert abs(rate - 2.0e-5 * 100 / sim.ARCSEC) < 1e-6 and err < 1e-6


# ---------------------------------------------------------------- the experiments


def test_second_law_experiment():
    data = rx.second_law_experiment(n_wedges=12)
    m = data["mercury"]
    assert m["area_spread"] < 1e-6, "twelve equal-time wedges have equal area"
    assert abs(m["sum_over_ellipse"] - 1.0) < 1e-6, "and together they make the whole ellipse"
    h = data["halley"]
    assert abs(h["perihelion_au"] - 0.586) < 0.001 and abs(h["aphelion_au"] - 35.08) < 0.01
    assert abs(h["max_speed_kms"] - 54.57) < 0.05
    assert abs(h["period_yr"] - 75.3) < 0.05
    assert h["max_speed_kms"] / h["min_speed_kms"] > 55, "the comet speeds up 60-fold at perihelion"
    assert h["areal_velocity_spread"] < 1e-9 and h["closure_error_au"] < 1e-6


def test_convergence_experiment_matches_theory():
    conv = rx.convergence_experiment(steps=(200, 400, 800, 1600))
    for method, d in conv["methods"].items():
        assert abs(d["measured_order"] - d["expected_order"]) < 0.3, method


def test_short_solar_system_run_conserves_the_invariants():
    traj, cons = rx.run_solar_system(years=5.0, dt_days=0.5, save_every=10)
    assert cons["energy_relative_range"] < 1e-6
    assert cons["angular_momentum_relative_change"] < 1e-12
    assert cons["momentum_max"] < 1e-14
    rows = rx.analyse_planets(traj)
    assert [r["name"] for r in rows] == ["Mercury", "Venus", "Earth", "Mars"], "only planets with two orbits in 5 years"
    for row in rows:
        assert abs(row["kepler3_ratio"] - 1.0) < 2e-3, row["name"]
        assert row["first_law_max_residual"] < 5e-4
        assert row["areal_velocity_spread"] < 5e-3
        assert abs(row["period_difference_percent"]) < 0.3


def test_figures_and_animations_are_drawn():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        rx.apply_style()
        traj, _ = rx.run_solar_system(years=5.0, dt_days=0.5, save_every=10)
        rows = rx.analyse_planets(traj)
        k, err, c = sim.fit_kepler_third_law([r["a_sim_au"] for r in rows], [r["period_sim_yr"] for r in rows])
        rx.plot_solar_system(traj, rows, out / "f1.png")
        rx.plot_third_law(rows, {"slope": k, "slope_error": err, "intercept": c}, out / "f5.png")
        rx.plot_integrators(rx.integrator_energy_experiment(n_orbits=25), out / "f2.png")
        rx.plot_convergence(rx.convergence_experiment(steps=(200, 400, 800)), out / "f3.png")
        rx.plot_second_law(rx.second_law_experiment(), out / "f4.png")
        rx.plot_precession(rx.precession_experiment(years=2.0, contributions=True), out / "f6.png")
        for name in ("f1", "f2", "f3", "f4", "f5", "f6"):
            assert (out / f"{name}.png").stat().st_size > 20_000, name

        rx.animate_halley(out / "halley.gif", frames=6)
        rx.animate_inner_planets(traj, out / "inner.gif", years=2.0, frames=6)
        for name in ("halley", "inner"):
            with Image.open(out / f"{name}.gif") as im:
                assert im.n_frames == 6, name


def test_quick_pipeline_runs_end_to_end():
    import json

    with tempfile.TemporaryDirectory() as tmp:
        rc = rx.main(["--quick", "--no-animation", "--out-dir", tmp, "--results", str(Path(tmp) / "r.json")])
        assert rc == 0
        results = json.loads((Path(tmp) / "r.json").read_text())
        assert results["energy_orbits"] == 200
        assert abs(results["third_law_fit"]["slope"] - 1.5) < 0.01
        assert abs(results["precession"]["two_body_gr"]["rate"] / results["precession"]["gr_analytic"] - 1.0) < 0.05
        assert (Path(tmp) / "fig1_solar_system.png").exists() and (Path(tmp) / "fig3_convergence.png").exists()


if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, func in tests:
        try:
            func()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - report every failure, not just the first
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
