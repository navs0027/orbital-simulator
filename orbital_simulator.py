"""A Newtonian N-body solar system simulator, built to test Kepler's laws and to measure
the precession of Mercury's orbit.

This module is the physics: units, planet data, three fixed-step integrators, a
high-accuracy reference solver (SciPy's solve_ivp), and the analysis routines that pull Kepler's laws out of a simulation.
`run_experiments.py` uses it to make the figures and animations, and `test_orbital_simulator.py`
tests it.

Units. Positions are in astronomical units (AU), times in Julian years (365.25 days) and
masses in solar masses. In these units G M_sun is 39.478..., very close to 4 pi^2, so every
number in the simulation is of order one and floating-point rounding is not a problem.
Positions and velocities are converted to SI only for reporting.

Coordinates. The frame is the barycentric J2000 ecliptic frame: origin at the centre of mass
of the whole system, x towards the March equinox of J2000, z towards the ecliptic north pole.
Body 0 is always the Sun.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

# --------------------------------------------------------------------------- units and constants

AU_M = 149_597_870_700.0  # metres in one AU (exact, IAU 2012)
DAY_S = 86_400.0
YEAR_S = 365.25 * DAY_S  # Julian year
GM_SUN_SI = 1.32712440018e20  # m^3 s^-2, heliocentric gravitational constant
C_SI = 299_792_458.0  # m/s, exact

G = GM_SUN_SI * YEAR_S**2 / AU_M**3  # G in AU^3 M_sun^-1 yr^-2 (i.e. G M_sun, since M_sun = 1)
C_AU_YR = C_SI * YEAR_S / AU_M  # speed of light in AU/yr
KMS_PER_AU_YR = AU_M / YEAR_S / 1000.0  # 1 AU/yr in km/s
DAY = 1.0 / 365.25  # one day, in years
ARCSEC = np.pi / (180.0 * 3600.0)  # one arcsecond, in radians
CENTURY = 100.0  # years

# --------------------------------------------------------------------------- planet data

# Mean Keplerian elements at J2000 (JD 2451545.0) from Table 1 of "Keplerian Elements for
# Approximate Positions of the Major Planets", E. M. Standish, JPL Solar System Dynamics,
# valid 1800-2050 AD: semi-major axis a (AU), eccentricity e, inclination i, mean longitude
# L, longitude of perihelion varpi and longitude of ascending node Omega (all degrees), and
# the rate of change of L (degrees per century). "Earth" is the Earth-Moon barycentre.
# mass_ratio is M_sun / M_planet (the planet's mass includes its moons).


@dataclass(frozen=True)
class PlanetData:
    name: str
    a: float
    e: float
    incl: float
    mean_long: float
    long_peri: float
    long_node: float
    mean_long_rate: float
    mass_ratio: float

    @property
    def mass(self) -> float:
        """Mass in solar masses."""
        return 1.0 / self.mass_ratio

    @property
    def reference_period(self) -> float:
        """Sidereal period in years implied by the JPL mean-longitude rate."""
        return CENTURY * 360.0 / self.mean_long_rate


PLANETS = [
    PlanetData("Mercury", 0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593, 149472.67411175, 6023600.0),
    PlanetData("Venus", 0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255, 58517.81538729, 408523.71),
    PlanetData("Earth", 1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0, 35999.37244981, 328900.56),
    PlanetData("Mars", 1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891, 19140.30268499, 3098708.0),
    PlanetData("Jupiter", 5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909, 3034.74612775, 1047.3486),
    PlanetData("Saturn", 9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448, 1222.49362201, 3497.898),
    PlanetData("Uranus", 19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503, 428.48202785, 22902.98),
    PlanetData("Neptune", 30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574, 218.45945325, 19412.24),
]
PLANET_BY_NAME = {p.name: p for p in PLANETS}

# Halley's Comet (JPL Small-Body Database): a = 17.834 AU, e = 0.96714, mass about 2.2e14 kg.
HALLEY_A = 17.834
HALLEY_E = 0.96714
HALLEY_MASS = 2.2e14 / 1.98841e30  # solar masses

# Mercury's observed perihelion precession about the Sun, arcsec per Julian century
# (Clemence 1947), and the general-relativity contribution (Park et al. 2017).
MERCURY_OBSERVED_PRECESSION = (574.10, 0.65)
MERCURY_GR_PRECESSION = 42.9799

# --------------------------------------------------------------------------- Kepler's equation


def solve_kepler(mean_anomaly, e: float, tol: float = 1e-14, max_iter: int = 100):
    """Solve Kepler's equation E - e sin E = M for the eccentric anomaly E by Newton's method.

    M is reduced to (-pi, pi]; the returned E lies in the same interval, so cos E and sin E
    (all that positions need) are unaffected by the reduction. Starting from E = pi keeps
    the iteration safe for eccentric orbits such as Halley's (e = 0.967).
    """
    m = np.mod(np.asarray(mean_anomaly, dtype=float) + np.pi, 2 * np.pi) - np.pi
    ecc = np.asarray(e, dtype=float)
    big = ecc > 0.8
    E = np.where(big, np.pi * np.where(m >= 0, 1.0, -1.0), m + ecc * np.sin(m))
    for _ in range(max_iter):
        delta = (E - ecc * np.sin(E) - m) / (1.0 - ecc * np.cos(E))
        E = E - delta
        if np.all(np.abs(delta) < tol):
            break
    return E


def perifocal_state(a: float, e: float, mean_anomaly: float, mu: float):
    """Position and velocity in the orbital plane (x towards perihelion) for given a, e, M.

    mu is G(M_central + m). Returns two length-3 arrays with z = 0.
    """
    E = float(solve_kepler(mean_anomaly, e))
    root = np.sqrt(1.0 - e * e)
    n = np.sqrt(mu / a**3)
    denom = 1.0 - e * np.cos(E)
    r = np.array([a * (np.cos(E) - e), a * root * np.sin(E), 0.0])
    v = np.array([-a * n * np.sin(E) / denom, a * n * root * np.cos(E) / denom, 0.0])
    return r, v


def rotation_to_ecliptic(incl: float, node: float, arg_peri: float) -> np.ndarray:
    """Rotation matrix Rz(node) Rx(incl) Rz(arg_peri), angles in radians: it takes
    perifocal coordinates to ecliptic coordinates."""

    def rz(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def rx(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    return rz(node) @ rx(incl) @ rz(arg_peri)


def heliocentric_state(planet: PlanetData):
    """Heliocentric ecliptic position (AU) and velocity (AU/yr) of a planet at J2000."""
    node = np.radians(planet.long_node)
    arg_peri = np.radians(planet.long_peri - planet.long_node)
    mean_anomaly = np.radians(planet.mean_long - planet.long_peri)
    mu = G * (1.0 + planet.mass)
    r, v = perifocal_state(planet.a, planet.e, mean_anomaly, mu)
    rot = rotation_to_ecliptic(np.radians(planet.incl), node, arg_peri)
    return rot @ r, rot @ v


# --------------------------------------------------------------------------- the system


@dataclass
class System:
    """Masses (solar masses), positions (AU) and velocities (AU/yr) of N bodies. Body 0 is the Sun."""

    names: list
    mass: np.ndarray
    pos: np.ndarray
    vel: np.ndarray

    def copy(self) -> "System":
        return System(list(self.names), self.mass.copy(), self.pos.copy(), self.vel.copy())

    def index(self, name: str) -> int:
        return self.names.index(name)


def to_barycentre(names, mass, helio_pos, helio_vel) -> System:
    """Build a System from heliocentric data (Sun first, at rest at the origin), then shift to the
    frame in which the centre of mass is at rest at the origin."""
    mass = np.asarray(mass, dtype=float)
    pos = np.asarray(helio_pos, dtype=float)
    vel = np.asarray(helio_vel, dtype=float)
    r_cm = (mass[:, None] * pos).sum(axis=0) / mass.sum()
    v_cm = (mass[:, None] * vel).sum(axis=0) / mass.sum()
    return System(list(names), mass, pos - r_cm, vel - v_cm)


def solar_system(planets=None) -> System:
    """The Sun and the chosen planets (default: all eight) at J2000, in the barycentric frame."""
    chosen = PLANETS if planets is None else [PLANET_BY_NAME[p] for p in planets]
    names = ["Sun"] + [p.name for p in chosen]
    mass = [1.0] + [p.mass for p in chosen]
    pos = [np.zeros(3)]
    vel = [np.zeros(3)]
    for p in chosen:
        r, v = heliocentric_state(p)
        pos.append(r)
        vel.append(v)
    return to_barycentre(names, mass, pos, vel)


def two_body(a: float, e: float, body_mass: float, mean_anomaly: float = 0.0, name: str = "Planet") -> System:
    """The Sun and one body on a Keplerian orbit in the xy plane, perihelion along +x, in the
    barycentric frame. mean_anomaly = 0 starts the body at perihelion."""
    mu = G * (1.0 + body_mass)
    r, v = perifocal_state(a, e, mean_anomaly, mu)
    return to_barycentre(["Sun", name], [1.0, body_mass], [np.zeros(3), r], [np.zeros(3), v])


def orbital_period(a: float, body_mass: float = 0.0) -> float:
    """Kepler's third law with the two-body correction: T = 2 pi sqrt(a^3 / (G (M + m))), in years."""
    return 2.0 * np.pi * np.sqrt(a**3 / (G * (1.0 + body_mass)))


# --------------------------------------------------------------------------- physics


def accelerations(pos: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Newtonian gravitational acceleration of every body due to all the others.

    a_i = G sum_{j != i} m_j (r_j - r_i) / |r_j - r_i|^3
    """
    diff = pos[None, :, :] - pos[:, None, :]  # diff[i, j] = r_j - r_i
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)
    np.fill_diagonal(dist2, np.inf)  # no self-force: 1 / inf = 0
    weight = mass[None, :] * dist2**-1.5
    return G * (weight[:, :, None] * diff).sum(axis=1)


def potential_energy(pos, mass):
    """U = -G sum_{i<j} m_i m_j / r_ij, for a single snapshot (N, 3) or a series (S, N, 3)."""
    single = pos.ndim == 2
    p = pos[None] if single else pos
    diff = p[:, None, :, :] - p[:, :, None, :]
    dist = np.sqrt(np.einsum("sijk,sijk->sij", diff, diff))
    n = p.shape[1]
    iu = np.triu_indices(n, k=1)
    pair = (mass[iu[0]] * mass[iu[1]])[None, :] / dist[:, iu[0], iu[1]]
    u = -G * pair.sum(axis=1)
    return u[0] if single else u


def total_energy(pos, vel, mass):
    """Total mechanical energy in M_sun AU^2 yr^-2; accepts one snapshot or a time series."""
    if pos.ndim == 2:
        return 0.5 * (mass * (vel * vel).sum(axis=1)).sum() + potential_energy(pos, mass)
    ke = 0.5 * (mass[None, :] * (vel * vel).sum(axis=2)).sum(axis=1)
    return ke + potential_energy(pos, mass)


def angular_momentum(pos, vel, mass):
    """Total angular momentum vector about the origin; accepts one snapshot or a series."""
    if pos.ndim == 2:
        return (mass[:, None] * np.cross(pos, vel)).sum(axis=0)
    return (mass[None, :, None] * np.cross(pos, vel)).sum(axis=1)


def linear_momentum(vel, mass):
    if vel.ndim == 2:
        return (mass[:, None] * vel).sum(axis=0)
    return (mass[None, :, None] * vel).sum(axis=1)


# --------------------------------------------------------------------------- fixed-step integrators
#
# Every step function takes (pos, vel, acc, mass, dt), where acc is the acceleration at pos,
# and returns (pos, vel, acc) after one step, with acc evaluated at the new position.


def step_euler(pos, vel, acc, mass, dt):
    """Explicit Euler: both x and v are advanced using the old values. First order."""
    new_pos = pos + dt * vel
    new_vel = vel + dt * acc
    return new_pos, new_vel, accelerations(new_pos, mass)


def step_symplectic_euler(pos, vel, acc, mass, dt):
    """Semi-implicit (symplectic) Euler: update v first, then x with the *new* v. First order."""
    new_vel = vel + dt * acc
    new_pos = pos + dt * new_vel
    return new_pos, new_vel, accelerations(new_pos, mass)


def step_leapfrog(pos, vel, acc, mass, dt):
    """Leapfrog in kick-drift-kick form (velocity Verlet). Second order and symplectic."""
    half = vel + 0.5 * dt * acc
    new_pos = pos + dt * half
    new_acc = accelerations(new_pos, mass)
    return new_pos, half + 0.5 * dt * new_acc, new_acc


INTEGRATORS = {
    "euler": step_euler,
    "symplectic_euler": step_symplectic_euler,
    "leapfrog": step_leapfrog,
}
INTEGRATOR_ORDER = {"euler": 1, "symplectic_euler": 1, "leapfrog": 2}
INTEGRATOR_LABEL = {
    "euler": "Euler",
    "symplectic_euler": "Symplectic Euler",
    "leapfrog": "Leapfrog",
}


@dataclass
class Trajectory:
    """Saved output of a fixed-step run."""

    t: np.ndarray  # years, shape (S,)
    pos: np.ndarray  # AU, shape (S, N, 3)
    vel: np.ndarray  # AU/yr, shape (S, N, 3)
    mass: np.ndarray
    names: list


def run(system: System, method: str, dt: float, n_steps: int, save_every: int = 1) -> Trajectory:
    """Integrate with a fixed step dt (years) for n_steps, saving every save_every steps."""
    step = INTEGRATORS[method]
    pos, vel = system.pos.copy(), system.vel.copy()
    acc = accelerations(pos, system.mass)
    n_saved = n_steps // save_every + 1
    t_out = np.empty(n_saved)
    pos_out = np.empty((n_saved,) + pos.shape)
    vel_out = np.empty((n_saved,) + vel.shape)
    t_out[0], pos_out[0], vel_out[0] = 0.0, pos, vel
    k = 1
    for i in range(1, n_steps + 1):
        pos, vel, acc = step(pos, vel, acc, system.mass, dt)
        if i % save_every == 0:
            t_out[k], pos_out[k], vel_out[k] = i * dt, pos, vel
            k += 1
    return Trajectory(t_out, pos_out, vel_out, system.mass, list(system.names))


# --------------------------------------------------------------------------- reference solver (SciPy)


def gr_acceleration(r, v, gm: float = G):
    """First post-Newtonian correction to the acceleration of a test body orbiting a mass M
    (gm = G M), in harmonic coordinates:

        a = (G M / c^2 r^3) [ (4 G M / r - v^2) r + 4 (r . v) v ]

    r and v are the position and velocity relative to the central mass. It is what turns the
    closed Newtonian ellipse into a rosette: the perihelion advances by 6 pi G M / (c^2 a (1 - e^2))
    per orbit."""
    rn = np.linalg.norm(r)
    return gm / (C_AU_YR**2 * rn**3) * ((4.0 * gm / rn - v @ v) * r + 4.0 * (r @ v) * v)


def integrate_adaptive(
    system: System,
    t_end: float,
    t_eval=None,
    rtol: float = 1e-12,
    atol: float = 1e-14,
    gr_body: int | None = None,
):
    """High-accuracy reference solver. It is not one of the integrators being compared above: it
    hands the equations to SciPy's solve_ivp (8th-order Dormand-Prince method, DOP853), which
    chooses the step size automatically to keep the local error below rtol. It is used where a
    result needs far more accuracy than a fixed step can give cheaply: Halley's very eccentric
    orbit, and Mercury's precession of 43 arcseconds per century. Returns (t, pos, vel).

    If gr_body is given, the first post-Newtonian correction is added to that body's motion
    about the Sun (body 0), with the equal and opposite reaction on the Sun.
    """
    n = len(system.mass)
    mass = system.mass

    def rhs(_t, y):
        pos = y[: 3 * n].reshape(n, 3)
        vel = y[3 * n :].reshape(n, 3)
        acc = accelerations(pos, mass)
        if gr_body is not None:
            extra = gr_acceleration(pos[gr_body] - pos[0], vel[gr_body] - vel[0], G * mass[0])
            acc[gr_body] += extra
            acc[0] -= mass[gr_body] / mass[0] * extra
        return np.concatenate([vel.ravel(), acc.ravel()])

    y0 = np.concatenate([system.pos.ravel(), system.vel.ravel()])
    sol = solve_ivp(rhs, (0.0, t_end), y0, method="DOP853", t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"adaptive integration failed: {sol.message}")
    pos = sol.y[: 3 * n].T.reshape(-1, n, 3)
    vel = sol.y[3 * n :].T.reshape(-1, n, 3)
    return sol.t, pos, vel


# --------------------------------------------------------------------------- analysis: orbits


def relative_orbit(pos, vel, i: int, j: int = 0):
    """Position and velocity of body i relative to body j (default the Sun), for a series."""
    return pos[:, i, :] - pos[:, j, :], vel[:, i, :] - vel[:, j, :]


def jacobi_orbit(pos, vel, mass, i: int):
    """Orbit of body i about the centre of mass of every *interior* body 0 .. i-1 (a Jacobi
    coordinate), for a series of snapshots. Returns relative position, relative velocity and
    the gravitational parameter mu = G (M_interior + m_i).

    For Mercury the interior mass is just the Sun, so this is the ordinary heliocentric orbit.
    For the outer planets it matters: Jupiter drags the Sun about 0.005 AU around the
    barycentre, which makes heliocentric distances and semi-major axes wobble by ~0.5%, while
    the orbit about the interior centre of mass stays steady.
    """
    inner = slice(0, i)
    m_in = mass[inner].sum()
    w = mass[inner][None, :, None]
    r_c = (w * pos[:, inner, :]).sum(axis=1) / m_in
    v_c = (w * vel[:, inner, :]).sum(axis=1) / m_in
    return pos[:, i, :] - r_c, vel[:, i, :] - v_c, G * (m_in + mass[i])


def kepler_positions(a: float, e: float, mean_anomaly):
    """Exact positions (x, y) in the orbital plane, with the Sun at the origin and perihelion
    along +x, for an array of mean anomalies. Uses x = a (cos E - e), y = a sqrt(1-e^2) sin E."""
    big_e = np.atleast_1d(solve_kepler(mean_anomaly, e))
    return np.column_stack([a * (np.cos(big_e) - e), a * np.sqrt(1.0 - e * e) * np.sin(big_e)])


def osculating_elements(r, v, mu: float):
    """Osculating semi-major axis, eccentricity vector and angular momentum from relative
    position r and velocity v (arrays of shape (S, 3)), for the two-body problem with
    gravitational parameter mu = G (M + m).

        a = 1 / (2 / r - v^2 / mu),   h = r x v,   e_vec = (v x h) / mu - r / |r|

    e_vec points from the Sun to perihelion and its length is the eccentricity.
    """
    rn = np.linalg.norm(r, axis=1)
    v2 = np.einsum("sk,sk->s", v, v)
    a = 1.0 / (2.0 / rn - v2 / mu)
    h = np.cross(r, v)
    e_vec = np.cross(v, h) / mu - r / rn[:, None]
    return a, e_vec, h


def areal_velocity(r, v):
    """dA/dt = (1/2) |r x v|: the rate at which the radius vector sweeps out area."""
    return 0.5 * np.linalg.norm(np.cross(r, v), axis=1)


def orbit_plane_basis(h_mean):
    """Two orthonormal in-plane axes (e1, e2) for an orbit whose angular momentum points along h_mean."""
    n = h_mean / np.linalg.norm(h_mean)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - (ref @ n) * n
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def fit_orbit_equation(r):
    """Fit r(phi) = p / (1 + e cos(phi - phi0)) to a sampled orbit, to test Kepler's first law.

    The orbit equation is linear in u = 1/r:  u = A + B cos(phi) + C sin(phi), with
    A = 1/p, e = sqrt(B^2 + C^2) / A and p = a (1 - e^2). A linear least-squares fit therefore
    gives the semi-major axis and eccentricity of the ellipse with the Sun at one focus, and the
    residuals show how far the path departs from a perfect ellipse. r has shape (S, 3), taken
    about the Sun, over (about) one orbit.
    """
    h_mean = np.cross(r[:-1], r[1:]).sum(axis=0)
    e1, e2 = orbit_plane_basis(h_mean)
    x, y = r @ e1, r @ e2
    phi = np.arctan2(y, x)
    u = 1.0 / np.hypot(x, y)
    design = np.column_stack([np.ones_like(phi), np.cos(phi), np.sin(phi)])
    (A, B, C), *_ = np.linalg.lstsq(design, u, rcond=None)
    ecc = np.hypot(B, C) / A
    resid = (u - design @ np.array([A, B, C])) / u
    return {
        "a": 1.0 / (A * (1.0 - ecc**2)),
        "e": float(ecc),
        "max_relative_residual": float(np.abs(resid).max()),
        "rms_relative_residual": float(np.sqrt((resid**2).mean())),
    }


def measure_periods(t, r):
    """Sidereal periods from a sampled orbit: the times at which the heliocentric longitude
    (unwrapped) has advanced by 2 pi, 4 pi, ... Returns the array of successive periods."""
    theta = np.unwrap(np.arctan2(r[:, 1], r[:, 0]))
    if theta[-1] < theta[0]:
        theta = -theta  # retrograde orbit: count the angle the other way
    turns = int(np.floor((theta[-1] - theta[0]) / (2 * np.pi)))
    if turns < 1:
        raise ValueError("the run is shorter than one orbit")
    marks = theta[0] + 2 * np.pi * np.arange(1, turns + 1)
    times = np.concatenate([[t[0]], np.interp(marks, theta, t)])
    return np.diff(times)


def fit_kepler_third_law(a, period):
    """Fit log10(T) = k log10(a) + c. Kepler's third law says k = 3/2. Returns k, its
    standard error, and c."""
    x, y = np.log10(a), np.log10(period)
    design = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = len(x) - 2
    cov = (resid @ resid / dof) * np.linalg.inv(design.T @ design)
    return float(beta[0]), float(np.sqrt(cov[0, 0])), float(beta[1])


def swept_area(path):
    """Area swept out by the radius vector along a sampled path (S, 2), measured from the origin
    (the Sun): the sum of the triangle areas (1/2)|r_i x r_{i+1}|. It uses only positions, so it
    is a geometric measurement, independent of any velocity or conservation law."""
    x, y = path[:, 0], path[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


# --------------------------------------------------------------------------- analysis: precession


def gr_precession_per_orbit(a: float, e: float) -> float:
    """General-relativistic advance of the perihelion per orbit, in radians:
    6 pi G M / (c^2 a (1 - e^2))."""
    return 6.0 * np.pi * G / (C_AU_YR**2 * a * (1.0 - e**2))


def gr_precession_arcsec_per_century(a: float, e: float, body_mass: float = 0.0) -> float:
    orbits_per_century = CENTURY / orbital_period(a, body_mass)
    return gr_precession_per_orbit(a, e) * orbits_per_century / ARCSEC


def perihelion_longitude(e_vec, h):
    """Longitude of perihelion, varpi = Omega + omega, in radians (unwrapped), from the
    eccentricity vector and angular momentum vector of a series of osculating orbits.

    Omega is the longitude of the ascending node and omega the angle from the node to
    perihelion measured in the orbital plane. Adding them gives the angle used in tables of
    planetary elements. (Simply taking the angle of e_vec projected onto the ecliptic would
    be off by a few arcseconds per century for an inclined orbit like Mercury's.)
    """
    h_hat = h / np.linalg.norm(h, axis=1)[:, None]
    node = np.column_stack([-h_hat[:, 1], h_hat[:, 0], np.zeros(len(h))])  # z x h, points to the ascending node
    length = np.linalg.norm(node, axis=1)
    flat = length < 1e-12  # orbit in the ecliptic plane: the node is undefined, so measure from +x
    node[flat] = [1.0, 0.0, 0.0]
    node[~flat] /= length[~flat, None]
    e_hat = e_vec / np.linalg.norm(e_vec, axis=1)[:, None]
    big_omega = np.arctan2(node[:, 1], node[:, 0])
    omega = np.arctan2(np.einsum("sk,sk->s", np.cross(node, e_hat), h_hat), np.einsum("sk,sk->s", node, e_hat))
    return np.unwrap(big_omega) + np.unwrap(omega)


def perihelion_precession(t, varpi):
    """Rate of advance of the perihelion, in arcseconds per century, and its standard error,
    from a straight-line fit to the longitude of perihelion varpi(t) (radians)."""
    design = np.column_stack([t, np.ones_like(t)])
    beta, *_ = np.linalg.lstsq(design, varpi, rcond=None)
    resid = varpi - design @ beta
    cov = (resid @ resid / (len(t) - 2)) * np.linalg.inv(design.T @ design)
    scale = CENTURY / ARCSEC
    return float(beta[0] * scale), float(np.sqrt(cov[0, 0]) * scale)
