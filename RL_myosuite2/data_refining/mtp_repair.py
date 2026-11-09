"""Conservative, auditable repairs of *periodic* MTP angle/velocity samples.

V3 retains the V2 spike detector and local Hermite bridge, but NEVER changes
an original cycle-end angle and NEVER accepts a repair exceeding user-defined
limits on per-sample angle change or duration.  Failed repairs are returned
UNCHANGED and explicitly marked as rejected, so they cannot silently enter
training. Source NPZ files are never edited by this module.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class MTPRepairConfig:
    spike_threshold_rad_s: float = 14.0
    target_peak_rad_s: float = 10.0
    initial_margin_frames: int = 8
    max_margin_frames: int = 24
    expansion_frames: int = 2
    max_angle_change_rad: float = 0.20
    max_interval_fraction: float = 0.15

    def __post_init__(self):
        if not (0 < self.target_peak_rad_s < self.spike_threshold_rad_s):
            raise ValueError('Require 0 < target_peak < spike_threshold')
        if not (2 <= self.initial_margin_frames <= self.max_margin_frames):
            raise ValueError('Invalid initial/max margin')
        if self.expansion_frames < 1:
            raise ValueError('expansion_frames must be positive')
        if self.max_angle_change_rad <= 0:
            raise ValueError('max_angle_change_rad must be positive')
        if not (0 < self.max_interval_fraction < 1):
            raise ValueError('max_interval_fraction must lie between 0 and 1')


def periodic_derivative(q: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Nonuniform central derivative of a *scalar* with a duplicated cycle endpoint."""
    q = np.asarray(q, dtype=float)
    t = np.asarray(time, dtype=float)
    if q.ndim != 1 or t.ndim != 1 or len(q) != len(t) or len(t) < 8:
        raise ValueError('Expected matching scalar position/time arrays, N>=8')
    if t[0] != 0 or t[-1] <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError('Invalid time array')
    if abs(q[0] - q[-1]) > 1e-4:
        raise ValueError('MTP angle does not close periodically')
    m = len(q) - 1
    T = float(t[-1])
    out = np.empty(len(q))
    for j in range(m):
        jm = (j - 1) % m
        jp = (j + 1) % m
        before_t = t[jm] - (T if j == 0 else 0.0)
        after_t = t[jp] + (T if j == m-1 else 0.0)
        out[j] = (q[jp]-q[jm])/(after_t-before_t)
    out[-1] = out[0]
    return out


def _runs_circular(mask: np.ndarray) -> list[list[int]]:
    """Return connected circular components of a boolean mask, in cyclic order."""
    mask = np.asarray(mask, dtype=bool)
    m = len(mask)
    if not mask.any():
        return []
    if mask.all():
        raise ValueError('Repair region covers the entire gait cycle')
    # Begin just after a False, so the wraparound component is not split.
    boundary = int(np.flatnonzero(~mask)[0])
    order = (boundary + 1 + np.arange(m)) % m
    result, current = [], []
    for i in order:
        if mask[i]:
            current.append(int(i))
        elif current:
            result.append(current)
            current = []
    if current:
        result.append(current)
    return result


def _circular_time(j: int, t: np.ndarray, m: int) -> float:
    return float(t[j % m] + (j // m) * t[-1])


def _slope(q: np.ndarray, t: np.ndarray, anchor: int, direction: int, points: int = 3) -> float:
    m = len(q)
    vals = []
    for k in range(points):
        a = anchor + direction*k
        b = a + direction
        xa, xb = _circular_time(a, t, m), _circular_time(b, t, m)
        vals.append((q[b % m]-q[a % m])/(xb-xa))
    return float(np.median(vals))


def _smooth_mask(q: np.ndarray, t: np.ndarray, mask: np.ndarray):
    """Interpolate inside circular masked runs; end anchors remain unchanged."""
    m = len(q)
    out = q.copy()
    intervals = []
    for run in _runs_circular(mask):
        start = run[0]
        # Unroll the circular component into a strictly increasing integer index.
        unrolled = [start]
        for j in run[1:]:
            unrolled.append(unrolled[-1] + ((j - unrolled[-1]) % m))
        left, right = unrolled[0]-1, unrolled[-1]+1
        t0, t1 = _circular_time(left,t,m),_circular_time(right,t,m)
        y0,y1 = float(q[left%m]),float(q[right%m])
        d0 = _slope(q,t,left,-1)
        d1 = _slope(q,t,right,+1)
        # Endpoint slopes should not introduce artificial overshoots or reverse
        # the sign of an otherwise monotone sharp jump.
        secant = (y1-y0)/(t1-t0)
        if abs(secant) > 1e-10:
            d0 = np.clip(d0, 0, 3*secant) if secant > 0 else np.clip(d0, 3*secant, 0)
            d1 = np.clip(d1, 0, 3*secant) if secant > 0 else np.clip(d1, 3*secant, 0)
        for j in unrolled:
            s = (_circular_time(j,t,m)-t0)/(t1-t0)
            out[j%m] = ((2*s**3-3*s**2+1)*y0 + (s**3-2*s**2+s)*(t1-t0)*d0
                        + (-2*s**3+3*s**2)*y1 + (s**3-s**2)*(t1-t0)*d1)
        intervals.append(dict(start_frame=int(unrolled[0]%m),
                              end_frame=int(unrolled[-1]%m),
                              wraps_cycle=bool(unrolled[-1]>=m),
                              frame_count=len(unrolled),
                              start_phase=float((t[unrolled[0]%m]/t[-1])),
                              end_phase=float((t[unrolled[-1]%m]/t[-1]))))
    return out, intervals


def _unmodified_report(orig_q: np.ndarray, orig_v: np.ndarray,
                       orig_fd: np.ndarray, trigger_frames: int = 0,
                       rejected: bool = False,
                       rejection_reasons: list[str] | None = None,
                       attempts: list[dict] | None = None) -> dict:
    """Report only changes ACTUALLY applied; proposed repairs live in attempts."""
    return dict(
        trigger_frames=int(trigger_frames), modified=False, rejected=bool(rejected),
        rejection_reasons=list(rejection_reasons or []),
        attempted_candidates=list(attempts or []), intervals=[],
        max_abs_angle_change_rad=0.0,
        max_abs_velocity_before_rad_s=float(np.max(np.abs(orig_v))),
        max_abs_velocity_after_rad_s=float(np.max(np.abs(orig_v))),
        max_abs_central_derivative_before_rad_s=float(np.max(np.abs(orig_fd))),
        max_abs_central_derivative_after_rad_s=float(np.max(np.abs(orig_fd))),
        modified_qpos_rows=0, modified_qvel_rows=0,
        achieved_target=not rejected, margin_frames=0)


def repair_mtp(qpos: np.ndarray, qvel: np.ndarray, time: np.ndarray,
               config: MTPRepairConfig = MTPRepairConfig()):
    """Attempt a bounded repair of ONE scalar MTP joint on a closed gait cycle.

    Returns qpos, qvel, report, affected_rows. Rejected proposals NEVER alter
    returned arrays. Report `rejected=True` must prevent use of this reference
    in a generated training dataset unless explicitly reviewed/resolved.
    """
    orig_q = np.asarray(qpos, dtype=np.float64)
    orig_v = np.asarray(qvel, dtype=np.float64)
    t = np.asarray(time, dtype=np.float64)
    if orig_q.shape != orig_v.shape or orig_q.shape != t.shape:
        raise ValueError('MTP position, velocity and time shapes differ')
    m = len(t) - 1
    if m < 20:
        raise ValueError('Expected at least 20 unique gait samples')
    if not np.isfinite(orig_q).all() or not np.isfinite(orig_v).all():
        raise ValueError('Nonfinite MTP data')
    original_fd = periodic_derivative(orig_q, t)
    triggers = ((np.abs(orig_v[:m]) > config.spike_threshold_rad_s)
                | (np.abs(original_fd[:m]) > config.spike_threshold_rad_s))
    if abs(orig_v[-1]) > config.spike_threshold_rad_s:
        triggers[0] = True
    ntriggers = int(np.count_nonzero(triggers))
    if not ntriggers:
        return (orig_q.copy(), orig_v.copy(),
                _unmodified_report(orig_q, orig_v, original_fd),
                np.zeros(len(t), dtype=bool))

    # We cannot guarantee both original endpoint angles AND a periodic result
    # if the original reference is itself not closed to numeric tolerance.
    if not np.isclose(orig_q[0], orig_q[-1], atol=1e-8, rtol=0):
        return (orig_q.copy(), orig_v.copy(),
                _unmodified_report(orig_q, orig_v, original_fd, ntriggers,
                                   True, ['source_endpoint_mismatch']),
                np.zeros(len(t), dtype=bool))

    attempts = []
    # Try the smallest radius first: the first candidate satisfying *every*
    # constraint is retained. No unconstrained fallback to a huge bridge.
    for radius in range(config.initial_margin_frames,
                        config.max_margin_frames + 1, config.expansion_frames):
        mask = np.zeros(m, dtype=bool)
        for offset in range(-radius, radius + 1):
            mask |= np.roll(triggers, offset)
        # HARD ANCHOR: cycle-end angle is never edited, even for wrap repairs.
        mask[0] = False
        if mask.all():
            attempts.append(dict(radius=radius, accepted=False,
                                 failures=['whole_cycle_mask']))
            continue
        qfix, regions = _smooth_mask(orig_q[:m], t, mask)
        qnew = np.r_[qfix, orig_q[-1]]
        # Respect both original boundary angles, not merely q[0] == q[-1].
        assert qnew[0] == orig_q[0] and qnew[-1] == orig_q[-1]
        fdnew = periodic_derivative(qnew, t)

        # Recompute only velocities influenced by the qpos edit (plus its
        # neighbors); preserve velocities elsewhere.
        recompute = mask | np.roll(mask, 1) | np.roll(mask, -1)
        vnew = orig_v.copy()
        vnew[:m][recompute] = fdnew[:m][recompute]
        vnew[-1] = vnew[0]  # period closure (velocity may need seam repair)
        modified_vel = np.r_[recompute, recompute[0]]
        changed_q = np.abs(qnew[:m] - orig_q[:m]) > 1e-12
        max_delta = float(np.max(np.abs(qnew - orig_q)))
        longest_fraction = max((r['frame_count'] / m for r in regions), default=0.)
        peak_velocity = float(np.max(np.abs(vnew[:m][recompute]))) if recompute.any() else 0.
        failures = []
        if max_delta > config.max_angle_change_rad + 1e-12:
            failures.append('max_angle_change_exceeded')
        if longest_fraction > config.max_interval_fraction + 1e-12:
            failures.append('max_interval_fraction_exceeded')
        if peak_velocity > config.target_peak_rad_s + 1e-8:
            failures.append('target_velocity_not_reached')
        # The endpoint cannot be a hidden outlier just because it is a hard
        # anchor. Record an unresolved seam velocity if it remains too large.
        if max(abs(vnew[0]), abs(vnew[-1])) > config.spike_threshold_rad_s + 1e-8:
            failures.append('unresolved_endpoint_velocity')
        attempts.append(dict(radius=int(radius), accepted=not failures,
                             failures=failures, max_abs_angle_change_rad=max_delta,
                             max_interval_fraction=float(longest_fraction),
                             max_abs_recomputed_velocity_rad_s=peak_velocity,
                             intervals=regions))
        if failures:
            continue

        modified_q = np.r_[changed_q, changed_q[0]]
        # The recompute window can include identical velocities; report the
        # rows whose numeric values actually changed, not merely visited.
        changed_v = np.abs(vnew - orig_v) > 1e-12
        report = dict(trigger_frames=ntriggers, modified=True,
                      rejected=False, rejection_reasons=[],
                      attempted_candidates=attempts, intervals=regions,
                      modified_qpos_rows=int(np.count_nonzero(modified_q)),
                      modified_qvel_rows=int(np.count_nonzero(changed_v)),
                      max_abs_angle_change_rad=max_delta,
                      max_abs_velocity_before_rad_s=float(np.max(np.abs(orig_v))),
                      max_abs_velocity_after_rad_s=float(np.max(np.abs(vnew))),
                      max_abs_central_derivative_before_rad_s=float(np.max(np.abs(original_fd))),
                      max_abs_central_derivative_after_rad_s=float(np.max(np.abs(fdnew))),
                      margin_frames=int(radius), achieved_target=True)
        return qnew, vnew, report, modified_q | changed_v

    # No candidate was sufficiently conservative. Never silently keep the
    # over-smoothed version, and never hide a bad original as 'unchanged'.
    reasons = sorted({reason for attempt in attempts
                      for reason in attempt.get('failures', [])})
    return (orig_q.copy(), orig_v.copy(),
            _unmodified_report(orig_q, orig_v, original_fd, ntriggers,
                               True, reasons or ['no_acceptable_candidate'], attempts),
            np.zeros(len(t), dtype=bool))
