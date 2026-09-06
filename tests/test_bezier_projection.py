#!/usr/bin/env python3
"""The vectorized point-to-cubic projection must equal the scalar definition.

``tools/tracing/max_error.py`` keeps two implementations of one thing: the
readable per-point routines (``closest_point_on_cubic`` and friends) and the
array versions the fitter actually calls, which exist only because the scalar
ones were 75% of a corpus build. Their arithmetic is written to agree bit for
bit, which is a property nothing else checks and which an innocent-looking
edit to either side would quietly break -- the fit would still produce curves,
just slightly different ones, and only a diff of two full corpus builds would
show it.

Usage::

    .venv/bin/python tests/test_bezier_projection.py
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "tracing"))

import max_error as me  # noqa: E402


def fail(message: str) -> None:
    raise SystemExit(f"bezier projection regression failed: {message}")


def scalar_max_ortho(pts, u, curve, seed_samples, iters):
    max_err, split = -1.0, len(pts) // 2
    for i, t0 in enumerate(u):
        _, _, err = me.closest_point_on_cubic(
            curve, pts[i], seed_t=t0, seed_samples=seed_samples,
            newton_iters=iters)
        if err > max_err:
            max_err, split = err, i
    return max_err, split


def scalar_refined(pts, u, curve, seed_samples, iters):
    split = len(pts) // 2
    u_ref, errs = list(u), [0.0] * len(pts)
    for i, t0 in enumerate(u):
        t_star, _, err = me.closest_point_on_cubic(
            curve, pts[i], seed_t=None if seed_samples > 0 else t0,
            seed_samples=seed_samples, newton_iters=iters)
        u_ref[i], errs[i] = t_star, err
    prev = 0.0
    for i in range(len(pts)):
        if u_ref[i] < prev:
            u_ref[i] = prev
            errs[i] = me.v_dist(me.bezier_q(curve, prev), pts[i])
        else:
            prev = u_ref[i]
    max_err = -1.0
    for i, err in enumerate(errs):
        if err > max_err:
            max_err, split = err, i
    return max_err, split, u_ref


def scalar_errors(pts, u, curve, seed_samples, iters):
    return [me.closest_point_on_cubic(
        curve, pts[i], seed_t=t0, seed_samples=seed_samples,
        newton_iters=iters)[2] for i, t0 in enumerate(u)]


# (seed_samples, newton_iters). The first two are what bezier_fit.py actually
# uses -- a sampled seed on the opening iteration, then the previous
# parameters -- and the rest guard the other entry points.
SETTINGS = ((33, 2), (0, 2), (0, 8), (33, 8))


def check_equivalence(cases: int) -> None:
    random.seed(20)
    compared = 0
    for _ in range(cases):
        count = random.randint(2, 90)
        curve = tuple((random.uniform(-400, 400), random.uniform(-400, 400))
                      for _ in range(4))
        pts = [(random.uniform(-400, 400), random.uniform(-400, 400))
               for _ in range(count)]
        u = sorted(random.uniform(0.0, 1.0) for _ in range(count))
        for seed_samples, iters in SETTINGS:
            compared += 1
            want = scalar_max_ortho(pts, u, curve, seed_samples, iters)
            got = me.compute_max_ortho_error(
                pts, u, curve, seed_samples=seed_samples, newton_iters=iters)
            if want != got:
                fail(f"compute_max_ortho_error {want} != {got}")

            want = scalar_refined(pts, u, curve, seed_samples, iters)
            got = me.compute_max_ortho_error_and_refined_u(
                pts, u, curve, seed_samples=seed_samples, newton_iters=iters)
            if want[:2] != got[:2]:
                fail(f"refined error/split {want[:2]} != {got[:2]}")
            if want[2] != got[2]:
                index = next(i for i, (x, y) in enumerate(zip(want[2], got[2]))
                             if x != y)
                fail(f"refined parameter {index}: {want[2][index]!r} "
                     f"!= {got[2][index]!r}")

            want = scalar_errors(pts, u, curve, seed_samples, iters)
            got = me.compute_ortho_errors(
                pts, u, curve, seed_samples=seed_samples, newton_iters=iters)
            if want != got:
                index = next(i for i, (x, y) in enumerate(zip(want, got))
                             if x != y)
                fail(f"ortho error {index}: {want[index]!r} != {got[index]!r}")
    print(f"  {compared} comparisons agree exactly, across "
          f"{len(SETTINGS)} seed/iteration settings")


def check_degenerate() -> None:
    """Empty and single-point inputs kept their old return shapes."""
    curve = ((0.0, 0.0), (1.0, 2.0), (3.0, 2.0), (4.0, 0.0))
    if me.compute_max_ortho_error([], [], curve) != (-1.0, 0):
        fail("empty input no longer returns (-1.0, 0)")
    if me.compute_ortho_errors([], [], curve) != []:
        fail("empty input no longer returns an empty error list")
    error, split, params = me.compute_max_ortho_error_and_refined_u(
        [(2.0, 1.0)], [0.5], curve)
    if split != 0 or len(params) != 1 or not (error >= 0.0):
        fail(f"single point gave {(error, split, params)}")
    print("  empty and single-point inputs behave as before")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=400,
                        help="random curve/point sets to compare")
    args = parser.parse_args()
    print("bezier projection regression")
    check_equivalence(args.cases)
    check_degenerate()
    print("bezier projection regression passed")


if __name__ == "__main__":
    main()
