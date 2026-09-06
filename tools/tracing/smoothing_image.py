from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import cv2

Point = Tuple[float, float]


# --------------------------- Utilities: sampling SDF ---------------------------

def _bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Bilinear sampling of single-channel float image at (x,y).
    x,y are float arrays in image coordinates (col=x, row=y).
    """
    h, w = img.shape[:2]

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, w - 1); x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1); y1 = np.clip(y1, 0, h - 1)

    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def _sdf_and_grad(phi: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample signed distance phi and its gradient at pts.
    pts: (N,2) float in (x,y).
    Returns:
      val: (N,)
      grad: (N,2) in (dphi/dx, dphi/dy)
    """
    x = pts[:, 0]
    y = pts[:, 1]

    # sample phi
    val = _bilinear_sample(phi, x, y)

    # compute gradients with Sobel once (outside loop ideally); here done inside helper for clarity
    # Better: precompute gx, gy once and pass them.
    raise RuntimeError("Use _sdf_and_grad_precomputed for performance.")


def _sdf_and_grad_precomputed(phi: np.ndarray, gx: np.ndarray, gy: np.ndarray, pts: np.ndarray):
    x = pts[:, 0]
    y = pts[:, 1]
    val = _bilinear_sample(phi, x, y)
    gxx = _bilinear_sample(gx, x, y)
    gyy = _bilinear_sample(gy, x, y)
    grad = np.stack([gxx, gyy], axis=1)
    return val, grad


# --------------------------- Polyline processing ---------------------------

def _resample_closed_polyline(pts: np.ndarray, step: float) -> np.ndarray:
    """Uniform arc-length resampling for closed polyline."""
    if len(pts) < 4:
        return pts

    dif = np.roll(pts, -1, axis=0) - pts
    seg = np.sqrt((dif ** 2).sum(axis=1))
    L = float(seg.sum())
    if L <= 1e-6:
        return pts

    m = max(64, int(np.ceil(L / max(step, 1e-3))))
    t = np.linspace(0.0, L, num=m, endpoint=False)

    s = np.concatenate([[0.0], np.cumsum(seg)])
    # enforce strict monotonicity
    eps = 1e-6
    for i in range(1, len(s)):
        if s[i] <= s[i - 1]:
            s[i] = s[i - 1] + eps

    p2 = np.vstack([pts, pts[:1]])
    out = np.zeros((m, 2), dtype=np.float32)

    j = 0
    for i, ti in enumerate(t):
        while j + 1 < len(s) - 1 and s[j + 1] <= ti:
            j += 1
        a = s[j]
        b = s[j + 1]
        u = (ti - a) / (b - a)
        out[i] = (1.0 - u) * p2[j] + u * p2[j + 1]
    return out


def _curvature_estimate(pts: np.ndarray) -> np.ndarray:
    """
    Simple discrete curvature magnitude estimate per vertex for closed polyline.
    Returns (N,) curvature-ish values (not exact κ, but good for corner weights).
    """
    p_prev = np.roll(pts, 1, axis=0)
    p = pts
    p_next = np.roll(pts, -1, axis=0)

    v1 = p - p_prev
    v2 = p_next - p
    n1 = np.linalg.norm(v1, axis=1) + 1e-8
    n2 = np.linalg.norm(v2, axis=1) + 1e-8

    # angle between v1 and v2
    dot = (v1 * v2).sum(axis=1) / (n1 * n2)
    dot = np.clip(dot, -1.0, 1.0)
    ang = np.arccos(dot)  # [0..pi]
    # normalize by local length scale
    return ang / (0.5 * (n1 + n2))


def _fair_step(pts: np.ndarray, alpha: float, corner_k: float) -> np.ndarray:
    """
    One fairing (smoothing) step with corner preservation.
    alpha: smoothing strength (0..1 small)
    corner_k: controls how strongly corners are protected (bigger -> protect more)
    """
    p_prev = np.roll(pts, 1, axis=0)
    p_next = np.roll(pts, -1, axis=0)
    lap = (p_prev + p_next) * 0.5 - pts  # umbrella Laplacian

    # corner preservation: reduce smoothing where curvature is high
    kappa = _curvature_estimate(pts)
    w = 1.0 / (1.0 + corner_k * (kappa ** 2))  # (N,)
    w = w.astype(np.float32)[:, None]

    return pts + alpha * w * lap


def _project_to_zero_level(
    pts: np.ndarray,
    phi: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    newton_steps: int = 2,
    clamp_step: float = 1.5
) -> np.ndarray:
    """
    Project points onto φ=0 using a few Newton iterations:
      p <- p - φ(p) * ∇φ(p) / (||∇φ||^2)
    clamp_step limits overshoot per iteration (pixels).
    """
    h, w = phi.shape[:2]
    p = pts.copy()

    for _ in range(newton_steps):
        val, grad = _sdf_and_grad_precomputed(phi, gx, gy, p)
        g2 = (grad[:, 0] ** 2 + grad[:, 1] ** 2) + 1e-8
        step = (val / g2).astype(np.float32)  # (N,)

        dp = grad * step[:, None]  # (N,2)

        # Clamp step magnitude to avoid jumping across thin strokes / noisy gradients
        mag = np.linalg.norm(dp, axis=1) + 1e-8
        scale = np.minimum(1.0, clamp_step / mag).astype(np.float32)
        dp = dp * scale[:, None]

        p = p - dp

        # keep inside image
        p[:, 0] = np.clip(p[:, 0], 0, w - 1)
        p[:, 1] = np.clip(p[:, 1], 0, h - 1)

    return p


def resample_polyline(pts: np.ndarray, step: float, closed: bool) -> np.ndarray:
    """
    Uniform arc-length resampling of a polyline.
    pts: (N,2) float
    step: target spacing in pixels
    closed: treat as closed loop
    """
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 2:
        return pts

    if closed:
        dif = np.roll(pts, -1, axis=0) - pts
        seg = np.sqrt((dif ** 2).sum(axis=1))
        L = float(seg.sum())
        if L <= 1e-6:
            return pts
        m = max(16, int(np.ceil(L / max(step, 1e-6))))
        t = np.linspace(0.0, L, num=m, endpoint=False)

        s = np.concatenate([[0.0], np.cumsum(seg)])  # length N+1
        # Ensure strictly increasing
        eps = 1e-6
        for i in range(1, len(s)):
            if s[i] <= s[i - 1]:
                s[i] = s[i - 1] + eps

        p2 = np.vstack([pts, pts[:1]])
        out = np.zeros((m, 2), dtype=np.float32)
        j = 0
        for i, ti in enumerate(t):
            while j + 1 < len(s) - 1 and s[j + 1] <= ti:
                j += 1
            a, b = s[j], s[j + 1]
            u = (ti - a) / (b - a)
            out[i] = (1.0 - u) * p2[j] + u * p2[j + 1]
        return out

    else:
        dif = pts[1:] - pts[:-1]
        seg = np.sqrt((dif ** 2).sum(axis=1))
        L = float(seg.sum())
        if L <= 1e-6:
            return pts

        m = max(2, int(np.ceil(L / max(step, 1e-6))) + 1)
        t = np.linspace(0.0, L, num=m)

        s = np.concatenate([[0.0], np.cumsum(seg)])  # length N
        eps = 1e-6
        for i in range(1, len(s)):
            if s[i] <= s[i - 1]:
                s[i] = s[i - 1] + eps

        out = np.zeros((m, 2), dtype=np.float32)
        j = 0
        for i, ti in enumerate(t):
            while j + 1 < len(s) and s[j + 1] <= ti:
                j += 1
            if j == len(seg):
                out[i] = pts[-1]
            else:
                a, b = s[j], s[j + 1]
                u = (ti - a) / (b - a)
                out[i] = (1.0 - u) * pts[j] + u * pts[j + 1]
        return out


def gaussian_kernel_1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    """
    1D Gaussian kernel normalized to sum=1.
    radius = ceil(truncate*sigma)
    """
    sigma = float(sigma)
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = int(np.ceil(truncate * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k


def convolve_1d_reflect(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve 1D signal with reflect padding (for open curves).
    """
    r = len(kernel) // 2
    if r == 0:
        return signal.copy()
    padded = np.pad(signal, (r, r), mode="reflect")
    out = np.convolve(padded, kernel, mode="valid")
    return out.astype(np.float32)


def convolve_1d_circular(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve 1D signal with circular wrap-around (for closed curves).
    """
    r = len(kernel) // 2
    if r == 0:
        return signal.copy()
    padded = np.concatenate([signal[-r:], signal, signal[:r]])
    out = np.convolve(padded, kernel, mode="valid")
    return out.astype(np.float32)


def curve_domain_gaussian_smooth(
    points: List[Point],
    *,
    sigma_px: float = 0.9,
    resample_step_px: float = 1.0,
    closed: bool = True,
    truncate: float = 3.0,
) -> List[Point]:
    """
    Curve-domain Gaussian smoothing:
      1) resample polyline to uniform spacing (resample_step_px)
      2) Gaussian smooth x[i], y[i] along index (arc-length)
    sigma_px is expressed in *pixels along the curve*, assuming uniform spacing.

    Returns smoothed points (same spacing).
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 4:
        return [(float(x), float(y)) for x, y in pts]

    # 1) uniform resample
    pts_u = resample_polyline(pts, step=resample_step_px, closed=closed)

    # 2) build kernel
    # sigma in "samples" is sigma_px / step_px
    sigma_samples = float(sigma_px) / max(float(resample_step_px), 1e-6)
    k = gaussian_kernel_1d(sigma_samples, truncate=truncate)

    x = pts_u[:, 0]
    y = pts_u[:, 1]

    if closed:
        xs = convolve_1d_circular(x, k)
        ys = convolve_1d_circular(y, k)
    else:
        xs = convolve_1d_reflect(x, k)
        ys = convolve_1d_reflect(y, k)

    out = np.stack([xs, ys], axis=1)
    return [(float(a), float(b)) for a, b in out]


# --------------------------- Main pipeline ---------------------------

def fair_outer_contour_close_to_ink(
    gray_4800: np.ndarray,
    *,
    downscale: int = 4,                  # 4800 -> 1200
    sigma_4800: float = 1.5,             # preblur in 4800-space
    binarize: str = "otsu",              # "otsu" or "adaptive"
    morph_close_1200: int = 1,
    min_area_1200: int = 200,
    step_px_1200: float = 1.0,           # resampling spacing (1200-space)
    iters: int = 100,                     # fair+project iterations
    alpha: float = 0.5,                 # fairing strength per iter (0.1..0.5)
    corner_k: float = 8.0,               # bigger => preserve corners more
    newton_steps: int = 2,
) -> List[Point]:
    """
    Input: grayscale image at ~4800 DPI (uint8).
    Output: fair outer contour points in 1200-DPI space, close to ink boundary.
    """
    if gray_4800.dtype != np.uint8:
        gray_4800 = np.clip(gray_4800, 0, 255).astype(np.uint8)

    # --- Preblur (anti-edge-buzz) in 4800-space ---
    g_blur = cv2.GaussianBlur(gray_4800, (0, 0), sigmaX=sigma_4800, sigmaY=sigma_4800)

    # --- Downscale x4 using AREA ---
    h0, w0 = g_blur.shape[:2]
    g = cv2.resize(g_blur, (w0 // downscale, h0 // downscale), interpolation=cv2.INTER_AREA)

    # --- Binarize to get ink mask (ink=255) ---
    if binarize.lower() == "otsu":
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif binarize.lower() == "adaptive":
        block = 251  # good for ~22-26 px stroke at 1200
        c = 7
        bw = cv2.adaptiveThreshold(
            g, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block, C=c
        )
    else:
        raise ValueError("binarize must be 'otsu' or 'adaptive'")

    # --- Morph close to bridge tiny holes/gaps (1200-space) ---
    if morph_close_1200 > 0:
        k = 2 * morph_close_1200 + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, se, iterations=1)

    # --- Find outer contour (largest component) ---
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    contours = [c for c in contours if cv2.contourArea(c) >= float(min_area_1200)]
    if not contours:
        return []

    c = max(contours, key=cv2.contourArea)
    pts = c.reshape(-1, 2).astype(np.float32)
    

    # Ensure we start from a reasonably uniform sampling
    pts = _resample_closed_polyline(pts, step=step_px_1200)

    # --- Build signed distance field φ ---
    # ink mask: bw>0 is inside
    ink = (bw > 0).astype(np.uint8)

    # distance to background from inside ink
    dist_in = cv2.distanceTransform(ink, distanceType=cv2.DIST_L2, maskSize=5)
    # distance to ink from outside
    dist_out = cv2.distanceTransform(1 - ink, distanceType=cv2.DIST_L2, maskSize=5)

    # signed distance: negative inside ink, positive outside
    phi = dist_out - dist_in
    phi = phi.astype(np.float32)

    # Gradient of SDF (Sobel)
    gx = cv2.Sobel(phi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(phi, cv2.CV_32F, 0, 1, ksize=3)

    # --- Iterate: fair then project to φ=0 ---
    for _ in range(max(1, iters)):
        pts = _fair_step(pts, alpha=alpha, corner_k=corner_k)
        pts = _project_to_zero_level(
            pts, phi, gx, gy, newton_steps=newton_steps, clamp_step=1.5
        )
        pts = _resample_closed_polyline(pts, step=step_px_1200)

    return [(float(x), float(-y)) for x, y in pts]
