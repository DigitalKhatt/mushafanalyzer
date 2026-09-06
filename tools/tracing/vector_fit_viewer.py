import math
import os

os.environ.setdefault("QT_API", "pyside6")

import sys

import cv2
import matplotlib as mpl

mpl.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from preprocess_image import (
    crop_to_ink,
    read_bw_white_is_ink,
)
from png_to_ttf_no_potrace import (
    contour_to_beziers,
    find_contours_with_hierarchy,
    skip_tiny_hole_subtrees,
)
from bezier_fit import FitConfig

MIN_HOLE_AREA_PX = 500.0


DEFAULT_PATH_CANDIDATES = [
    "output/oldmedina2-analysis/page003/wordimgs/l10_w09.png",
    "input/test.png",
    "input/test2.png",
]

ANALYSIS_ROOT = "output/oldmedina2-analysis"


def load_initial_image():
  for path in DEFAULT_PATH_CANDIDATES:
    if os.path.isfile(path):
      return path, pad(crop_to_ink(read_bw_white_is_ink(path)), 24)
  return None, None


def load_image(path):
  return pad(crop_to_ink(read_bw_white_is_ink(path)), 24)


def bez3(p0, p1, p2, p3, t):
  u = 1.0 - t
  b0 = u * u * u
  b1 = 3 * u * u * t
  b2 = 3 * u * t * t
  b3 = t * t * t
  return (
      p0[0] * b0 + p1[0] * b1 + p2[0] * b2 + p3[0] * b3,
      p0[1] * b0 + p1[1] * b1 + p2[1] * b2 + p3[1] * b3,
  )


def sample_cubic(curve, n=40):
  p0, p1, p2, p3 = curve
  ts = np.linspace(0, 1, n)
  pts = [bez3(p0, p1, p2, p3, float(t)) for t in ts]
  return np.array(pts)


def pad(bw, p=24):
  return cv2.copyMakeBorder(bw, p, p, p, p, cv2.BORDER_CONSTANT, value=0)


class _NoScrollSpinBox(QDoubleSpinBox):
  """QDoubleSpinBox that only reacts to the wheel when it has focus.

  Without this, scrolling over the controls panel silently changes a
  parameter and triggers a full re-fit.
  """

  def wheelEvent(self, event):
    if self.hasFocus():
      super().wheelEvent(event)
    else:
      event.ignore()


class FloatParam(QWidget):
  """A labelled QDoubleSpinBox that reports float values."""

  valueChanged = Signal(float)

  def __init__(self, name, minimum, maximum, init, step=0.1, parent=None):
    super().__init__(parent)

    layout = QHBoxLayout(self)
    layout.setContentsMargins(0, 0, 0, 0)

    name_label = QLabel(name)
    name_label.setFixedWidth(105)

    self.spin = _NoScrollSpinBox()
    self.spin.setDecimals(0 if step >= 1.0 else max(0, -round(math.log10(step))))
    self.spin.setRange(minimum, maximum)
    self.spin.setSingleStep(step)
    self.spin.setValue(init)
    self.spin.setFocusPolicy(Qt.StrongFocus)
    # Emit once the edit is committed (Enter / focus-out / arrows) instead of
    # on every keystroke, so typing "9.5" doesn't re-fit for "9" first.
    self.spin.setKeyboardTracking(False)
    self.spin.valueChanged.connect(self.valueChanged)

    layout.addWidget(name_label)
    layout.addWidget(self.spin, 1)

  def setToolTip(self, text):
    super().setToolTip(text)
    self.spin.setToolTip(text)

  def value(self):
    return self.spin.value()


class VectorFitViewer(QMainWindow):
  def __init__(self):
    super().__init__()
    self.setWindowTitle("Vector Fit Viewer")
    self.resize(1400, 900)
    self.setWindowState(Qt.WindowMaximized)

    self.path, self.bw0 = load_initial_image()
    self._has_content = False

    self.figure = Figure()
    self.canvas = FigureCanvasQTAgg(self.figure)
    self.toolbar = NavigationToolbar2QT(self.canvas, self)
    self.toolbar.zoom()  # start with "zoom to rectangle" active
    self.ax = self.figure.add_subplot(111)

    central = QWidget()
    self.setCentralWidget(central)
    root_layout = QHBoxLayout(central)

    self.nav_tree = QTreeWidget()
    self.nav_tree.setHeaderHidden(True)
    self.nav_tree.setFixedWidth(260)
    self.nav_tree.itemClicked.connect(self.on_nav_item_clicked)
    self.populate_nav_tree()
    root_layout.addWidget(self.nav_tree)

    canvas_layout = QVBoxLayout()
    canvas_layout.addWidget(self.toolbar)
    canvas_layout.addWidget(self.canvas, 1)
    root_layout.addLayout(canvas_layout, 1)

    controls = QWidget()
    controls.setFixedWidth(280)
    controls_layout = QVBoxLayout(controls)

    self.open_btn = QPushButton("Open image")
    self.open_btn.clicked.connect(self.on_open)
    controls_layout.addWidget(self.open_btn)

    self.s_scale = FloatParam("scale", 0.01, 1.0, 1.0, step=0.01)
    self.s_sigma = FloatParam("blur σ", 0.0, 50.0, 0.0, step=0.1)
    self.s_step = FloatParam("step px", 0.0, 50.0, 0.0, step=0.1)
    self.s_straight_seg_tol = FloatParam("straight_seg_tol", 0.0, 100.0, 9.0, step=0.1)
    self.s_err = FloatParam("fit err", 0.5, 100.0, 8.0, step=0.1)

    for s in (self.s_scale, self.s_sigma, self.s_step, self.s_straight_seg_tol, self.s_err):
      controls_layout.addWidget(s)
      s.valueChanged.connect(self.redraw)

    toggles_box = QGroupBox("Overlays")
    toggles_layout = QVBoxLayout(toggles_box)

    self.cb_points = QCheckBox("Points")
    self.cb_curve = QCheckBox("Curve")
    self.cb_handles = QCheckBox("Handles")
    self.cb_segments = QCheckBox("Segments")
    self.cb_extrema = QCheckBox("Extrema")
    self.cb_inflections = QCheckBox("Inflections")

    self.cb_curve.setChecked(True)
    self.cb_segments.setChecked(False)
    self.cb_extrema.setChecked(True)
    self.cb_inflections.setChecked(True)

    for cb in (
        self.cb_points,
        self.cb_curve,
        self.cb_handles,
        self.cb_segments,
        self.cb_extrema,
        self.cb_inflections,
    ):
      toggles_layout.addWidget(cb)
      cb.stateChanged.connect(self.redraw)

    controls_layout.addWidget(toggles_box)

    fit_box = QGroupBox("Fitting")
    fit_layout = QVBoxLayout(fit_box)
    self.cb_retry_without_inflection_check = QCheckBox("Retry split w/o inflection check")
    self.cb_retry_without_inflection_check.setChecked(False)
    self.cb_retry_without_inflection_check.stateChanged.connect(self.redraw)
    fit_layout.addWidget(self.cb_retry_without_inflection_check)

    self.cb_naive_split_only = QCheckBox("Naive split only")
    self.cb_naive_split_only.setChecked(False)
    self.cb_naive_split_only.stateChanged.connect(self.redraw)
    fit_layout.addWidget(self.cb_naive_split_only)

    self.cb_no_inflection_fit = QCheckBox("No-inflection fit at leaves/candidates")
    self.cb_no_inflection_fit.setChecked(False)
    self.cb_no_inflection_fit.setToolTip(
        "Sets FitConfig.use_no_inflection_fit: fits with "
        "fit_single_cubic_no_inflection instead of fit_single_cubic_2, "
        "falling back to fit_single_cubic_2 wherever the no-inflection "
        "guarantee doesn't apply (parallel/diverging tangents)."
    )
    self.cb_no_inflection_fit.stateChanged.connect(self.redraw)
    fit_layout.addWidget(self.cb_no_inflection_fit)

    self.s_max_splits = FloatParam("max splits", 0, 20, 2, step=1.0)
    self.s_max_splits.setToolTip(
        "FitConfig.max_splits: cap on how many times a single-cubic fit may "
        "recursively split to bring its error under 'fit err'. A leaf over "
        "tolerance at this depth is kept as-is rather than split further."
    )
    self.s_max_splits.valueChanged.connect(self.redraw)
    fit_layout.addWidget(self.s_max_splits)

    controls_layout.addWidget(fit_box)

    infl_box = QGroupBox("Inflection candidate gating")
    infl_layout = QVBoxLayout(infl_box)

    self.cb_infl_max_mid_dev = QCheckBox("Reject if mid segment not straight")
    self.cb_infl_max_mid_dev.setChecked(False)
    self.cb_infl_max_mid_dev.stateChanged.connect(self.redraw)
    infl_layout.addWidget(self.cb_infl_max_mid_dev)

    self.s_infl_max_mid_dev = FloatParam("max mid dev", 0.1, 20.0, 2.0, step=0.1)
    self.s_infl_max_mid_dev.setToolTip(
        "FitConfig.infl_max_mid_dev (px): reject an inflection candidate "
        "whose mid segment deviates from its own fitted line by more than "
        "this. Only applied when the checkbox above is checked."
    )
    self.s_infl_max_mid_dev.valueChanged.connect(self.redraw)
    infl_layout.addWidget(self.s_infl_max_mid_dev)

    self.s_infl_min_mid_len = FloatParam("min mid len", 0.0, 100.0, 0.0, step=1.0)
    self.s_infl_min_mid_len.setToolTip(
        "FitConfig.infl_min_mid_len (px): reject an inflection candidate "
        "whose mid segment is shorter than this."
    )
    self.s_infl_min_mid_len.valueChanged.connect(self.redraw)
    infl_layout.addWidget(self.s_infl_min_mid_len)

    controls_layout.addWidget(infl_box)

    controls_layout.addStretch(1)
    root_layout.addWidget(controls)

    self.redraw(keep_view=False)

  def populate_nav_tree(self):
    self.nav_tree.clear()
    if not os.path.isdir(ANALYSIS_ROOT):
      return
    for page_name in sorted(os.listdir(ANALYSIS_ROOT)):
      wordimgs_dir = os.path.join(ANALYSIS_ROOT, page_name, "wordimgs")
      if not os.path.isdir(wordimgs_dir):
        continue
      page_item = QTreeWidgetItem([page_name])
      self.nav_tree.addTopLevelItem(page_item)
      for fname in sorted(os.listdir(wordimgs_dir)):
        if not fname.lower().endswith(".png"):
          continue
        img_item = QTreeWidgetItem([fname])
        img_item.setData(0, Qt.UserRole, os.path.join(wordimgs_dir, fname))
        page_item.addChild(img_item)

  def on_nav_item_clicked(self, item, _column):
    path = item.data(0, Qt.UserRole)
    if not path:
      return  # a page folder was clicked, not an image
    self.path = path
    self.bw0 = load_image(path)
    self.redraw(keep_view=False)

  def on_open(self, _=None):
    p, _filter = QFileDialog.getOpenFileName(
        self, "Open image", "", "PNG (*.png);;All files (*.*)"
    )
    if not p:
      return
    self.path = p
    self.bw0 = load_image(p)
    self.redraw(keep_view=False)

  def redraw(self, *_, keep_view=True):
    """Re-run the fit and repaint.

    keep_view=True (the default, used by every parameter/overlay control)
    restores whatever zoom/pan the user had before repainting, so tweaking a
    parameter doesn't snap the view back out. keep_view=False is used when
    switching images, where starting fully zoomed out is what's wanted.
    """
    prev_xlim = prev_ylim = None
    if keep_view and self._has_content:
      prev_xlim = self.ax.get_xlim()
      prev_ylim = self.ax.get_ylim()

    self._render()

    # Sync draw: with set_aspect("equal") the final limits are only settled
    # at draw time, so the "home" view pushed below must be captured after
    # this, not before.
    self.canvas.draw()
    self.toolbar.update()       # drop the stale nav stack from the last redraw
    self.toolbar.push_current()  # entry 0 = full extent -> this is "Home"

    if prev_xlim is not None:
      self.ax.set_xlim(prev_xlim)
      self.ax.set_ylim(prev_ylim)
      self.toolbar.push_current()  # entry 1 = the view the user was on
      self.canvas.draw_idle()

  def _render(self):
    ax = self.ax
    self._has_content = False

    sigma = self.s_sigma.value()
    step = self.s_step.value()
    straight_seg_tol = self.s_straight_seg_tol.value()
    err = self.s_err.value()
    scale = self.s_scale.value()

    print_points = self.cb_points.isChecked()
    print_curve = self.cb_curve.isChecked()
    print_handles = self.cb_handles.isChecked()
    print_segments = self.cb_segments.isChecked()
    print_extrema = self.cb_extrema.isChecked()
    print_inflections = self.cb_inflections.isChecked()
    retry_without_inflection_check = self.cb_retry_without_inflection_check.isChecked()
    use_naive_split_only = self.cb_naive_split_only.isChecked()
    use_no_inflection_fit = self.cb_no_inflection_fit.isChecked()
    max_splits = int(self.s_max_splits.value())
    infl_max_mid_dev = (
        self.s_infl_max_mid_dev.value() if self.cb_infl_max_mid_dev.isChecked() else None
    )
    infl_min_mid_len = self.s_infl_min_mid_len.value()

    cfg = FitConfig(
        tol=err,
        reparam_iters=5,
        max_splits=max_splits,
        min_arc_height=6,
        straight_seg_tol=straight_seg_tol,
        retry_without_inflection_check=retry_without_inflection_check,
        use_naive_split_only=use_naive_split_only,
        scale=scale,
        blur=sigma,
        step=step,
        use_no_inflection_fit=use_no_inflection_fit,
        infl_max_mid_dev=infl_max_mid_dev,
        infl_min_mid_len=infl_min_mid_len,
    )

    ax.clear()
    ax.invert_yaxis()
    ax.set_title(
        "Zoom/pan with toolbar. Blue=curve, Red=control pts, Orange=corners, Gray=raw contour"
    )
    ax.set_aspect("equal")

    if self.bw0 is None:
      ax.text(0.5, 0.5, "No image loaded — click Open image",
               transform=ax.transAxes, ha="center")
      return

    nodes = find_contours_with_hierarchy(self.bw0)

    if not nodes:
      ax.text(0.5, 0.5, "No contour", transform=ax.transAxes, ha="center")
      return

    # Same filtering as png_to_ttf_no_potrace.py: drop sub-pixel-scale
    # holes (scan/threshold noise, not real counters) and degenerate
    # tiny contours.
    skip_idx = skip_tiny_hole_subtrees(nodes, MIN_HOLE_AREA_PX)

    for i, node in enumerate(nodes):
      if i in skip_idx or len(node.pts) < 12:
        continue
      # OpenCV y+ down; display with y inverted so "up" is up.
      # Outer contours wind CCW, holes wind CW (nonzero fill rule);
      # contour_to_beziers handles orientation/scale/blur/resample itself.
      bez = contour_to_beziers(pts_np=node.pts, is_hole=node.is_hole, cfg=cfg)
      cubics = bez.cubics
      resampled = bez.points

      resampled_np = np.array(resampled)

      if print_points:
        ax.plot(resampled_np[::1, 0], resampled_np[::1, 1],
                ".", markersize=1, alpha=0.35)

      # first point
      ax.annotate("", xytext=resampled[0], xy=(resampled[0][0] - 30, resampled[0][1]),
                  arrowprops=dict(arrowstyle="->"))

      if print_segments or print_extrema or print_inflections:
        # Same segs/infls/extrema the fitter itself used to produce `cubics`
        # (bez.segs/infls/extrema), not a separate recomputation -- so
        # overlays always match what was actually fit, including the
        # infl_min_mid_len/infl_max_mid_dev gates.
        segs, infls, extrema = bez.segs, bez.infls, bez.extrema

        if print_extrema:
          for idx, etype in extrema:
            # vertical tangent (xmin/xmax) = red, horizontal tangent (ymin/ymax) = blue
            color = 'red' if etype in ('xmin', 'xmax') else 'blue'
            ax.plot(resampled_np[idx, 0], resampled_np[idx, 1],
                    "o", markersize=3, alpha=1, color=color)

        if print_segments:
          for seg in segs:
            ax.plot([resampled_np[seg.i0, 0], resampled_np[seg.i1, 0]], [resampled_np[seg.i0, 1],
                    resampled_np[seg.i1, 1]], alpha=0.4, color='red', linestyle='-')

        if print_inflections:
          for inf in infls:
            seg = segs[inf.mid_idx]
            if print_segments:
              ax.plot([resampled_np[seg.i0, 0], resampled_np[seg.i1, 0]],
                      [resampled_np[seg.i0, 1],
                      resampled_np[seg.i1, 1]], alpha=0.7,
                      color='green', linestyle='-')
            ax.plot(resampled_np[seg.mid_index, 0], resampled_np[seg.mid_index, 1],
                    "o", markersize=3, alpha=0.35, color='green')

      if print_curve or print_handles:
        for cu in cubics:
          if print_curve:
            pts = sample_cubic(cu, n=150)
            ax.plot(pts[:, 0], pts[:, 1], "-")

          if not print_handles:
            continue

          p0, p1, p2, p3 = cu
          ctrl = np.array([p0, p1, p2, p3])
          ax.plot(ctrl[:, 0], ctrl[:, 1], "--", linewidth=0.5)
          ax.plot(
              np.array([ctrl[0], ctrl[-1]])[:, 0],
              np.array([ctrl[0], ctrl[-1]])[:, 1],
              "o",
              markersize=3,
          )
          ax.plot(
              np.array([ctrl[1], ctrl[2]])[:, 0],
              np.array([ctrl[1], ctrl[2]])[:, 1],
              "s",
              markersize=3,
          )

    self._has_content = True


def main():
  app = QApplication(sys.argv)
  viewer = VectorFitViewer()
  viewer.show()
  sys.exit(app.exec())


if __name__ == "__main__":
  main()
