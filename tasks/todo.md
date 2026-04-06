# Correction Range Refactor — Implementation Plan

## Motivation

Today's "Fix Mask / Apply & Propagate" always re-propagates from the current
frame to the end of the sequence. Real workflows often need to fix a short
stretch of bad frames in the middle, not "everything after". This refactor
turns the feature into a **range-based correction**: the user picks an
anchor frame, drops correction points there, and selects `[start, end]`.
SAM2 is then run forward and/or backward to cover exactly that interval.

Overwriting `masks/` in-place is kept (matches original author intent).

## Confirmed Design Decisions

| Topic | Decision |
|---|---|
| Propagation direction | **Bidirectional** (forward + reverse from anchor) |
| Range input UI | Two SpinBoxes on the toolbar, flanking the two action buttons |
| Default range on entry | `[current, current]` (only the current frame) |
| Frame navigation in CORRECTION | Allowed, but first point locks anchor to that frame |
| Points on non-anchor frames | Rejected once anchor is locked |
| inference_state hygiene | After each successful Apply, `reset_state()` + replay original annotation points (tracked separately from corrections) |
| Write target | Keep overwriting `masks/` in place (no separate dir) |
| Button names | `Add Correction` / `Re-run Range` |

## Architecture Overview

```
core/mask_generator.py
  ├─ _original_conditioning: list[(frame_idx, points, labels, obj_id)]
  ├─ add_original_points(...)      ← new; used during Start Processing
  ├─ add_correction(...)           ← unchanged signature, but no longer tracked
  ├─ propagate_range(anchor, start, end, ...)   ← new; fwd + rev dispatch
  └─ reset_corrections()           ← new; reset_state + replay original

controllers/correction_controller.py   ← NEW
  ├─ anchor_frame: Optional[int]
  ├─ range_start, range_end: int
  ├─ signals: anchor_changed, range_changed
  ├─ try_add_point(frame, x, y, label) → bool  (enforces anchor lock)
  ├─ on_frame_changed(frame)          (no-op unless we need to refuse)
  ├─ on_enter_correction(current_frame, total_frames)
  ├─ on_exit_correction()             (clear anchor, reset range)
  └─ can_apply() → bool

controllers/workers/correction_worker.py
  ├─ now takes (anchor, range_start, range_end)
  ├─ calls mask_generator.propagate_range(...)
  └─ calls mask_generator.reset_corrections() after successful completion

gui/panels/toolbar.py
  ├─ Rename buttons: Add Correction / Re-run Range
  ├─ Add two QSpinBoxes (Start / End) between/near the action buttons
  ├─ New signals: correction_range_changed(int, int)
  └─ set_correction_range(s, e), set_frame_count(total)

gui/main_window.py
  ├─ Instantiate CorrectionController
  ├─ Wire toolbar.correction_range_changed ↔ controller
  ├─ Intercept annotation point-add in CORRECTION state via try_add_point
  ├─ _on_add_correction:  enter state + seed default range [cur, cur]
  ├─ _on_apply_correction: validate + launch worker with (anchor, s, e)
  └─ _update_ui_for_state: enable spin boxes only during CORRECTION
```

## TDD Steps

### Phase A — Core: propagate_range + original-points tracking

- [ ] A1. Write `tests/test_correction_core.py` against a fake predictor:
  - [ ] `test_original_points_tracked_on_add_original_points`
  - [ ] `test_add_correction_does_not_touch_original_list`
  - [ ] `test_propagate_range_single_frame_runs_forward_only_one_frame`
  - [ ] `test_propagate_range_forward_only_when_anchor_equals_start`
  - [ ] `test_propagate_range_backward_only_when_anchor_equals_end`
  - [ ] `test_propagate_range_bidirectional_writes_anchor_exactly_once`
  - [ ] `test_propagate_range_invalid_anchor_outside_raises`
  - [ ] `test_propagate_range_empty_range_raises`
  - [ ] `test_reset_corrections_calls_reset_state_then_replays_originals`
  - [ ] `test_reset_corrections_noop_when_no_originals`
  - Fake predictor: captures calls to `propagate_in_video`, `add_new_points_or_box`, `reset_state`; returns dummy mask logits.
- [ ] A2. Implement in `core/mask_generator.py`:
  - [ ] `self._original_conditioning: list[tuple[int, np.ndarray, np.ndarray, int]]`
  - [ ] `add_original_points(frame_idx, points, labels, obj_id=1)` — delegates to `add_points` AND appends to the list
  - [ ] `propagate_range(anchor, range_start, range_end, threshold, frame_callback, stop_check)`:
    - Validate `range_start <= anchor <= range_end`, non-empty
    - Call `propagate_in_video(start=anchor, max=range_end-anchor, reverse=False)`; dispatch via `frame_callback`
    - If `anchor > range_start`: call `propagate_in_video(start=anchor, max=anchor-range_start, reverse=True)`; **skip the first yield** (anchor frame already written)
  - [ ] `reset_corrections()` — `predictor.reset_state(...)`, then re-play every entry in `_original_conditioning` via `add_new_points_or_box`
- [ ] A3. Run tests → GREEN

### Phase B — CorrectionController

- [ ] B1. Write `tests/test_correction_controller.py` (pure Qt-free logic):
  - [ ] `test_initial_state`: anchor None, range (1, 1)
  - [ ] `test_on_enter_correction_sets_default_range`
  - [ ] `test_try_add_point_locks_anchor_to_current_frame`
  - [ ] `test_try_add_point_on_anchor_frame_succeeds`
  - [ ] `test_try_add_point_on_other_frame_rejected_once_locked`
  - [ ] `test_clear_anchor_on_exit`
  - [ ] `test_set_range_rejects_start_greater_than_end`
  - [ ] `test_set_range_clamps_to_total_frames`
  - [ ] `test_set_range_with_locked_anchor_outside_rejected`
  - [ ] `test_can_apply_requires_locked_anchor_and_valid_range`
- [ ] B2. Implement `controllers/correction_controller.py`:
  - [ ] `CorrectionController(QObject)`
  - [ ] Signals: `anchor_changed(object)`  (Optional[int]), `range_changed(int, int)`
  - [ ] `try_add_point(frame_idx) -> bool` — caller still adds the point itself, this just guards the state machine
  - [ ] `on_enter_correction(current_frame, total_frames)`
  - [ ] `on_exit_correction()`
  - [ ] `set_range(start, end)` + `set_total_frames(n)`
  - [ ] `can_apply(points_count: int) -> bool`
- [ ] B3. Run tests → GREEN

### Phase C — Toolbar range UI

- [ ] C1. Edit `gui/panels/toolbar.py`:
  - [ ] Rename `_add_correction_btn` label → `"Add Correction"` (tooltip updated)
  - [ ] Rename `_apply_correction_btn` label → `"Re-run Range"` (tooltip updated)
  - [ ] Add `_range_start_spin`, `_range_end_spin` (`QSpinBox`, min=1)
  - [ ] Small "Range:" label; horizontal layout between the two buttons
  - [ ] New pyqtSignal: `correction_range_changed(int, int)`
  - [ ] New public methods: `set_correction_range(s, e)`, `set_correction_frame_count(total)`, `set_correction_range_enabled(bool)`
  - [ ] SpinBox valueChanged → debounced emit of `correction_range_changed`
- [ ] C2. Sanity check: import the toolbar in a tiny script, verify signals fire.

### Phase D — Worker refactor

- [ ] D1. Update `controllers/workers/correction_worker.py`:
  - [ ] New params: `anchor_frame_idx`, `range_start`, `range_end` (0-based inclusive)
  - [ ] Drop `frame_idx` param (replaced by `anchor_frame_idx`)
  - [ ] Call `propagate_range(anchor, start, end, ...)` instead of `propagate_from`
  - [ ] Progress total = `range_end - range_start + 1`
  - [ ] On successful completion, call `self._mask_generator.reset_corrections()`
  - [ ] On error / stop: also call `reset_corrections()` in finally? **No** — only on success. On stop/error, leave state dirty and force a note in logs.
- [ ] D2. Update `controllers/processing_controller.py` (wherever `start_correction` lives) to accept the new signature

### Phase E — MainWindow integration

- [ ] E1. Instantiate `CorrectionController` in `MainWindow.__init__`, wire signals
- [ ] E2. `_on_add_correction`:
  - [ ] Check model initialized (unchanged)
  - [ ] Clear `_state.points`
  - [ ] Call `corr_ctrl.on_enter_correction(current_frame, total)`
  - [ ] Transition to CORRECTION state
- [ ] E3. Hook the annotation point-add path:
  - [ ] Before adding a point in CORRECTION state, call `corr_ctrl.try_add_point(current_frame)`
  - [ ] On rejection, show a toast/status message ("Points can only be added on the anchor frame")
- [ ] E4. Toolbar range SpinBox signals → `corr_ctrl.set_range(...)`
- [ ] E5. `corr_ctrl.range_changed` / `anchor_changed` → toolbar sync
- [ ] E6. `_on_apply_correction`:
  - [ ] Validate via `corr_ctrl.can_apply(len(points))`
  - [ ] Read `(anchor, s, e)` from controller
  - [ ] Launch worker with new signature
- [ ] E7. `_on_processing_finished` (for correction runs): clear `_state.points`, call `corr_ctrl.on_exit_correction()`, transition to REVIEWING
- [ ] E8. `_update_ui_for_state`:
  - [ ] Spin boxes enabled only during CORRECTION
  - [ ] Range must be clamped to `[1, total_frames]` when `set_image_files` happens
- [ ] E9. `_shortcut_escape` in CORRECTION: already transitions to REVIEWING; additionally call `corr_ctrl.on_exit_correction()`

### Phase F — Replace original-points tracking at Start Processing

- [ ] F1. Find where `ProcessingWorker` (or `start_processing`) calls `mask_generator.add_points` with the initial annotation
- [ ] F2. Switch that call to `add_original_points` so the list is seeded
- [ ] F3. Verify via existing `tests/test_phase4_core.py` / `test_phase5_processing.py` don't break

### Phase G — Manual verification

- [ ] G1. Run full sequence, enter REVIEWING
- [ ] G2. Navigate to a bad frame, click `Add Correction` → CORRECTION state, range defaults to `[cur, cur]`
- [ ] G3. Drop a correction point — anchor frame locks, confirmed visually
- [ ] G4. Expand the range to cover a 15-frame window (e.g. anchor in the middle)
- [ ] G5. Click `Re-run Range` — verify only those frames are re-propagated (check progress + file mtimes)
- [ ] G6. Navigate inside the range — points stay locked to anchor, can't add on other frames
- [ ] G7. Do a second correction — verify `reset_corrections` cleaned up inference_state (second correction is independent of the first)
- [ ] G8. Click Escape mid-correction → returns to REVIEWING cleanly, anchor cleared
- [ ] G9. Confirm original annotation points still work (no regression in first-time Start Processing)

## Non-Goals (Phase 1)

- No separate `masks_corrected/` output directory (still overwrites `masks/`)
- No multi-object support (still hardcoded `obj_id=1`)
- No persisting correction history to `.s2proj`
- No per-frame correction points (one anchor per Apply)
- No undo of a completed Apply (you can re-run over the same range)
- No "preview" before committing (Apply is immediate)

## Review

(To be filled in after execution)
