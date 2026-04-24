"""SAM2 video predictor lifecycle wrapper.

Provides a clean API over SAM2's video prediction workflow.
Manages model loading, inference state, and GPU memory.
Architecture: single-object now, obj_id parameter for future multi-object.
"""

import os
import logging
from typing import Optional, Callable

import numpy as np
import torch

logger = logging.getLogger("sam2studio.mask_generator")


# (frame_idx, points, labels, obj_id) — the "baseline" conditioning points
# added by the user during Start Processing. Tracked separately from
# corrections so `reset_corrections()` can rebuild a clean inference state.
OriginalEntry = tuple[int, np.ndarray, np.ndarray, int]


class MaskGenerator:
    """Manages SAM2 video predictor lifecycle and inference."""

    def __init__(self, base_dir: str = ""):
        self._predictor = None
        self._inference_state = None
        self._config_key: Optional[tuple] = None
        self._video_dir: Optional[str] = None
        self._base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Points that were added as initial conditioning (not corrections).
        # Used by reset_corrections() to rebuild a clean state.
        self._original_conditioning: list[OriginalEntry] = []

    def initialize(
        self,
        model_cfg: str,
        checkpoint: str,
        device: str = "cuda",
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Load SAM2 model. Reuses existing if config matches."""
        checkpoint_path = os.path.join(self._base_dir, "checkpoints", checkpoint)
        # Hydra searches from pkg://sam2, so path is relative to sam2 package root
        config_path = f"configs/sam2.1/{model_cfg}"
        new_key = (checkpoint_path, config_path, device)

        if self._predictor is not None and self._config_key == new_key:
            logger.info("Reusing existing predictor (config unchanged)")
            return

        # Cleanup old predictor
        self.cleanup()

        if progress_callback:
            progress_callback("Loading SAM2 model...")

        logger.info(f"Initializing SAM2: model={model_cfg}, device={device}")

        from sam2.build_sam import build_sam2_video_predictor
        self._predictor = build_sam2_video_predictor(
            config_path, checkpoint_path, device=device
        )
        self._config_key = new_key
        logger.info("SAM2 model loaded successfully")

    def set_video(self, image_dir: str) -> None:
        """Initialize inference state from image directory.

        Optimized: if the same directory was already loaded, uses reset_state()
        to clear prompts/masks without re-reading video frames from disk.
        """
        if self._predictor is None:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")

        if self._inference_state is not None and self._video_dir == image_dir:
            # Same directory — reset state without re-reading frames
            self._predictor.reset_state(self._inference_state)
            self._original_conditioning.clear()
            logger.info(f"Reset inference state (reusing loaded frames from {image_dir})")
            return

        logger.info(f"Loading video frames from: {image_dir}")
        self._inference_state = self._predictor.init_state(video_path=image_dir)
        self._video_dir = image_dir
        self._original_conditioning.clear()

    def add_points(
        self,
        frame_idx: int,
        points: np.ndarray,
        labels: np.ndarray,
        obj_id: int = 1,
    ) -> Optional[np.ndarray]:
        """Add annotation points at a specific frame.

        Args:
            frame_idx: 0-based frame index.
            points: Array of shape (N, 2) with (x, y) coordinates.
            labels: Array of shape (N,) with 1=foreground, 0=background.
            obj_id: Object ID (default 1, for future multi-object).

        Returns:
            Preview mask logits for the frame, or None on error.
        """
        if self._predictor is None or self._inference_state is None:
            raise RuntimeError("Predictor or inference state not initialized.")

        _, out_obj_ids, out_mask_logits = self._predictor.add_new_points_or_box(
            inference_state=self._inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
        return out_mask_logits

    def propagate(
        self,
        threshold: float = 0.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        frame_callback: Optional[Callable[[int, np.ndarray], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
        start_frame_idx: Optional[int] = None,
    ) -> dict[int, np.ndarray]:
        """Run mask propagation through frames.

        Args:
            threshold: Mask logit threshold.
            progress_callback: Called with (frame_count, total_frames).
            frame_callback: Called with (frame_idx, binary_mask) per frame.
                When provided, masks are NOT accumulated in memory.
            stop_check: Callable returning True when propagation should stop.
            start_frame_idx: 0-based frame to start from. None = earliest
                annotated frame. Passed directly to SAM2's propagate_in_video.

        Returns:
            Dict mapping frame_idx to binary mask. Empty if frame_callback
            is used (caller handles each frame via callback).
        """
        if self._predictor is None or self._inference_state is None:
            raise RuntimeError("Predictor or inference state not initialized.")

        video_segments: dict[int, np.ndarray] = {}
        frame_count = 0

        for out_frame_idx, out_obj_ids, out_mask_logits in self._predictor.propagate_in_video(
            self._inference_state,
            start_frame_idx=start_frame_idx,
        ):
            if stop_check and stop_check():
                logger.info("Propagation stopped by user")
                break

            mask = (out_mask_logits[0] > threshold).cpu().numpy().squeeze()
            binary_mask = mask.astype(np.uint8) * 255
            frame_count += 1

            if frame_callback:
                frame_callback(out_frame_idx, binary_mask)
            else:
                video_segments[out_frame_idx] = binary_mask

            if progress_callback:
                progress_callback(frame_count, -1)

        logger.info(f"Propagation complete: {frame_count} frames")
        return video_segments

    def add_original_points(
        self,
        frame_idx: int,
        points: np.ndarray,
        labels: np.ndarray,
        obj_id: int = 1,
    ) -> None:
        """Add baseline conditioning points (from initial Start Processing).

        Forwards to SAM2's add_new_points_or_box *and* records the entry in
        `_original_conditioning`, so that a subsequent call to
        `reset_corrections()` can rebuild a clean inference state containing
        just these points.
        """
        self.add_points(frame_idx, points, labels, obj_id)
        self._original_conditioning.append(
            (
                int(frame_idx),
                np.asarray(points).copy(),
                np.asarray(labels).copy(),
                int(obj_id),
            )
        )

    def add_correction(
        self,
        frame_idx: int,
        points: np.ndarray,
        labels: np.ndarray,
        obj_id: int = 1,
    ) -> None:
        """Add correction points at a specific frame for re-propagation.

        Unlike `add_original_points`, corrections are NOT tracked in the
        originals list — they live only on the SAM2 inference_state and will
        be wiped by `reset_corrections()`.
        """
        self.add_points(frame_idx, points, labels, obj_id)

    def propagate_from(
        self,
        start_frame_idx: int,
        threshold: float = 0.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        frame_callback: Optional[Callable[[int, np.ndarray], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> dict[int, np.ndarray]:
        """Re-propagate from a specific frame onward.

        Passes start_frame_idx directly to SAM2's propagate_in_video
        so only frames from the correction point forward are computed.
        """
        return self.propagate(
            threshold=threshold,
            progress_callback=progress_callback,
            frame_callback=frame_callback,
            stop_check=stop_check,
            start_frame_idx=start_frame_idx,
        )

    def propagate_range(
        self,
        anchor_frame_idx: int,
        range_start: int,
        range_end: int,
        threshold: float = 0.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        frame_callback: Optional[Callable[[int, np.ndarray], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Re-propagate over `[range_start, range_end]` around an anchor frame.

        The anchor is where correction points have been added on this call.
        SAM2 is run forward (anchor → range_end) and, if needed, in reverse
        (anchor → range_start). The anchor frame is emitted exactly once
        (forward leg), even though SAM2 yields it for both legs.

        Raises:
            ValueError: if the range is empty or the anchor is outside it.
        """
        if self._predictor is None or self._inference_state is None:
            raise RuntimeError("Predictor or inference state not initialized.")

        if range_end < range_start:
            raise ValueError(
                f"Empty range: range_start={range_start} > range_end={range_end}"
            )
        if not (range_start <= anchor_frame_idx <= range_end):
            raise ValueError(
                f"Anchor frame {anchor_frame_idx} is outside range "
                f"[{range_start}, {range_end}]"
            )

        total = (range_end - range_start) + 1
        processed = 0

        def _emit(out_frame_idx: int, out_mask_logits) -> None:
            nonlocal processed
            mask = (out_mask_logits[0] > threshold).cpu().numpy().squeeze()
            binary_mask = mask.astype(np.uint8) * 255
            if frame_callback is not None:
                frame_callback(out_frame_idx, binary_mask)
            processed += 1
            if progress_callback is not None:
                progress_callback(processed, total)

        # --- Forward leg: anchor → range_end ---------------------------------
        fwd_max = range_end - anchor_frame_idx
        for out_frame_idx, _obj_ids, out_mask_logits in (
            self._predictor.propagate_in_video(
                self._inference_state,
                start_frame_idx=anchor_frame_idx,
                max_frame_num_to_track=fwd_max,
                reverse=False,
            )
        ):
            if stop_check is not None and stop_check():
                logger.info("propagate_range: forward leg stopped by user")
                return
            _emit(out_frame_idx, out_mask_logits)

        # --- Reverse leg: anchor → range_start (skip anchor re-emit) ---------
        if anchor_frame_idx > range_start:
            rev_max = anchor_frame_idx - range_start
            first_yield = True
            for out_frame_idx, _obj_ids, out_mask_logits in (
                self._predictor.propagate_in_video(
                    self._inference_state,
                    start_frame_idx=anchor_frame_idx,
                    max_frame_num_to_track=rev_max,
                    reverse=True,
                )
            ):
                if stop_check is not None and stop_check():
                    logger.info("propagate_range: reverse leg stopped by user")
                    return
                # SAM2's reverse loop yields the start frame (== anchor) first;
                # skip it so the anchor is written exactly once.
                if first_yield:
                    first_yield = False
                    continue
                _emit(out_frame_idx, out_mask_logits)

        logger.info(
            "propagate_range complete: anchor=%d range=[%d,%d] emitted=%d",
            anchor_frame_idx, range_start, range_end, processed,
        )

    def refine_early_frames(
        self,
        anchor_frame_idx: int,
        anchor_mask: np.ndarray,
        overwrite_count: int,
        threshold: float = 0.0,
        frame_callback: Optional[Callable[[int, np.ndarray], None]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Refine the earliest frames of the sequence via reverse propagation.

        Rationale
        ---------
        SAM2's first frame is a permanent conditioning frame, and its memory
        attention has nothing to attend to — so the first frame's mask is
        usually the worst of the sequence. Worse, errors in frame 0 can
        propagate forward through the whole video because memory attention
        keeps cross-attending to the (bad) frame 0 mask.

        This method fixes that by treating a later frame's (good) mask as a
        new conditioning source on a reset inference state, then reverse-
        propagating from that anchor back to frame 0. The reverse pass's
        memory bank fills with good context before it reaches frame 0, so
        frame 0 comes out substantially cleaner. The earliest K frames are
        then overwritten on disk via `frame_callback`; the remaining frames
        between K and the anchor are still computed (to build memory) but
        their outputs are discarded.

        Args:
            anchor_frame_idx: 0-based index of the frame whose already-good
                mask will be used as the reverse-propagation source.
                Must be ``>= 1``.
            anchor_mask: Binary mask (ndarray) for the anchor frame. Any
                shape is accepted — SAM2 resizes internally.
            overwrite_count: Number K of earliest frames to overwrite. Valid
                range: ``1 <= K <= anchor_frame_idx``. The frames written via
                ``frame_callback`` are 0-based indices ``[0, K-1]``.
            threshold: Mask logit threshold (same semantics as ``propagate``).
            frame_callback: Called as ``fn(frame_idx, binary_mask)`` for each
                overwritten frame. ``binary_mask`` is ``uint8`` in [0, 255].
            progress_callback: Called as ``fn(processed, total)`` where
                ``total == overwrite_count``.
            stop_check: If provided, checked before each yielded frame;
                returning True aborts the loop early.

        Raises:
            ValueError: if ``anchor_frame_idx < 1``, ``overwrite_count < 1``,
                or ``overwrite_count > anchor_frame_idx``.
            RuntimeError: if the predictor / inference state is not ready.
        """
        if self._predictor is None or self._inference_state is None:
            raise RuntimeError("Predictor or inference state not initialized.")

        if anchor_frame_idx < 1:
            raise ValueError(
                f"anchor_frame_idx must be >= 1 (got {anchor_frame_idx})"
            )
        if overwrite_count < 1:
            raise ValueError(
                f"overwrite_count must be >= 1 (got {overwrite_count})"
            )
        if overwrite_count > anchor_frame_idx:
            raise ValueError(
                f"overwrite_count ({overwrite_count}) must be "
                f"<= anchor_frame_idx ({anchor_frame_idx})"
            )

        # --- Wipe any prior conditioning on the current inference_state -----
        # We can't use reset_corrections() here because that replays the
        # original points, which would re-introduce frame-0 bias. Call
        # reset_state directly and inject the anchor mask as the sole source.
        self._predictor.reset_state(self._inference_state)

        # --- Inject the anchor mask as the SOLE conditioning ----------------
        mask_tensor = torch.as_tensor(np.asarray(anchor_mask)).bool()
        self._predictor.add_new_mask(
            inference_state=self._inference_state,
            frame_idx=anchor_frame_idx,
            obj_id=1,
            mask=mask_tensor,
        )

        # --- Reverse propagate anchor -> 0, saving only the earliest K ------
        processed = 0
        for out_frame_idx, _obj_ids, out_mask_logits in (
            self._predictor.propagate_in_video(
                self._inference_state,
                start_frame_idx=anchor_frame_idx,
                max_frame_num_to_track=anchor_frame_idx,
                reverse=True,
            )
        ):
            if stop_check is not None and stop_check():
                logger.info("refine_early_frames: stopped by user")
                break

            frame_idx = int(out_frame_idx)
            # Skip the anchor itself (yielded first) and any frame outside
            # the earliest K window. They still walked through the loop so
            # the memory bank accumulated their context.
            if frame_idx >= overwrite_count:
                continue

            mask = (out_mask_logits[0] > threshold).cpu().numpy().squeeze()
            binary_mask = mask.astype(np.uint8) * 255

            if frame_callback is not None:
                frame_callback(frame_idx, binary_mask)

            processed += 1
            if progress_callback is not None:
                progress_callback(processed, overwrite_count)

        logger.info(
            "refine_early_frames complete: anchor=%d overwrite=%d emitted=%d",
            anchor_frame_idx, overwrite_count, processed,
        )

        # --- Restore the "clean baseline" state -----------------------------
        # After refine the inference_state contains the anchor mask as
        # conditioning, which would interfere with a subsequent Re-run Range.
        # Replay the originally-added points so the state mirrors what it
        # was at the end of Start Processing.
        self.reset_corrections()

    def reset_corrections(self) -> None:
        """Wipe SAM2 inference state and replay the original conditioning.

        This is the counterpart to `add_correction`: after a correction pass
        finishes, call this to get back to a clean state that contains only
        the points originally added during Start Processing. It lets the user
        start a fresh correction without interference from previous ones.

        Safe to call with no originals tracked (still resets any leftover
        SAM2 state).
        """
        if self._predictor is None or self._inference_state is None:
            return

        self._predictor.reset_state(self._inference_state)
        for frame_idx, points, labels, obj_id in self._original_conditioning:
            self._predictor.add_new_points_or_box(
                inference_state=self._inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )

    def cleanup(self) -> None:
        """Release model and clear GPU cache."""
        if self._predictor is not None and self._inference_state is not None:
            try:
                self._predictor.reset_state(self._inference_state)
            except Exception as e:
                logger.warning(f"Error resetting state: {e}")

        self._inference_state = None
        self._predictor = None
        self._config_key = None
        self._video_dir = None
        self._original_conditioning.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

    @property
    def is_initialized(self) -> bool:
        return self._predictor is not None

    @property
    def has_inference_state(self) -> bool:
        return self._inference_state is not None
