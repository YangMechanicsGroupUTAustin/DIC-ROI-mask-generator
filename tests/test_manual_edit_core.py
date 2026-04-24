"""Unit tests for core.manual_edit (Phase A, pure brush/stroke functions)."""

import numpy as np
import pytest


class TestPaintDot:
    def test_paint_dot_fills_circle(self):
        """A dot at (50,50) r=5 fills pixels within radius, leaves the rest alone."""
        from core.manual_edit import paint_dot

        mask = np.zeros((100, 100), dtype=np.uint8)
        paint_dot(mask, cx=50, cy=50, radius=5, value=255)

        # Center and pixels strictly inside the disk
        assert mask[50, 50] == 255
        assert mask[50, 53] == 255  # 3 px right
        assert mask[53, 50] == 255  # 3 px down
        assert mask[47, 47] == 255  # diagonal, dist ~4.24

        # Pixels clearly outside the disk must be untouched
        assert mask[50, 60] == 0  # 10 px away
        assert mask[30, 30] == 0  # far away
        assert mask[0, 0] == 0

    def test_paint_dot_bbox_is_tight(self):
        """The returned bbox is the exact (clipped) extent of the disk."""
        from core.manual_edit import paint_dot

        mask = np.zeros((100, 100), dtype=np.uint8)
        bbox = paint_dot(mask, cx=50, cy=50, radius=5, value=255)

        # Numpy-slice style: (x0, y0, x1, y1), x1/y1 exclusive.
        assert bbox == (45, 45, 56, 56)

    def test_paint_dot_clipped_at_image_edge(self):
        """Dots near the edge must clip their bbox to image bounds."""
        from core.manual_edit import paint_dot

        mask = np.zeros((100, 100), dtype=np.uint8)

        # Upper-left corner: radius overflows negative region.
        bbox_ul = paint_dot(mask, cx=2, cy=2, radius=5, value=255)
        assert bbox_ul == (0, 0, 8, 8)

        # Lower-right corner: radius overflows beyond image.
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        bbox_lr = paint_dot(mask2, cx=98, cy=98, radius=5, value=255)
        assert bbox_lr == (93, 93, 100, 100)

        # Dot completely outside the image should still return a valid
        # (possibly empty) bbox clipped to image bounds, and not crash.
        mask3 = np.zeros((100, 100), dtype=np.uint8)
        bbox_out = paint_dot(mask3, cx=-50, cy=-50, radius=5, value=255)
        assert bbox_out is None or bbox_out == (0, 0, 0, 0)


class TestPaintStroke:
    def test_paint_stroke_connects_points_with_line(self):
        """A horizontal stroke must paint every intermediate pixel."""
        from core.manual_edit import paint_stroke

        mask = np.zeros((100, 100), dtype=np.uint8)
        paint_stroke(mask, points=[(10, 50), (90, 50)], radius=1, value=255)

        # The central row between the two endpoints should be fully painted.
        for x in range(10, 91):
            assert mask[50, x] == 255, f"pixel ({x}, 50) should be painted"

    def test_paint_stroke_eraser_clears_to_zero(self):
        """Stroking with value=0 erases existing mask along the path."""
        from core.manual_edit import paint_stroke

        mask = np.full((100, 100), 255, dtype=np.uint8)
        paint_stroke(mask, points=[(20, 20), (80, 80)], radius=5, value=0)

        # Endpoints cleared
        assert mask[20, 20] == 0
        assert mask[80, 80] == 0
        # Midpoint on the diagonal cleared
        assert mask[50, 50] == 0
        # A point well outside the stroke must remain painted
        assert mask[5, 95] == 255

    def test_paint_stroke_respects_radius(self):
        """Pixels inside the thick line are painted, pixels outside are not."""
        from core.manual_edit import paint_stroke

        mask = np.zeros((100, 100), dtype=np.uint8)
        paint_stroke(mask, points=[(10, 50), (90, 50)], radius=3, value=255)

        # Pixels within radius 3 of the horizontal line y=50 should be painted.
        assert mask[48, 50] == 255  # 2 px above
        assert mask[52, 50] == 255  # 2 px below
        assert mask[50, 50] == 255  # on the line

        # Pixels well outside (distance > 3) must remain zero.
        assert mask[45, 50] == 0  # 5 px above
        assert mask[55, 50] == 0  # 5 px below

    def test_paint_stroke_bbox_covers_all_points(self):
        """The stroke bbox must span from min-point-radius to max-point+radius+1."""
        from core.manual_edit import paint_stroke

        mask = np.zeros((100, 100), dtype=np.uint8)
        bbox = paint_stroke(
            mask, points=[(20, 30), (60, 70)], radius=4, value=255
        )
        assert bbox == (16, 26, 65, 75)

    def test_paint_stroke_single_point_behaves_like_dot(self):
        """A single-point stroke is equivalent to paint_dot."""
        from core.manual_edit import paint_dot, paint_stroke

        mask_a = np.zeros((100, 100), dtype=np.uint8)
        mask_b = np.zeros((100, 100), dtype=np.uint8)

        paint_dot(mask_a, cx=50, cy=50, radius=6, value=255)
        paint_stroke(mask_b, points=[(50, 50)], radius=6, value=255)

        assert np.array_equal(mask_a, mask_b)

    def test_paint_stroke_empty_points_is_noop(self):
        """An empty stroke must not crash and must not modify the mask."""
        from core.manual_edit import paint_stroke

        mask = np.zeros((100, 100), dtype=np.uint8)
        bbox = paint_stroke(mask, points=[], radius=4, value=255)

        assert bbox is None
        assert mask.sum() == 0
