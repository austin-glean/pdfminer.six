import random
import unittest
from collections.abc import Iterator
from unittest import mock

from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LAParams,
    LTLayoutContainer,
    LTTextBox,
    LTTextBoxHorizontal,
    LTTextBoxVertical,
    LTTextLine,
    LTTextLineHorizontal,
    LTTextLineVertical,
)
from pdfminer.utils import Plane, uniq
from tests.helpers import absolute_sample_path


def _line(
    bbox: tuple[float, float, float, float], vertical: bool = False
) -> LTTextLine:
    line = LTTextLineVertical(0.1) if vertical else LTTextLineHorizontal(0.1)
    line.set_bbox(bbox)
    return line


def _reference_group_textlines(
    layout: LTLayoutContainer, laparams: LAParams, lines: list[LTTextLine]
) -> Iterator[LTTextBox]:
    """The previous, quadratic group_textlines, with each line treated as its
    own neighbor (which it always is for the non-empty lines analyze() passes).
    """
    plane: Plane[LTTextLine] = Plane(layout.bbox)
    plane.extend(lines)
    boxes: dict[LTTextLine, LTTextBox] = {}
    for line in lines:
        neighbors = line.find_neighbors(plane, laparams.line_margin)
        if line not in neighbors:
            neighbors.append(line)
        members = [line]
        for obj1 in neighbors:
            members.append(obj1)
            if obj1 in boxes:
                members.extend(boxes.pop(obj1))
        box: LTTextBox
        if isinstance(line, LTTextLineHorizontal):
            box = LTTextBoxHorizontal()
        else:
            box = LTTextBoxVertical()
        for obj in uniq(members):
            box.add(obj)
            boxes[obj] = box
    done = set()
    for line in lines:
        if line in boxes and boxes[line] not in done:
            done.add(boxes[line])
            if not boxes[line].is_empty():
                yield boxes[line]


def _random_lines(rng: random.Random) -> list[LTTextLine]:
    """Clusters of aligned, tightly spaced, and overlapping lines."""
    lines = []
    for _ in range(rng.randint(1, 6)):
        x, y = rng.uniform(0, 500), rng.uniform(0, 700)
        height = rng.choice([2.0, 6.0, 10.0, 12.0])
        for _ in range(rng.randint(1, 60)):
            x0 = x + rng.choice([0.0, 0.0, rng.uniform(-3, 3), rng.uniform(-60, 60)])
            y0 = y + rng.uniform(-30, 30) * rng.choice([0, 0.05, 0.3, 1])
            h = height + rng.choice([0, 0, rng.uniform(-1, 1)])
            bbox = (x0, y0, x0 + rng.uniform(1, 200), y0 + h)
            lines.append(_line(bbox, vertical=rng.random() < 0.15))
    rng.shuffle(lines)
    return lines


def _boxes(
    textboxes: Iterator[LTTextBox],
) -> list[tuple[str, tuple[int, ...], tuple[float, float, float, float]]]:
    return [(type(box).__name__, tuple(map(id, box)), box.bbox) for box in textboxes]


class TestGroupTextLines(unittest.TestCase):
    def test_parent_with_wrong_bbox_returns_non_empty_neighbour_list(self):
        """LTLayoutContainer.group_textlines() should return all the lines in a
        separate LTTextBoxes if they do not overlap. Even when the bounding box
        of the parent container does not contain all the lines.
        """
        laparams = LAParams()
        layout = LTLayoutContainer((0, 0, 50, 50))
        line1 = LTTextLineHorizontal(laparams.word_margin)
        line1.set_bbox((0, 0, 50, 5))
        line2 = LTTextLineHorizontal(laparams.word_margin)
        line2.set_bbox((0, 50, 50, 55))
        lines = [line1, line2]

        textboxes = list(layout.group_textlines(laparams, lines))

        self.assertEqual(len(textboxes), 2)

    def test_matches_reference_grouping(self):
        """Boxes, their types, and the order of lines within them match the
        straightforward (quadratic) grouping.
        """
        laparams = LAParams()
        layout = LTLayoutContainer((0, 0, 612, 792))
        for seed in range(300):
            lines = _random_lines(random.Random(seed))
            self.assertEqual(
                _boxes(layout.group_textlines(laparams, lines)),
                _boxes(_reference_group_textlines(layout, laparams, lines)),
                f"seed={seed}",
            )

    def test_dense_lines_are_added_to_a_box_once(self):
        """Overprinted, tightly spaced lines form one large box. Building it
        must not re-add the lines of the growing box for every new line.
        """
        laparams = LAParams()
        layout = LTLayoutContainer((0, 0, 612, 792))
        lines = [
            _line((36 + (i % 16) * 0.3, 750 - (i // 16) * 7, 336, 760 - (i // 16) * 7))
            for i in range(1600)
        ]
        with mock.patch.object(
            LTTextBoxHorizontal, "add", autospec=True, side_effect=LTTextBox.add
        ) as add:
            textboxes = list(layout.group_textlines(laparams, lines))

        self.assertEqual(len(textboxes), 1)
        self.assertEqual(add.call_count, 1600)

    def test_line_that_is_not_its_own_neighbor_is_in_one_box(self):
        """A zero-width line does not find itself as a neighbor. It is still
        grouped together with the box it is already in.
        """
        laparams = LAParams()
        layout = LTLayoutContainer((0, 0, 612, 792))
        tall = _line((10, 100, 110, 110))
        zero_width = _line((60, 102, 60, 108))
        short = _line((40, 103, 80, 107))

        textboxes = list(layout.group_textlines(laparams, [tall, zero_width, short]))

        grouped = [line for box in textboxes for line in box]
        self.assertCountEqual(grouped, [tall, zero_width, short])

    def test_repeated_line_is_grouped_once(self):
        laparams = LAParams()
        layout = LTLayoutContainer((0, 0, 612, 792))
        line1 = _line((10, 100, 110, 110))
        line2 = _line((10, 112, 110, 122))

        textboxes = list(layout.group_textlines(laparams, [line1, line2, line1]))

        self.assertEqual(len(textboxes), 1)
        self.assertCountEqual(list(textboxes[0]), [line1, line2])


class TestFindNeigbors(unittest.TestCase):
    def test_find_neighbors_horizontal(self):
        laparams = LAParams()
        plane = Plane((0, 0, 50, 50))

        line = LTTextLineHorizontal(laparams.word_margin)
        line.set_bbox((10, 4, 20, 6))
        plane.add(line)

        left_aligned_above = LTTextLineHorizontal(laparams.word_margin)
        left_aligned_above.set_bbox((10, 6, 15, 8))
        plane.add(left_aligned_above)

        right_aligned_below = LTTextLineHorizontal(laparams.word_margin)
        right_aligned_below.set_bbox((15, 2, 20, 4))
        plane.add(right_aligned_below)

        centrally_aligned_overlapping = LTTextLineHorizontal(laparams.word_margin)
        centrally_aligned_overlapping.set_bbox((13, 5, 17, 7))
        plane.add(centrally_aligned_overlapping)

        not_aligned = LTTextLineHorizontal(laparams.word_margin)
        not_aligned.set_bbox((0, 6, 5, 8))
        plane.add(not_aligned)

        wrong_height = LTTextLineHorizontal(laparams.word_margin)
        wrong_height.set_bbox((10, 6, 15, 10))
        plane.add(wrong_height)

        neighbors = line.find_neighbors(plane, laparams.line_margin)
        self.assertCountEqual(
            neighbors,
            [
                line,
                left_aligned_above,
                right_aligned_below,
                centrally_aligned_overlapping,
            ],
        )

    def test_find_neighbors_vertical(self):
        laparams = LAParams()
        plane = Plane((0, 0, 50, 50))

        line = LTTextLineVertical(laparams.word_margin)
        line.set_bbox((4, 10, 6, 20))
        plane.add(line)

        bottom_aligned_right = LTTextLineVertical(laparams.word_margin)
        bottom_aligned_right.set_bbox((6, 10, 8, 15))
        plane.add(bottom_aligned_right)

        top_aligned_left = LTTextLineVertical(laparams.word_margin)
        top_aligned_left.set_bbox((2, 15, 4, 20))
        plane.add(top_aligned_left)

        centrally_aligned_overlapping = LTTextLineVertical(laparams.word_margin)
        centrally_aligned_overlapping.set_bbox((5, 13, 7, 17))
        plane.add(centrally_aligned_overlapping)

        not_aligned = LTTextLineVertical(laparams.word_margin)
        not_aligned.set_bbox((6, 0, 8, 5))
        plane.add(not_aligned)

        wrong_width = LTTextLineVertical(laparams.word_margin)
        wrong_width.set_bbox((6, 10, 10, 15))
        plane.add(wrong_width)

        neighbors = line.find_neighbors(plane, laparams.line_margin)
        self.assertCountEqual(
            neighbors,
            [
                line,
                bottom_aligned_right,
                top_aligned_left,
                centrally_aligned_overlapping,
            ],
        )


def test_pdf_with_empty_characters_horizontal():
    """Regression test for issue #449

    See: https://github.com/pdfminer/pdfminer.six/pull/689

    The page aggregator should separate the 3 horizontal lines in the
    sample PDF. The used PDF sample has multiple explicit space characters
    in between lines with text.
    """
    path = absolute_sample_path("contrib/issue-449-horizontal.pdf")
    pages = extract_pages(path)
    textboxes = [
        textbox for textbox in next(pages) if isinstance(textbox, LTTextBoxHorizontal)
    ]
    assert len(textboxes) == 3


def test_pdf_with_empty_characters_vertical():
    """Regression test for issue #449

    See: https://github.com/pdfminer/pdfminer.six/pull/689

    The page aggregator should separate the 3 horizontal lines in the
    sample PDF. The used PDF sample has multiple explicit space characters
    in between lines with text.
    """
    path = absolute_sample_path("contrib/issue-449-vertical.pdf")
    laparams = LAParams(detect_vertical=True)
    pages = extract_pages(path, laparams=laparams)
    textboxes = [
        textbox for textbox in next(pages) if isinstance(textbox, LTTextBoxVertical)
    ]
    assert len(textboxes) == 3
