from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz


@dataclass(slots=True)
class PredictedText:
    bounding_box: np.ndarray
    confidence: float
    text: str


def iter_word_predictions(result) -> list[tuple[np.ndarray, float, str]]:
    """Extract ``(box, confidence, text)`` tuples from a PaddleOCR 3.x result.

    ``PaddleOCR.predict()`` returns dict-like ``OCRResult`` objects. The
    ``rec_*`` entries are already filtered and aligned with each other:
    ``rec_polys`` holds the quad boxes, ``rec_texts`` the recognized strings
    and ``rec_scores`` their confidences. Older minor versions may only
    expose ``dt_polys`` or ``rec_boxes``, so fall back to those.
    """
    texts = list(result.get('rec_texts') or [])
    scores = list(result.get('rec_scores') or [])

    polys = result.get('rec_polys')
    if polys is None:
        polys = result.get('dt_polys')

    if polys is not None:
        boxes = [np.asarray(poly, dtype=float).reshape(-1, 2) for poly in polys]
    else:
        # Fallback: xyxy boxes -> quad format.
        boxes = [
            np.asarray([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], dtype=float)
            for b in (result.get('rec_boxes') or [])
        ]

    return [
        (box, float(conf), text)
        for box, conf, text in zip(boxes, scores, texts)
        if text
    ]


class PredictedFrames:
    start_index: int  # 0-based index of the frame
    end_index: int
    words: list[PredictedText]
    confidence: float  # total confidence of all words
    text: str

    def __init__(self, index: int, pred_data: list, conf_threshold: float):
        self.start_index = index
        self.end_index = index
        self.lines: list[list[PredictedText]] = []

        total_conf = 0.0
        raw_word_count = 0
        word_count = 0
        current_line: list[PredictedText] = []
        current_line_max_y = None
        for bounding_box, conf, text in iter_word_predictions(pred_data[0]):
            raw_word_count += 1

            # word predictions with low confidence will be filtered out
            if conf >= conf_threshold:
                total_conf += conf
                word_count += 1

                # add word to current line or create a new line
                max_y = max(bounding_box[0][1], bounding_box[1][1], bounding_box[2][1], bounding_box[3][1])

                if current_line_max_y is None:
                    current_line_max_y = max_y
                    current_line.append(PredictedText(bounding_box, conf, text))
                else:
                    min_y = min(bounding_box[0][1], bounding_box[1][1], bounding_box[2][1], bounding_box[3][1])
                    height = max_y - min_y
                    height_overlap_allowance = height * 0.1
                    if min_y >= current_line_max_y - height_overlap_allowance:  # new line
                        self.lines.append(current_line)
                        current_line = [PredictedText(bounding_box, conf, text)]
                        current_line_max_y = max_y
                    else:
                        current_line.append(PredictedText(bounding_box, conf, text))
                        current_line_max_y = max(current_line_max_y, max_y)

        if len(current_line) > 0:
            self.lines.append(current_line)
        if self.lines:
            self.confidence = total_conf / word_count
            for line in self.lines:
                line.sort(key=lambda word: word.bounding_box[0][0])
        elif raw_word_count == 0:
            self.confidence = 100
        else:
            self.confidence = 0
        self.text = '\n'.join(' '.join(word.text for word in line) for line in self.lines)

    def is_similar_to(self, other: PredictedFrames, threshold=70) -> bool:
        return fuzz.partial_ratio(self.text, other.text) >= threshold


class PredictedSubtitle:
    frames: list[PredictedFrames]
    sim_threshold: int
    text: str

    def __init__(self, frames: list[PredictedFrames], sim_threshold: int):
        self.frames = [f for f in frames if f.confidence > 0]
        self.frames.sort(key=lambda frame: frame.start_index)
        self.sim_threshold = sim_threshold

        if self.frames:
            self.text = max(self.frames, key=lambda f: f.confidence).text
        else:
            self.text = ''

    @property
    def index_start(self) -> int:
        if self.frames:
            return self.frames[0].start_index
        return 0

    @property
    def index_end(self) -> int:
        if self.frames:
            return self.frames[-1].end_index
        return 0

    def is_similar_to(self, other: PredictedSubtitle) -> bool:
        return fuzz.partial_ratio(self.text.replace(' ', ''), other.text.replace(' ', '')) >= self.sim_threshold

    def __repr__(self):
        return '{} - {}. {}'.format(self.index_start, self.index_end, self.text)
