from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import paddle
import yaml
from paddleocr import PaddleOCR

from . import utils
from .models import PredictedFrames, PredictedSubtitle
from .opencv_adapter import Capture

logger = logging.getLogger(__name__)

# PaddleOCR/PaddleX log a lot of noise; keep the console readable.
logging.getLogger('paddleocr').setLevel(logging.ERROR)
logging.getLogger('paddlex').setLevel(logging.ERROR)

def _model_name_from_dir(model_dir: str | None) -> str | None:
    """Read ``Global.model_name`` from a PaddleX inference model directory.

    PaddleX rejects a model directory whose ``inference.yml`` declares a
    different model name than the expected one, so the name stored in the
    directory has to be passed along with the directory.
    """
    if not model_dir:
        return None
    config_file = Path(model_dir) / 'inference.yml'
    if not config_file.is_file():
        return None
    config = yaml.safe_load(config_file.read_text(encoding='utf-8')) or {}
    return config.get('Global', {}).get('model_name')


def _cuda_available() -> bool:
    """Return whether a CUDA GPU is usable by the installed PaddlePaddle build."""
    try:
        return paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except Exception:
        return False


class Video:
    def __init__(self, path: str, det_model_dir: str | None = None, rec_model_dir: str | None = None):
        self.path = path
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        with Capture(path) as v:
            self.num_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = v.get(cv2.CAP_PROP_FPS)
            self.height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pred_frames: list[PredictedFrames] = []
        self.pred_subs: list[PredictedSubtitle] = []

    def _create_ocr(self, use_gpu: bool, lang: str) -> PaddleOCR:
        """Create a PaddleOCR 3.x pipeline.

        Compared to PaddleOCR 2.x, the following arguments were renamed:
        ``det_model_dir``/``rec_model_dir`` -> ``text_detection_model_dir``/
        ``text_recognition_model_dir``, ``use_gpu`` -> ``device``,
        ``use_angle_cls`` -> ``use_textline_orientation``, and ``show_log``
        no longer exists (logging is configured through the logging module).

        Unless explicit model directories are given, PaddleOCR's own default
        models (server-grade PP-OCRv6 "medium", or the prebuilt models for
        the given ``lang``) are used for the best quality.

        A GPU request automatically falls back to CPU when no GPU is usable
        (for example on a CPU-only Colab runtime).
        """
        if use_gpu and not _cuda_available():
            logger.warning('use_gpu=True, but no GPU is available; falling back to CPU')
            use_gpu = False

        kwargs = {}
        if self.det_model_dir is None and self.rec_model_dir is None:
            # Let PaddleOCR pick the default models for the language.
            kwargs['lang'] = lang
        else:
            # Explicit model directories take precedence; PaddleX requires
            # the model names declared inside those directories.
            kwargs['text_detection_model_name'] = _model_name_from_dir(self.det_model_dir)
            kwargs['text_recognition_model_name'] = _model_name_from_dir(self.rec_model_dir)

        if not use_gpu:
            # PaddlePaddle 3.3.x crashes on many CPUs when oneDNN (MKL-DNN) is
            # used with the PP-OCRv6 detection model:
            #   NotImplementedError: (Unimplemented)
            #   ConvertPirAttribute2RuntimeAttribute not support
            # Disable oneDNN for CPU inference to keep it reliable.
            kwargs['enable_mkldnn'] = False

        return PaddleOCR(
            device='gpu' if use_gpu else 'cpu',
            # Explicit model directories, when given, take precedence over
            # language/model selection.
            text_detection_model_dir=self.det_model_dir,
            text_recognition_model_dir=self.rec_model_dir,
            use_textline_orientation=True,
            # Video subtitles are upright frames; the document orientation
            # and unwarping modules are unnecessary and would only slow
            # down inference and require extra model downloads.
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            **kwargs,
        )

    def run_ocr(self, use_gpu: bool, lang: str, time_start: str, time_end: str,
                conf_threshold: int, use_fullframe: bool, brightness_threshold: int | None,
                similar_image_threshold: int | None, similar_pixel_threshold: int | None,
                frames_to_skip: int, crop_x: int | None, crop_y: int | None,
                crop_width: int | None, crop_height: int | None) -> None:
        # Initialize parameters
        self.lang = lang
        self.use_fullframe = use_fullframe
        conf_threshold_percent = float(conf_threshold / 100)
        ocr = self._create_ocr(use_gpu, lang)

        # Calculate frame ranges
        ocr_start = utils.get_frame_index(time_start, self.fps) if time_start else 0
        ocr_end = utils.get_frame_index(time_end, self.fps) if time_end else self.num_frames
        if ocr_end < ocr_start:
            raise ValueError('time_start is later than time_end')

        num_ocr_frames = ocr_end - ocr_start
        modulo = frames_to_skip + 1
        frames_to_process = num_ocr_frames // modulo

        # Initialize progress tracking
        start_time = time.time()
        frames_processed = 0

        # Calculate crop coordinates
        crop_coords = self._get_crop_coordinates(crop_x, crop_y, crop_width, crop_height)

        with Capture(self.path) as v:
            v.set(cv2.CAP_PROP_POS_FRAMES, ocr_start)
            prev_grey = None
            predicted_frames = None

            for i in range(num_ocr_frames):
                if i % modulo == 0:
                    frame = v.read()[1]
                    frame = self._process_frame(frame, crop_coords, brightness_threshold)

                    if similar_image_threshold and prev_grey is not None:
                        skip_frame = self._check_similar_frame(frame, prev_grey, similar_image_threshold,
                                                            similar_pixel_threshold, predicted_frames, i + ocr_start)
                        if skip_frame:
                            continue

                    pred_data = ocr.predict(frame)
                    if pred_data and pred_data[0].get('rec_texts'):
                        predicted_frames = PredictedFrames(i + ocr_start, pred_data, conf_threshold_percent)
                        self.pred_frames.append(predicted_frames)

                    # Update progress
                    frames_processed += 1
                    if frames_to_process:
                        progress = (frames_processed / frames_to_process) * 100
                        elapsed_time = time.time() - start_time
                        estimated_total = elapsed_time / (progress / 100)
                        remaining_time = estimated_total - elapsed_time
                        print(f"\rProgress: {progress:.1f}% - Estimated time remaining: {remaining_time:.1f}s")
                else:
                    v.read()

        print("\nOCR Processing completed!")

    def _get_crop_coordinates(self, crop_x: int | None, crop_y: int | None,
                              crop_width: int | None, crop_height: int | None):
        if all([crop_x, crop_y, crop_width, crop_height]):
            return (crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)
        return None

    def _process_frame(self, frame, crop_coords, brightness_threshold: int | None):
        if crop_coords and not self.use_fullframe:
            x1, y1, x2, y2 = crop_coords
            frame = frame[y1:y2, x1:x2]
        elif not self.use_fullframe:
            frame = frame[self.height // 3:, :]

        if brightness_threshold:
            frame = cv2.bitwise_and(frame, frame,
                mask=cv2.inRange(frame, (brightness_threshold,)*3, (255,)*3))
        return frame

    def _check_similar_frame(self, frame, prev_grey, similar_image_threshold: int,
                           similar_pixel_threshold: int | None, predicted_frames,
                           frame_index: int):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, absdiff = cv2.threshold(cv2.absdiff(prev_grey, grey),
                                 similar_pixel_threshold, 255, cv2.THRESH_BINARY)

        if (np.count_nonzero(absdiff) < similar_image_threshold and
            predicted_frames is not None):
            predicted_frames.end_index = frame_index
            return True
        return False

    def get_subtitles(self, sim_threshold: int) -> str:
        self._generate_subtitles(sim_threshold)
        return ''.join(
            f'{i}\n{utils.get_srt_timestamp(sub.index_start, self.fps)} --> '
            f'{utils.get_srt_timestamp(sub.index_end, self.fps)}\n{sub.text}\n\n'
            for i, sub in enumerate(self.pred_subs))

    def _generate_subtitles(self, sim_threshold: int) -> None:
        self.pred_subs = []
        max_frame_merge_diff = int(0.09 * self.fps)

        for frame in self.pred_frames:
            self._append_sub(PredictedSubtitle([frame], sim_threshold), max_frame_merge_diff)

        self.pred_subs = [sub for sub in self.pred_subs if sub.frames[0].lines]

    def _append_sub(self, sub: PredictedSubtitle, max_frame_merge_diff: int) -> None:
        if not sub.frames:
            return

        if (self.pred_subs and self.pred_subs[-1].frames[0].lines and
            sub.index_start - self.pred_subs[-1].index_end <= max_frame_merge_diff and
            self.pred_subs[-1].is_similar_to(sub)):

            last_sub = self.pred_subs.pop()
            sub = PredictedSubtitle(last_sub.frames + sub.frames, sub.sim_threshold)

        self.pred_subs.append(sub)
