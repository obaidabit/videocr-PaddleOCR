from __future__ import annotations
from typing import List
import cv2
import numpy as np
import time
from . import utils
from .models import PredictedFrames, PredictedSubtitle
from .opencv_adapter import Capture
from paddleocr import PaddleOCR
import logging

class Video:
    def __init__(self, path: str, det_model_dir: str, rec_model_dir: str):
        self.path = path
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        with Capture(path) as v:
            self.num_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = v.get(cv2.CAP_PROP_FPS)
            self.height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pred_frames = []
        self.pred_subs = []

    def run_ocr(self, use_gpu: bool, lang: str, time_start: str, time_end: str,
                conf_threshold: int, use_fullframe: bool, brightness_threshold: int, 
                similar_image_threshold: int, similar_pixel_threshold: int, frames_to_skip: int,
                crop_x: int, crop_y: int, crop_width: int, crop_height: int) -> None:
        
        # Disable PaddleOCR logging
        logging.getLogger("paddleocr").setLevel(logging.ERROR)
        
        # Initialize parameters
        self.lang = lang
        self.use_fullframe = use_fullframe
        conf_threshold_percent = float(conf_threshold/100)
        ocr = PaddleOCR(lang=self.lang, rec_model_dir=self.rec_model_dir, 
                        det_model_dir=self.det_model_dir, use_gpu=use_gpu, show_log=False, use_angle_cls=True)

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

                    pred_data = ocr.ocr(frame)
                    if pred_data and pred_data[0]:
                        predicted_frames = PredictedFrames(i + ocr_start, pred_data, conf_threshold_percent)
                        self.pred_frames.append(predicted_frames)
                    
                    # Update progress
                    frames_processed += 1
                    progress = (frames_processed / frames_to_process) * 100
                    elapsed_time = time.time() - start_time
                    estimated_total = elapsed_time / (progress / 100)
                    remaining_time = estimated_total - elapsed_time
                    
                    print(f"\rProgress: {progress:.1f}% - Estimated time remaining: {remaining_time:.1f}s")
                else:
                    v.read()
        
        print("\nOCR Processing completed!")

    def _get_crop_coordinates(self, crop_x: int, crop_y: int, crop_width: int, crop_height: int):
        if all([crop_x, crop_y, crop_width, crop_height]):
            return (crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)
        return None

    def _process_frame(self, frame, crop_coords, brightness_threshold: int):
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
                           similar_pixel_threshold: int, predicted_frames, frame_index: int):
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
        if not hasattr(self, 'pred_frames'):
            raise AttributeError('Please call self.run_ocr() first to perform ocr on frames')
            
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