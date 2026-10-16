from __future__ import annotations

from .video import Video


def get_subtitles(
        video_path: str, lang: str = 'ch', time_start: str = '0:00', time_end: str = '',
        conf_threshold: int = 75, sim_threshold: int = 80, use_fullframe: bool = False,
        det_model_dir: str | None = None, rec_model_dir: str | None = None, use_gpu: bool = False,
        brightness_threshold: int | None = None, similar_image_threshold: int | None = 100,
        similar_pixel_threshold: int | None = 25, frames_to_skip: int = 1,
        crop_x: int | None = None, crop_y: int | None = None,
        crop_width: int | None = None, crop_height: int | None = None) -> str:
    """Run OCR on the video and return the subtitles in SRT format."""
    v = Video(video_path, det_model_dir, rec_model_dir)
    v.run_ocr(use_gpu, lang, time_start, time_end, conf_threshold, use_fullframe, brightness_threshold,
              similar_image_threshold, similar_pixel_threshold, frames_to_skip,
              crop_x, crop_y, crop_width, crop_height)
    return v.get_subtitles(sim_threshold)


def save_subtitles_to_file(
        video_path: str, file_path: str = 'subtitle.srt', lang: str = 'ch',
        time_start: str = '0:00', time_end: str = '', conf_threshold: int = 75,
        sim_threshold: int = 80, use_fullframe: bool = False,
        det_model_dir: str | None = None, rec_model_dir: str | None = None, use_gpu: bool = False,
        brightness_threshold: int | None = None, similar_image_threshold: int | None = 100,
        similar_pixel_threshold: int | None = 25, frames_to_skip: int = 1,
        crop_x: int | None = None, crop_y: int | None = None,
        crop_width: int | None = None, crop_height: int | None = None) -> None:
    """Run OCR on the video and write the subtitles to ``file_path`` in SRT format."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(get_subtitles(
            video_path, lang, time_start, time_end, conf_threshold,
            sim_threshold, use_fullframe, det_model_dir, rec_model_dir, use_gpu,
            brightness_threshold, similar_image_threshold, similar_pixel_threshold,
            frames_to_skip, crop_x, crop_y, crop_width, crop_height))
