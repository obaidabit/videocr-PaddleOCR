# videocr

Extract hardcoded (burned-in) subtitles from videos using the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) OCR engine with Python. A Colab notebook for installing and running this library is included for convenience:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obaidabit/videocr-PaddleOCR/blob/master/videocr_PaddleOCR.ipynb)

```python
# example.py

from videocr import save_subtitles_to_file

if __name__ == '__main__':
    save_subtitles_to_file('example_cropped.mp4', 'example.srt', lang='ch', time_start='7:10', time_end='7:34',
     sim_threshold=80, conf_threshold=75, use_fullframe=True,
     brightness_threshold=210, similar_image_threshold=1000, frames_to_skip=1)
```

`$ python3 example.py`

example.srt:

``` 
0
00:07:10,000 --> 00:07:10,083
商城......现在没什么东西

1
00:07:10,416 --> 00:07:12,000
这边是战斗辅助系统

2
00:07:13,083 --> 00:07:14,500
要进去才能了解了

3
00:07:15,083 --> 00:07:15,916
没问题了吧

4
00:07:16,333 --> 00:07:17,166
我们准备登录

5
00:07:18,416 --> 00:07:21,083
啊对了， 登录没有服务器的选择么

6
00:07:21,333 --> 00:07:25,000
没有本游戏所有玩家， 都在个服务器内

7
00:07:25,833 --> 00:07:28,833
刺激了， 这么多玩家居然都不分流的么

8
00:07:29,500 --> 00:07:31,083
那......现在登录吗？

9
00:07:31,166 --> 00:07:32,416
好，登录吧！
```

## Install prerequisites

- Python **3.10 - 3.13** (3.13 recommended). PaddlePaddle does not ship wheels for Python 3.14 yet.
- `paddlepaddle` (CPU) or `paddlepaddle-gpu` (GPU)

`videocr` uses the PaddleOCR 3.x API. Install the PaddlePaddle framework that matches your platform first:

```bash
# CPU (from PyPI)
python -m pip install paddlepaddle

# GPU (official Paddle index; cu126 build needs driver >= 550, use cu118 for older drivers)
python -m pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

See https://www.paddlepaddle.org.cn/en/install/quick for the full list of framework builds (Windows, macOS, different CUDA versions, ...).

## Installation

`pip install git+https://github.com/obaidabit/videocr-PaddleOCR.git`

Alternatively for development:
1. Clone this repo
2. From the root directory of this repository run `python -m pip install -e ".[cpu]"` (or `python -m pip install -e .` if you installed PaddlePaddle yourself)

The OCR models (PP-OCRv5/v6) are downloaded automatically on the first run.

## Performance

The OCR process can be very slow on CPU. Running with `paddlepaddle-gpu` is recommended if you have a CUDA GPU.

By default PaddleOCR's server-grade `PP-OCRv6_medium` models (or the prebuilt models matching `lang`) are used, which give the best accuracy but are much slower on CPU (~5x slower than the mobile `PP-OCRv6_small` tier). A GPU is recommended — e.g. the included Colab notebook. Any other prebuilt model can be used instead by downloading it and passing `det_model_dir` / `rec_model_dir`.

## Tips

To shorten the amount of time it takes to perform OCR on each frame, you can use the `crop_x`, `crop_y`, `crop_width`, `crop_height` params to crop out only the areas of the videos where the subtitles appear. When cropping, leave a bit of buffer space above and below the text to ensure accurate readings.

### Quick Configuration Cheatsheet

|| More Speed | More Accuracy | Notes
-|------------|---------------|--------
Input Video Quality       | Use lower quality           | Use higher quality  | Performance impact of using higher resolution video can be reduced with cropping
`frames_to_skip`          | Higher number               | Lower number        |
`brightness_threshold`    | Higher threshold            | N/A                 | A brightness threshold can help speed up the OCR process by filtering out dark frames. In certain circumstances such as when subtitles are white and against a bright background, it may also help with accuracy.


## API

1. Return subtitle string in SRT format
    ```python
    get_subtitles(
        video_path: str, lang='ch', time_start='0:00', time_end='',
        conf_threshold=75, sim_threshold=80, use_fullframe=False,
        det_model_dir=None, rec_model_dir=None, use_gpu=False,
        brightness_threshold=None, similar_image_threshold=100, similar_pixel_threshold=25, frames_to_skip=1,
        crop_x=None, crop_y=None, crop_width=None, crop_height=None)
    ```

2. Write subtitles to `file_path`
    ```python
    save_subtitles_to_file(
        video_path: str, file_path='subtitle.srt', lang='ch', time_start='0:00', time_end='', 
        conf_threshold=75, sim_threshold=80, use_fullframe=False,
        det_model_dir=None, rec_model_dir=None, use_gpu=False,
        brightness_threshold=None, similar_image_threshold=100, similar_pixel_threshold=25, frames_to_skip=1,
        crop_x=None, crop_y=None, crop_width=None, crop_height=None)
    ```

### Parameters

- `lang`

  The language of the subtitles (`ch`, `en`, `japan`, `korean`, `chinese_cht`, ...). See the [PaddleOCR 3.x docs](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html) for the full list of supported languages and their abbreviations.

- `conf_threshold`

  Confidence threshold for word predictions. Words with lower confidence than this value will be discarded. The default value `75` is fine for most cases. 

  Make it closer to 0 if you get too few words in each line, or make it closer to 100 if there are too many excess words in each line.

- `sim_threshold`

  Similarity threshold for subtitle lines. Subtitle lines with larger [Levenshtein](https://en.wikipedia.org/wiki/Levenshtein_distance) ratios than this threshold will be merged together. The default value `80` is fine for most cases.

  Make it closer to 0 if you get too many duplicated subtitle lines, or make it closer to 100 if you get too few subtitle lines.

- `time_start` and `time_end`

  Extract subtitles from only a clip of the video. The subtitle timestamps are still calculated according to the full video length.

- `use_fullframe`

  By default, the specified cropped area is used for OCR or if a crop is not specified, then the bottom third of the frame will be used. By setting this value to `True` the entire frame will be used.

- `crop_x`, `crop_y`, `crop_width`, `crop_height`

  Specifies the bounding area in pixels for the portion of the frame that will be used for OCR. See image below for example:
  ![image](https://user-images.githubusercontent.com/8058852/226201081-f4ec9a23-4cc8-48d4-b15c-6ea2ac29ae93.png)

- `det_model_dir`

  the text detection inference model folder (passed to PaddleOCR as `text_detection_model_dir`; its model name is read automatically from `inference.yml`). 1. `None`: the built-in model is downloaded automatically to `~/.paddlex/official_models` on first run; 2. the path of a specific inference model folder containing the model and params files.

  See the PaddleOCR model list for prebuilt models: https://www.paddleocr.ai/latest/en/version3.x/model_list.html.

- `rec_model_dir`
  
  the text recognition inference model folder (passed to PaddleOCR as `text_recognition_model_dir`; its model name is read automatically from `inference.yml`). 1. `None`: the built-in model is downloaded automatically to `~/.paddlex/official_models` on first run; 2. the path of a specific inference model folder containing the model and params files.
  
  See the PaddleOCR model list for prebuilt models: https://www.paddleocr.ai/latest/en/version3.x/model_list.html.

- `use_gpu`

  Set to `True` if performing OCR with a GPU (requires the `paddlepaddle-gpu` package, see prerequisites). Internally maps to PaddleOCR's `device` parameter.

- `brightness_threshold`
  
  If set, pixels whose brightness are less than the threshold will be blackened out. Valid brightness values range from 0 (black) to 255 (white). This can help improve accuracy when performing OCR on videos with white subtitles.

- `similar_image_threshold`

  The number of non-similar pixels there can be before the program considers 2 consecutive frames to be different. If a frame is not different from the previous frame, then the OCR result from the previous frame will be used (which can save a lot of time depending on how fast each OCR inference takes).

- `similar_pixel_threshold`

  Brightness threshold from 0-255 used with the `similar_image_threshold` to determine if 2 consecutive frames are different. If the difference between 2 pixels exceeds the threshold, then they will be considered non-similar.

- `frames_to_skip`

  The number of frames to skip before sampling a frame for OCR. Keep in mind the fps of the input video before increasing.

## TODO
- [ ] parallel processing
- [ ] handle multiple lines of text in the same frame
- [ ] publish to pypi
- [ ] commandline interface
- [ ] user-friendly application for non-devs
