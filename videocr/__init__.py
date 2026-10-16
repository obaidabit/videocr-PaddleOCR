from importlib.metadata import PackageNotFoundError, version

from .api import get_subtitles, save_subtitles_to_file

try:
    __version__ = version('videocr')
except PackageNotFoundError:  # package is not installed
    __version__ = 'unknown'

__all__ = ['get_subtitles', 'save_subtitles_to_file', '__version__']
