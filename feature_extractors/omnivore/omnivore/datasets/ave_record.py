from .video_record import VideoRecord
from datetime import timedelta
import time


def timestamp_to_sec(timestamp):
    x = time.strptime(timestamp, '%H:%M:%S.%f')
    sec = float(timedelta(hours=x.tm_hour,
                          minutes=x.tm_min,
                          seconds=x.tm_sec).total_seconds()) + float(
                                  timestamp.split('.')[-1][:2]) / 100
    return sec


class AVEVideoRecord(VideoRecord):
    def __init__(self, tup):
        self._index = str(tup[0])
        self._series = tup[1]
        
    @property
    def untrimmed_video_name(self):
        return self._series['video_id']
    
    @property
    def fps(self):
        return self._series['fps']
    
    @property
    def start_frame(self):
        return int(round(timestamp_to_sec(self._series['start_timestamp']) * self.fps))

    @property
    def end_frame(self):
        return int(round(timestamp_to_sec(self._series['stop_timestamp']) * self.fps))

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame

    @property
    def label(self):
        # Dummy for feature extraction
        return {'class': -1}

    @property
    def metadata(self):
        return {'narration_id': self._index}
