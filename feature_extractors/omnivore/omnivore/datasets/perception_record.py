from .video_record import VideoRecord

class PerceptionVideoRecord(VideoRecord):
    def __init__(self, tup):
        self._index = str(tup[0])
        self._series = tup[1]
    
    @property
    def untrimmed_video_name(self):
        return self._series['video_id']

    @property
    def start_frame(self):
        return int(self._series['start_frame'])

    @property
    def end_frame(self):
        return int(self._series['stop_frame'])

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame

    @property
    def label(self):
        # Dummy for feature extraction
        return {'verb': -1, 'noun': -1}

    @property
    def metadata(self):
        return {'narration_id': self._index}
