from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class FrameStage(ABC):
    @abstractmethod
    def __call__(self, frame):
        """Takes a frame and returns a new frame"""


class Pipeline:
    def __init__(self, stages):
        self.stages = list(stages)

    def process(self, frame):
        for s in self.stages:
            frame = s(frame)
        return frame