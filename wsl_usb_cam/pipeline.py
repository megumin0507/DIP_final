from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class FrameStage(ABC):
    @abstractmethod
    def __call__(self, frame):
        """Takes a frame and returns a new frame"""
        return frame


class Pipeline:
    def __init__(self, stages):
        self.stages = []
        self.stage_names = []

        for item in stages:
            if isinstance(item, tuple) and len(item) == 2:
                name, stage = item
            else:
                stage = item
                name = stage.__class__.__name__
            
            self.stage_names.append(str(name))
            self.stages.append(stage)
        
        logger.info(f"Pipeline stages: {list(zip(self.stage_names, self.stages))}")


    def process(self, frame, enabled_stages=None):
        for name, s in zip(self.stage_names, self.stages):
            if enabled_stages is not None:
                flag = enabled_stages.get(name)
                if flag is None:
                    flag = enabled_stages.get(name.lower())
                if flag is False:
                    logger.debug(f"Skipping stage {name}")
                    continue
            frame = s(frame)
        return frame