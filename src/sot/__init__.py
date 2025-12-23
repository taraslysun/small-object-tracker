"""Small Object Tracker (SOT) - Pipeline-based tracking system."""
from .sot import SOT
from .pipeline import Pipeline, PipelineStep
from .steps import (
    DetectionStep,
    TrackingStep,
    VisualizationStep,
    ROIFilterStep,
    OpticalFlowStep,
    BackgroundSeparationStep,
    FramePreprocessingStep,
    SIFTFeatureStep,
    SIFTMatchingStep,
    SIFTEnhancedDetectionStep,
    SIFTVisualizationStep
)

__all__ = [
    'SOT',
    'Pipeline',
    'PipelineStep',
    'DetectionStep',
    'TrackingStep',
    'VisualizationStep',
    'ROIFilterStep',
    'OpticalFlowStep',
    'BackgroundSeparationStep',
    'FramePreprocessingStep',
    'SIFTFeatureStep',
    'SIFTMatchingStep',
    'SIFTEnhancedDetectionStep',
    'SIFTVisualizationStep',
]

