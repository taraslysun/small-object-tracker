"""Pipeline architecture for processing steps."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np


class PipelineStep(ABC):
    """Base class for pipeline steps."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.enabled = True
    
    @abstractmethod
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a frame and update context.
        
        Args:
            frame: Input frame
            context: Shared context dictionary containing results from previous steps
        
        Returns:
            Updated context dictionary
        """
        pass
    
    def reset(self):
        pass


class Pipeline:
    """Pipeline for chaining processing steps."""
    
    def __init__(self):
        self.steps: List[PipelineStep] = []
    
    def add_step(self, step: PipelineStep):
        self.steps.append(step)
        return self
    
    def remove_step(self, name: str):
        self.steps = [s for s in self.steps if s.name != name]
        return self
    
    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process frame through all pipeline steps.
        
        Args:
            frame: Input frame
            context: Initial context (optional)
        
        Returns:
            Final context with all results
        """
        if context is None:
            context = {}
        
        context['frame'] = frame
        
        for step in self.steps:
            if step.enabled:
                context = step.process(frame, context)
        
        return context
    
    def reset(self):
        """Reset all steps."""
        for step in self.steps:
            step.reset()
    
    def get_step(self, name: str) -> PipelineStep:
        """Get step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None
    
    def list_steps(self) -> List[str]:
        """List all step names."""
        return [s.name for s in self.steps]

