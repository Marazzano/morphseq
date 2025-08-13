"""
General Annotation Utilities Module

Provides general-purpose annotation processing utilities and shared functionality.
Embryo-specific metadata is handled in the metadata.embryo_metadata submodule.
"""

from .embryo_metadata import EmbryoMetadata
from .unified_managers import UnifiedEmbryoManager
from .annotation_batch import AnnotationBatch, EmbryoQuery

__all__ = [
    "EmbryoMetadata",
    "UnifiedEmbryoManager", 
    "AnnotationBatch",
    "EmbryoQuery"
]

__all__ = []
