"""
EmbryoMetadata Core Class - Simple and Focused
Inherits business logic from UnifiedEmbryoManager, handles file I/O and initialization.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

# Module imports
from utils.base_file_handler import BaseFileHandler
from utils.entity_id_tracker import EntityIDTracker
from utils.parsing_utils import (
    get_entity_type, 
    extract_embryo_id,
    extract_frame_number
)
from metadata.schema_manager import SchemaManager
from annotations.unified_managers import UnifiedEmbryoManager
from annotations.annotation_batch import AnnotationBatch, EmbryoQuery


class EmbryoMetadata(BaseFileHandler, UnifiedEmbryoManager):
    """
    Main embryo metadata class.
    
    Core functionality:
    - File I/O and persistence
    - SAM annotation integration
    - Data validation and consistency
    - Embryo/snip hierarchy management
    - Batch processing with validation and error handling
    
    Business logic inherited from UnifiedEmbryoManager.
    """
    
    def __init__(self, sam_annotation_path: Union[str, Path], 
                 embryo_metadata_path: Optional[Union[str, Path]] = None,
                 gen_if_no_file: bool = False, 
                 verbose: bool = True,
                 schema_path=None):
        
        # Setup paths
        self.sam_annotation_path = Path(sam_annotation_path)
        if not self.sam_annotation_path.exists():
            raise FileNotFoundError(f"SAM annotation file not found: {sam_annotation_path}")
        
        if embryo_metadata_path is None:
            embryo_metadata_path = self.sam_annotation_path.with_name(
                self.sam_annotation_path.stem + "_embryo_metadata.json"
            )
        
        # Initialize file handler
        super().__init__(embryo_metadata_path, verbose=verbose)
        
        # Configuration and schema
        self.config = {"default_author": "unknown"}
        self.schema_manager = SchemaManager(schema_path) if schema_path else SchemaManager()
        self._auto_validate = True  # Default validation enabled
        
        # Load SAM annotations
        self.sam_annotations = self.load_json(self.sam_annotation_path)
        
        # Load or create metadata
        if self.filepath.exists():
            self.data = self.load_json()
            self._validate_entity_tracking()
        elif gen_if_no_file:
            self.data = self._create_from_sam()
            self._initialize_entity_tracking()
        else:
            raise FileNotFoundError(f"Metadata not found: {embryo_metadata_path}")
        
        # Initialize lookup caches
        self._build_caches()
    
    def _validate_entity_tracking(self):
        """Validate entity hierarchy and update tracking section."""
        # Extract current entities
        current_entities = EntityIDTracker.extract_entities(self.data)
        
        # Validate hierarchy (using correct method signature)
        EntityIDTracker.validate_hierarchy(current_entities, check_hierarchy=True)
        
        # Update entity_tracking section
        if "entity_tracking" not in self.data:
            self.data["entity_tracking"] = {}
        
        self.data["entity_tracking"]["metadata"] = {
            entity_type: list(ids) for entity_type, ids in current_entities.items()
        }
        
        if self.verbose:
            counts = EntityIDTracker.get_counts(current_entities)
            print(f"📊 Entities: {counts}")
    
    def _initialize_entity_tracking(self):
        """Initialize entity tracking for new metadata."""
        # Extract from SAM annotations
        sam_entities = EntityIDTracker.extract_entities(self.sam_annotations)
        
        # Extract from created metadata
        metadata_entities = EntityIDTracker.extract_entities(self.data)
        
        # Find missing entities
        missing = EntityIDTracker.compare_entities(sam_entities, metadata_entities)
        
        # Store tracking info
        self.data["entity_tracking"] = {
            "sam_source": {entity_type: list(ids) for entity_type, ids in sam_entities.items()},
            "metadata": {entity_type: list(ids) for entity_type, ids in metadata_entities.items()},
            "missing": {entity_type: list(ids) for entity_type, ids in missing.items()}
        }
        
        if self.verbose and any(missing.values()):
            print(f"⚠️ Missing entities from SAM: {EntityIDTracker.get_counts(missing)}")
    
    def check_sam_consistency(self) -> Dict:
        """Check consistency with original SAM file."""
        expected_sam = self.data.get("file_info", {}).get("source_sam_filename")
        current_sam = Path(self.sam_annotation_path).name
        
        if expected_sam and expected_sam != current_sam:
            raise ValueError(f"SAM file mismatch: expected {expected_sam}, got {current_sam}")
        
        # Compare entities
        sam_entities = EntityIDTracker.extract_entities(self.sam_annotations)
        metadata_entities = EntityIDTracker.extract_entities(self.data)
        missing = EntityIDTracker.compare_entities(sam_entities, metadata_entities)
        
        return {"missing_from_metadata": missing, "consistent": not any(missing.values())}
    
    def _create_from_sam(self) -> Dict:
        """Create metadata structure from SAM annotations."""
        embryos = {}
        
        # Extract embryo structure from SAM
        for exp_id, exp_data in self.sam_annotations.get("experiments", {}).items():
            for video_id, video_data in exp_data.get("videos", {}).items():
                for embryo_id in video_data.get("embryo_ids", []):
                    if embryo_id not in embryos:
                        embryos[embryo_id] = {
                            "genotype": None,
                            "treatments": {},
                            "flags": {},
                            "notes": "",
                            "metadata": {"created": self.get_timestamp()},
                            "snips": {}
                        }
                    
                    # Add snips for this embryo
                    for image_id, image_data in video_data.get("images", {}).items():
                        if embryo_id in image_data.get("embryos", {}):
                            snip_id = image_data["embryos"][embryo_id].get("snip_id")
                            if snip_id:
                                embryos[embryo_id]["snips"][snip_id] = {"flags": []}
        
        return {
            "file_info": {
                "version": "1.0",
                "created": self.get_timestamp(),
                "source_sam": str(self.sam_annotation_path),
                "source_sam_filename": self.sam_annotation_path.name
            },
            "embryos": embryos,
            "entity_tracking": {}
        }
    
    def _build_caches(self):
        """Build lookup caches for performance."""
        self._snip_to_embryo = {}
        
        for embryo_id, embryo_data in self.data["embryos"].items():
            for snip_id in embryo_data.get("snips", {}):
                self._snip_to_embryo[snip_id] = embryo_id
    
    def get_embryo_id_from_snip(self, snip_id: str) -> Optional[str]:
        """Get embryo ID from snip ID."""
        return self._snip_to_embryo.get(snip_id)
    
    def get_available_snips(self, embryo_id: Optional[str] = None) -> List[str]:
        """Get available snip IDs."""
        if embryo_id:
            return list(self.data["embryos"].get(embryo_id, {}).get("snips", {}).keys())
        return list(self._snip_to_embryo.keys())
    
    def get_snip_data(self, snip_id: str) -> Optional[Dict]:
        """Get snip data."""
        embryo_id = self.get_embryo_id_from_snip(snip_id)
        if embryo_id:
            return self.data["embryos"][embryo_id]["snips"].get(snip_id)
        return None
    
    def save(self, backup: bool = True):
        """Save with entity validation using embedded tracker approach."""
        # EntityIDTracker is a PURE CONTAINER - use static methods for embedded tracking
        try:
            # Update embedded entity tracker in the embryo metadata
            self.data = EntityIDTracker.update_entity_tracker(
                self.data,
                pipeline_step="module_3_embryo_metadata"
            )
            
            # Update file info
            if "file_info" not in self.data:
                self.data["file_info"] = {}
            self.data["file_info"]["last_updated"] = self.get_timestamp()
            
            # Save using inherited atomic save
            self.save_json(self.data, create_backup=backup)
            
            if self.verbose:
                embryo_count = len(self.data["embryos"])
                snip_count = len(self._snip_to_embryo)
                print(f"💾 Saved: {embryo_count} embryos, {snip_count} snips (validated)")
                
        except Exception as e:
            if self.verbose:
                print(f"❌ Save failed: {e}")
            raise
    
    def reload(self):
        """Reload from file."""
        self.data = self.load_json()
        self._build_caches()
        if self.verbose:
            print("🔄 Reloaded metadata")
    
    def get_entity_counts(self) -> Dict[str, int]:
        """Get counts of all entity types by parsing IDs."""
        # Extract all entities and get counts
        current_entities = EntityIDTracker.extract_entities(self.data)
        return EntityIDTracker.get_counts(current_entities)
    
    @property
    def embryo_count(self) -> int:
        """Number of embryos."""
        return len(self.data["embryos"])
    
    @property
    def snip_count(self) -> int:
        """Number of snips."""
        return len(self._snip_to_embryo)
    
    def get_genotype(self, embryo_id: str) -> Optional[Dict]:
        """Get genotype for embryo."""
        return self.data["embryos"].get(embryo_id, {}).get("genotype")
    
    def get_phenotypes(self, snip_id: str) -> List[Dict]:
        """Get phenotypes for snip."""
        snip_data = self.get_snip_data(snip_id)
        if snip_data and "phenotype" in snip_data:
            return [snip_data["phenotype"]]  # Single phenotype per snip in this implementation
        return []
    
    def get_flags(self, entity_id: str, level: str = "auto") -> List[Dict]:
        """Get flags for entity."""
        if level == "auto":
            level = get_entity_type(entity_id)
        
        if level == "snip":
            snip_data = self.get_snip_data(entity_id)
            return snip_data.get("flags", []) if snip_data else []
        elif level == "embryo":
            embryo_data = self.data["embryos"].get(entity_id, {})
            return list(embryo_data.get("flags", {}).values())
        
        return []
    
    def query(self) -> EmbryoQuery:
        """Create new query builder."""
        return EmbryoQuery(self)
    
    def list_snips_by_phenotype(self, phenotype: str) -> List[str]:
        """List all snips with specific phenotype."""
        snips = []
        for embryo_id, embryo_data in self.data["embryos"].items():
            for snip_id, snip_data in embryo_data["snips"].items():
                if "phenotype" in snip_data and snip_data["phenotype"]["value"] == phenotype:
                    snips.append(snip_id)
        return snips
    
    def get_phenotype_statistics(self) -> Dict:
        """Get phenotype distribution statistics."""
        stats = {}
        for embryo_data in self.data["embryos"].values():
            for snip_data in embryo_data["snips"].values():
                if "phenotype" in snip_data:
                    value = snip_data["phenotype"]["value"]
                    stats[value] = stats.get(value, 0) + 1
        return stats
    
    def get_genotype_statistics(self) -> Dict[str, int]:
        """Get genotype distribution statistics."""
        stats = {}
        for embryo_data in self.data["embryos"].values():
            if embryo_data.get("genotype"):
                gene = embryo_data["genotype"]["value"]
                stats[gene] = stats.get(gene, 0) + 1
        return stats
    
    def get_summary(self) -> Dict:
        """Get overall metadata summary."""
        entity_counts = self.get_entity_counts()
        phenotype_stats = self.get_phenotype_statistics()
        genotype_stats = self.get_genotype_statistics()
        
        return {
            "entity_counts": entity_counts,
            "phenotype_stats": phenotype_stats,
            "genotype_stats": genotype_stats,
            "auto_validate": self._auto_validate,
            "file_info": self.data.get("file_info", {})
        }
    
    def auto_validate(self, enabled: bool = True):
        """Enable/disable automatic validation after changes."""
        self._auto_validate = enabled
        if self.verbose:
            status = "enabled" if enabled else "disabled"
            print(f"🔧 Auto-validation {status}")
