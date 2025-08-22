"""
AnnotationBatch: Inheritance-based batch annotation system

Provides a temporary workspace for safely manipulating embryo annotations
before applying them to the persistent EmbryoMetadata store.
"""

from typing import Dict, List, Optional, Any
from .embryo_metadata import EmbryoMetadata


class AnnotationBatch(EmbryoMetadata):
    """
    Temporary workspace inheriting all EmbryoMetadata functionality.
    
    Uses inheritance to reuse all validation logic while maintaining
    complete isolation from the main data store.
    """
    
    def __init__(self, data_structure: Dict, author: str, validate: bool = True):
        """
        Initialize batch workspace without calling parent constructor.
        
        Args:
            data_structure: Skeleton structure from metadata.initialize_batch()
            author: Default author for all operations
            validate: Enable/disable validation (same as parent)
        """
        if not author:
            raise ValueError("Author required for AnnotationBatch")
        
        # Manual attribute setup (skip super().__init__ to avoid file I/O)
        self.data = data_structure
        self.validate = validate
        self.author = author
        
        # No file paths for batch (temporary workspace)
        self.sam2_path = None
        self.annotations_path = None
        self.file_handler = None
        
        # Verify required attributes exist for parent methods
        self._verify_contract()
    
    def _verify_contract(self):
        """Ensure batch has required attributes for parent methods."""
        required_attrs = {
            "data": "Annotation data structure",
            "validate": "Validation toggle",
            "author": "Default author for operations"
        }
        
        missing = []
        for attr, description in required_attrs.items():
            if not hasattr(self, attr):
                missing.append(f"{attr} ({description})")
        
        if missing:
            raise AttributeError(
                f"AnnotationBatch missing required attributes:\n" + 
                "\n".join(f"  - {item}" for item in missing)
            )
    
    def add_phenotype(self, phenotype: str, author: str = None, embryo_id: str = None, 
                     target: str = None, snip_ids: List[str] = None, 
                     overwrite_dead: bool = False) -> Dict:
        """
        Add phenotype using inherited method with batch author default.
        
        Args:
            phenotype: Phenotype value
            author: Author (defaults to batch author if not provided)
            embryo_id: Target embryo ID (embryo approach)
            target: Target specification (embryo approach)
            snip_ids: List of snip IDs (snip approach)
            overwrite_dead: Whether to overwrite DEAD frames
            
        Returns:
            Dict with operation details
        """
        return super().add_phenotype(
            phenotype,
            author or self.author,
            embryo_id=embryo_id,
            target=target,
            snip_ids=snip_ids,
            overwrite_dead=overwrite_dead
        )
    
    def add_genotype(self, gene: str, author: str = None, embryo_id: str = None,
                    allele: Optional[str] = None, zygosity: str = "unknown",
                    overwrite: bool = False) -> Dict:
        """
        Add genotype using inherited method with batch author default.
        
        Args:
            gene: Gene name
            author: Author (defaults to batch author if not provided)
            embryo_id: Target embryo ID
            allele: Optional allele specification
            zygosity: Zygosity specification
            overwrite: Whether to overwrite existing genotype
            
        Returns:
            Dict with operation details
        """
        return super().add_genotype(
            gene,
            author or self.author,
            embryo_id,
            allele=allele,
            zygosity=zygosity,
            overwrite=overwrite
        )
    
    def add_treatment(self, treatment: str, author: str = None, embryo_id: str = None,
                     temperature_celsius: Optional[float] = None,
                     concentration: Optional[str] = None,
                     notes: Optional[str] = None) -> Dict:
        """
        Add treatment using inherited method with batch author default.
        
        Args:
            treatment: Treatment name
            author: Author (defaults to batch author if not provided)
            embryo_id: Target embryo ID
            temperature_celsius: Optional temperature
            concentration: Optional concentration
            notes: Optional notes
            
        Returns:
            Dict with operation details
        """
        return super().add_treatment(
            treatment,
            author or self.author,
            embryo_id,
            temperature_celsius=temperature_celsius,
            concentration=concentration,
            notes=notes
        )
    
    def preview(self, limit: int = 10) -> str:
        """
        Generate human-readable summary of batch contents.
        
        Args:
            limit: Maximum number of embryos to show in detail
            
        Returns:
            Formatted preview string
        """
        lines = [f"AnnotationBatch (Author: {self.author})", ""]
        
        embryo_count = 0
        total_phenotypes = 0
        total_genotypes = 0
        total_treatments = 0
        
        for embryo_id, embryo_data in self.data["embryos"].items():
            if embryo_count >= limit:
                lines.append(f"... and {len(self.data['embryos']) - limit} more embryos")
                break
            
            # Collect statistics for this embryo
            stats_parts = []
            
            # Genotype info
            if embryo_data.get("genotype"):
                g = embryo_data["genotype"]
                stats_parts.append(f"🧬 {g['gene']} ({g['zygosity']})")
                total_genotypes += 1
            
            # Phenotype info
            phenotype_counts = {}
            for snip_data in embryo_data.get("snips", {}).values():
                for phenotype in snip_data.get("phenotypes", []):
                    pheno_value = phenotype["value"]
                    phenotype_counts[pheno_value] = phenotype_counts.get(pheno_value, 0) + 1
                    total_phenotypes += 1
            
            if phenotype_counts:
                # Show top 3 phenotypes with counts
                top_phenotypes = sorted(phenotype_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                pheno_summary = ", ".join([f"{pheno}:{count}" for pheno, count in top_phenotypes])
                total_snip_count = sum(phenotype_counts.values())
                stats_parts.append(f"🔬 {total_snip_count} phenotypes ({pheno_summary})")
            
            # Treatment info
            treatment_count = len(embryo_data.get("treatments", []))
            if treatment_count > 0:
                stats_parts.append(f"💊 {treatment_count} treatments")
                total_treatments += treatment_count
            
            # Display embryo summary
            if stats_parts:
                lines.append(f"📋 {embryo_id}: {' | '.join(stats_parts)}")
            else:
                lines.append(f"📋 {embryo_id}: (no annotations)")
            
            embryo_count += 1
        
        # Add overall summary
        lines.append("")
        lines.append(f"Summary: {len(self.data['embryos'])} embryos, {total_genotypes} genotypes, "
                    f"{total_phenotypes} phenotypes, {total_treatments} treatments")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict:
        """Get statistics about batch contents."""
        stats = super().get_stats()
        stats["batch_author"] = self.author
        stats["is_batch"] = True
        return stats
    
    def save(self):
        """Override save to prevent accidental saves of batch data."""
        raise NotImplementedError("AnnotationBatch cannot be saved directly. Use metadata.apply_batch() instead.")


# Add batch initialization method to EmbryoMetadata
def initialize_batch(self, mode: str = "skeleton", author: str = None) -> AnnotationBatch:
    """
    Create an AnnotationBatch from current metadata.
    
    Args:
        mode: Initialization mode
            - "skeleton": Empty annotations, preserve embryo/snip structure
            - "copy": Full copy of current annotations
        author: Required batch author
        
    Returns:
        AnnotationBatch instance
    """
    if not author:
        raise ValueError("Author required for batch initialization")
    
    if mode == "skeleton":
        # Create empty structure preserving embryo/snip organization
        batch_data = {
            "metadata": {
                "source_sam2": self.data["metadata"].get("source_sam2"),
                "created": self.data["metadata"].get("created"),
                "version": "batch_skeleton",
                "batch_mode": mode
            },
            "embryos": {}
        }
        
        # Copy embryo structure but clear annotations
        for embryo_id, embryo_data in self.data["embryos"].items():
            batch_data["embryos"][embryo_id] = {
                "embryo_id": embryo_id,
                "experiment_id": embryo_data.get("experiment_id"),
                "video_id": embryo_data.get("video_id"),
                "genotype": None,
                "treatments": [],
                "snips": {}
            }
            
            # Copy snip structure but clear annotations
            for snip_id, snip_data in embryo_data.get("snips", {}).items():
                batch_data["embryos"][embryo_id]["snips"][snip_id] = {
                    "snip_id": snip_id,
                    "frame_number": snip_data.get("frame_number"),
                    "phenotypes": [],
                    "flags": []
                }
    
    elif mode == "copy":
        # Full copy of current data
        import copy
        batch_data = copy.deepcopy(self.data)
        batch_data["metadata"]["version"] = "batch_copy"
        batch_data["metadata"]["batch_mode"] = mode
    
    else:
        raise ValueError(f"Invalid batch mode: {mode}. Use 'skeleton' or 'copy'")
    
    return AnnotationBatch(batch_data, author, validate=self.validate)


# Add apply_batch method to EmbryoMetadata
def apply_batch(self, batch: AnnotationBatch, on_conflict: str = "error", dry_run: bool = False) -> Dict:
    """
    Apply batch changes to metadata with conflict resolution.
    
    Args:
        batch: AnnotationBatch instance
        on_conflict: Conflict resolution strategy
            - "error": Fail on any conflict
            - "skip": Keep existing data, skip conflicts
            - "overwrite": Replace existing data completely
            - "merge": Intelligently combine annotations
        dry_run: If True, validate without applying changes
        
    Returns:
        Report with applied count, conflicts, errors
    """
    report = {
        "operation": "apply_batch",
        "dry_run": dry_run,
        "on_conflict": on_conflict,
        "applied_count": 0,
        "skipped_count": 0,
        "conflicts": [],
        "errors": []
    }
    
    if not isinstance(batch, AnnotationBatch):
        raise ValueError("Must provide AnnotationBatch instance")
    
    # Apply changes
    for embryo_id, batch_embryo in batch.data["embryos"].items():
        try:
            # Ensure embryo exists in metadata
            if embryo_id not in self.data["embryos"]:
                report["errors"].append(f"Embryo {embryo_id} not found in metadata")
                continue
            
            # Apply genotype
            if batch_embryo.get("genotype"):
                existing_genotype = self.data["embryos"][embryo_id].get("genotype")
                if existing_genotype and on_conflict == "error":
                    report["conflicts"].append(f"Genotype conflict for {embryo_id}")
                elif existing_genotype and on_conflict == "skip":
                    report["skipped_count"] += 1
                elif not dry_run:
                    self.data["embryos"][embryo_id]["genotype"] = batch_embryo["genotype"]
                    report["applied_count"] += 1
            
            # Apply treatments
            if batch_embryo.get("treatments"):
                if not dry_run:
                    existing_treatments = self.data["embryos"][embryo_id].get("treatments", [])
                    if on_conflict == "overwrite":
                        self.data["embryos"][embryo_id]["treatments"] = batch_embryo["treatments"]
                    else:
                        # Merge treatments
                        self.data["embryos"][embryo_id]["treatments"] = existing_treatments + batch_embryo["treatments"]
                    report["applied_count"] += len(batch_embryo["treatments"])
            
            # Apply phenotypes
            for snip_id, batch_snip in batch_embryo.get("snips", {}).items():
                if batch_snip.get("phenotypes"):
                    if snip_id not in self.data["embryos"][embryo_id]["snips"]:
                        report["errors"].append(f"Snip {snip_id} not found")
                        continue
                    
                    if not dry_run:
                        existing_phenotypes = self.data["embryos"][embryo_id]["snips"][snip_id].get("phenotypes", [])
                        if on_conflict == "overwrite":
                            self.data["embryos"][embryo_id]["snips"][snip_id]["phenotypes"] = batch_snip["phenotypes"]
                        else:
                            # Merge phenotypes
                            self.data["embryos"][embryo_id]["snips"][snip_id]["phenotypes"] = existing_phenotypes + batch_snip["phenotypes"]
                        report["applied_count"] += len(batch_snip["phenotypes"])
        
        except Exception as e:
            report["errors"].append(f"Error processing {embryo_id}: {str(e)}")
    
    return report


# Monkey patch methods to EmbryoMetadata
EmbryoMetadata.initialize_batch = initialize_batch
EmbryoMetadata.apply_batch = apply_batch