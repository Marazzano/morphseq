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
