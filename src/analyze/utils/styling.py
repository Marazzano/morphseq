"""
Generic styling utilities for plots.

Reusable functions for color mapping and style configuration.
"""

from typing import Dict, List, Any, Optional


class ColorLookup:
    """
    A dict-like object that automatically performs suffix-based color matching.
    
    Can be passed directly to color_lookup parameters without pre-building the dict.
    
    Parameters
    ----------
    suffix_colors : Dict[str, str]
        Mapping of suffixes to color codes (e.g., {'wildtype': '#2ca02c'})
    suffix_order : List[str]
        Priority order for matching suffixes
    fallback_palette : Optional[List[str]]
        Colors to use for unmatched values
    
    Examples
    --------
    >>> GENOTYPE_COLORS = ColorLookup(
    ...     suffix_colors={'wildtype': '#2ca02c', 'homozygous': '#d62728'},
    ...     suffix_order=['wildtype', 'homozygous']
    ... )
    >>> plot_feature_over_time(df, features='metric_value', color_lookup=GENOTYPE_COLORS)
    """
    
    def __init__(
        self,
        suffix_colors: Dict[str, str],
        suffix_order: List[str],
        fallback_palette: Optional[List[str]] = None
    ):
        self.suffix_colors = suffix_colors
        self.suffix_order = suffix_order
        if fallback_palette is None:
            fallback_palette = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            ]
        self.fallback_palette = fallback_palette
        self._cache: Dict[Any, str] = {}
        self._fallback_index = 0
    
    def __getitem__(self, key: Any) -> str:
        """Get color for a value, using suffix matching if needed."""
        if key in self._cache:
            return self._cache[key]
        
        # Try suffix matching
        key_str = str(key)
        for suffix in self.suffix_order:
            if key_str.endswith('_' + suffix) or key_str == suffix:
                color = self.suffix_colors[suffix]
                self._cache[key] = color
                return color
        
        # Fallback to palette
        color = self.fallback_palette[self._fallback_index % len(self.fallback_palette)]
        self._fallback_index += 1
        self._cache[key] = color
        return color
    
    def get(self, key: Any, default: Optional[str] = None) -> Optional[str]:
        """Dict-like get method."""
        try:
            return self[key]
        except Exception:
            return default
    
    def __contains__(self, key: Any) -> bool:
        """Check if key can be resolved to a color."""
        try:
            self[key]
            return True
        except Exception:
            return False
    
    def keys(self):
        """Return cached keys."""
        return self._cache.keys()
    
    def values(self):
        """Return cached values."""
        return self._cache.values()
    
    def items(self):
        """Return cached items."""
        return self._cache.items()


def build_suffix_color_lookup(
    values: List[Any],
    suffix_colors: Dict[str, str],
    suffix_order: List[str],
    fallback_palette: Optional[List[str]] = None,
) -> Dict[Any, str]:
    """Build color lookup by matching value suffixes.
    
    Generic utility for suffix-based color assignment. Useful for
    genotypes, conditions, or any categorical data with naming conventions.
    
    Parameters
    ----------
    values : list
        Values to assign colors to
    suffix_colors : dict
        Mapping from suffix strings to hex colors
    suffix_order : list
        Order to check suffixes (precedence)
    fallback_palette : list, optional
        Colors to use for unmatched values
    
    Returns
    -------
    dict
        Mapping from values to hex colors
    
    Examples
    --------
    >>> suffix_colors = {'wt': '#1f77b4', 'mut': '#ff7f0e'}
    >>> suffix_order = ['wt', 'mut']
    >>> values = ['exp1_wt', 'exp1_mut', 'exp2_wt']
    >>> build_suffix_color_lookup(values, suffix_colors, suffix_order)
    {'exp1_wt': '#1f77b4', 'exp1_mut': '#ff7f0e', 'exp2_wt': '#1f77b4'}
    """
    if fallback_palette is None:
        # Default palette if no match found
        fallback_palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
    
    lookup = {}
    fallback_idx = 0
    
    for val in values:
        val_str = str(val)
        matched = False
        
        # Try to match suffix
        for suffix in suffix_order:
            if val_str.endswith('_' + suffix) or val_str == suffix:
                lookup[val] = suffix_colors[suffix]
                matched = True
                break
        
        # Use fallback if no match
        if not matched:
            lookup[val] = fallback_palette[fallback_idx % len(fallback_palette)]
            fallback_idx += 1
    
    return lookup
