import importlib
import sys
import unittest


try:
    import sklearn  # noqa: F401
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def _clear_modules(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]


def _import_fresh(module_name: str):
    _clear_modules(module_name)
    return importlib.import_module(module_name)


class CompatImportTests(unittest.TestCase):
    def test_statistics_shim_warns_and_exports(self):
        with self.assertWarns(DeprecationWarning):
            mod = _import_fresh("src.analyze.difference_detection.statistics")

        from src.analyze.difference_detection import distance_metrics

        self.assertIs(mod.compute_energy_distance, distance_metrics.compute_energy_distance)
        self.assertIs(mod.compute_mmd, distance_metrics.compute_mmd)
        self.assertIs(mod.compute_mean_distance, distance_metrics.compute_mean_distance)

    @unittest.skipIf(not SKLEARN_AVAILABLE, "sklearn not available")
    def test_comparison_shim_warns_and_exports(self):
        _clear_modules("src.analyze.difference_detection.compat")
        with self.assertWarns(DeprecationWarning):
            mod = _import_fresh("src.analyze.difference_detection.compat.comparison")
        self.assertTrue(hasattr(mod, "compare_groups"))

    @unittest.skipIf(not SKLEARN_AVAILABLE, "sklearn not available")
    def test_comparison_multiclass_shim_warns_and_exports(self):
        _clear_modules("src.analyze.difference_detection.compat")
        with self.assertWarns(DeprecationWarning):
            mod = _import_fresh("src.analyze.difference_detection.compat.comparison_multiclass")
        self.assertTrue(hasattr(mod, "compare_groups_multiclass"))


if __name__ == "__main__":
    unittest.main()
