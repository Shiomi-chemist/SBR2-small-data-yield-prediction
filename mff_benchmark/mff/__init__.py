"""Standalone re-implementation of MFF (Multiple Fingerprint Features).

Reference
---------
Sandfort, Strieth-Kalthoff, Kuehnemund, Beecks, Glorius,
"A Structure-Based Platform for Predicting Chemical Reactivity",
Chem 6, 1379-1390 (2020). https://doi.org/10.1016/j.chempr.2020.02.017

The fingerprint definitions follow ``EasyChemML/Encoder/MFF.py`` and
``EasyChemML/Encoder/impl_FingerprintEncoder/FingerprintGenerator.pyx``
from https://github.com/mkueh/EasyChemML (same authors) exactly, but
without the heavy framework (torch, deepchem, Cython, Rust) so that the
encoder runs on macOS with plain RDKit + scikit-learn.
"""
from .encoder import MFFEncoder, FP_SPECS

__all__ = ["MFFEncoder", "FP_SPECS"]
