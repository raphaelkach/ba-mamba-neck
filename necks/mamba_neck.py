"""V3 — MambaFPN Neck (Liang et al. 2026).

VSS-Blocks mit Cross-Scan (4 Richtungen) ersetzen Conv-Layers im FPN.

Input:  List[Tensor] — {C2, C3, C4, C5} vom ResNet-50.
Output: Tuple[Tensor] — {P3, P4, P5, P6, P7} für ATSSHead.
"""
