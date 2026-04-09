"""V2 — AIFI + CCFM Neck (RT-DETR, Zhao et al. 2024).

Input:  List[Tensor] — {C2, C3, C4, C5} vom ResNet-50.
Output: Tuple[Tensor] — {P3, P4, P5, P6, P7} für ATSSHead.
"""
