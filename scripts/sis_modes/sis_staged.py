from __future__ import annotations
# ここでは selective と同一実装でもOK（F1を多段にしたい場合は内部で更に分割）
from .sis_selective import SelectiveRunner as StagedRunner  # 同じ振る舞いなら流用
StagedRunner.__name__ = "StagedRunner"
