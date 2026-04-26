# Code-Review Teil 3/4 — Custom-Module + Notebook-Sync

**Datum:** 2026-04-26
**Commit:** d2f1eea7017b14e34c19dcd8fd0599249cd5b686
**Branch:** claude/review-mamba-neck-part1-MPXyt

## Zusammenfassung
Alle vier Custom-Module-Implementationen (BF16SafeFocalLoss,
SS2D-Float32-Fix, AifiNeck-Architektur, MambaNeck-Architektur) entsprechen
den Erwartungen und sind sauber gegen die jeweiligen numerischen Fallstricke
abgesichert. Die drei Trainings-Notebooks sind in 8 von 11 Zellen byte-identisch;
Cell 2 (Install) und Cell 6 (Config) weichen erwartungsgemäß ab, während
Cell 3 (Setup 1b) im Mamba-Notebook minimal abweicht (zwei Zeilen für
mamba-ssm-Import-Verifikation), was strikt genommen die Notebook-Sync-Regel
aus CLAUDE.md verletzt. Es gibt 0 kritische Befunde, 1 wichtigen Befund
(Setup-Sync-Abweichung) und 2 INFO-Befunde.

## Befund-Übersicht
- KRITISCH: 0
- WICHTIG: 1   (Setup-Cell in 02_train_mamba weicht von FPN/AIFI ab — Notebook-Sync-Regel)
- INFO: 2      (Module-Docstring der MambaNeck nennt den BF16-Overflow nicht explizit; AifiNeck input_proj enthält BN+SiLU statt nur 1×1)

## Aufgabe 1: BF16SafeFocalLoss

**Datei:** `losses/bf16_focal_loss.py`

| Prüfpunkt | Befund | Status |
|---|---|---|
| 1. Klasse vorhanden? | `class BF16SafeFocalLoss(FocalLoss)` mit `@MODELS.register_module()` | ✅ Zeile 19–20 |
| 2. Wo der Numerik-Fix? | **Upcast-vor-Forward**: Wenn `pred.dtype == torch.bfloat16`, werden `pred`, ggf. `target` (falls floating) und `weight` (falls bfloat16) auf `float32` gecastet, dann wird `super().forward(...)` aufgerufen. Der eigentliche `sigmoid_focal_loss` läuft dadurch in `float32`. | ✅ Zeile 24–29 |
| 3. Docstring? | Modul-Docstring (Z. 1–7) erklärt das Problem explizit: *"mmcv's sigmoid_focal_loss CUDA kernel only supports float16/float32. Under bfloat16 AMP the head produces bfloat16 logits which crash the kernel."* | ✅ |
| 4. Konfig-Referenz? | `configs/_base_/model.py:48` — `loss_cls=dict(type='BF16SafeFocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0)`. Snapshot-Custom-Imports enthalten `'losses'`, daher wird die Registry korrekt erweitert. | ✅ |

Anmerkung: Die Implementation ist ein dünner Wrapper um `mmdet.FocalLoss`,
nicht eine Re-Implementation des Operators. Der Fix erfolgt also nicht durch
selektives Float32-Casten von `sigmoid` oder `log` *innerhalb* des Operators,
sondern durch Hochcasten *aller* Eingänge — der mmcv-Kernel selbst wird in
float32 ausgeführt. Das ist semantisch äquivalent (sigmoid+log werden in fp32
gerechnet) und in Hinsicht auf Korrektheit/Reproduzierbarkeit unproblematisch.
Kein Befund.

## Aufgabe 2: SS2D-Float32-Fix

**Datei:** `necks/mamba_neck.py`, Klasse `SS2D` (Z. 100–176)

| Prüfpunkt | Befund | Status |
|---|---|---|
| 1. `autocast('cuda', enabled=False)`? | `with torch.amp.autocast('cuda', enabled=False):` umschließt den gesamten Forward-Body. | ✅ Zeile 152 |
| 2. Input → `x.float()`? | Ja, direkt nach autocast-Disable: `x = x.float()`. | ✅ Zeile 153 |
| 3. Output → `input_dtype`? | `input_dtype = x.dtype` wird vor autocast gespeichert (Z. 151), Rückkonvertierung über `.to(input_dtype)` am Ende des Forward (Z. 176). | ✅ |
| 4. `dt`, `B`, `C` zu `u.dtype` in `_scan_direction`? | Drei explizite Casts: `dt = dt.to(u.dtype)` (Z. 142), `B_raw = B_raw.to(u.dtype)` (Z. 145), `C_raw = C_raw.to(u.dtype)` (Z. 146). | ✅ |
| 5. Erklärender Kommentar? | Modul-Docstring (Z. 12–20, „Correctness notes“) dokumentiert nur die CUDA-vs.-Naive-Pfade. Der Grund für `autocast(enabled=False)` (selective_scan_cuda-Overflow trotz bfloat16) ist im Code selbst NICHT als Inline-Kommentar dokumentiert. | ⚠️ INFO |

Befund INFO: Der Float32-Fix ist korrekt implementiert, aber der *Warum*-Teil
(numerische Instabilität von `selective_scan_fn` unter bfloat16-AMP) sollte
als Inline-Kommentar an Z. 152 oder im Modul-Docstring ergänzt werden — für
Reproduzierbarkeitsdokumentation und BA-Methodikkapitel relevant.

## Aufgabe 3: AifiNeck

**Datei:** `necks/aifi_neck.py`

| Prüfpunkt | Befund | Status |
|---|---|---|
| 1. AIFI MHSA 8 Heads + FFN 1024 Dim? | `class AIFIBlock(... num_heads: int = 8, dim_ff: int = 1024 ...)` (Z. 87–88); im AifiNeck wird mit den Konstruktor-Defaults instanziiert (Z. 209–212). | ✅ |
| 2. PE: 2D sinusoidal, temperature=10000? | `class SinusoidalPosEmbed2D(... temperature: float = 10000.0 ...)` (Z. 58–61); 2D sin/cos über H und W mit Aufteilung in 4-Tupel (Z. 67–81). | ✅ |
| 3. CCFM: 4 Stages? | `td_45`, `td_34` (top-down ×2) + `bu_34`, `bu_45` (bottom-up ×2) = 4 RepConvBlock-Stacks (Z. 141–144), jeweils mit eigener Concat+1×1-Reduktion. | ✅ |
| 4. Asserts? | `assert num_outs == 5` (Z. 197), `assert start_level == 1` (Z. 198), `assert len(in_channels) == 4` (Z. 199), Forward-Assert auf Input-Länge (Z. 218–220). | ✅ |
| 5. 1×1 Input-Projection für C3, C4, C5 → 256? | `self.input_proj = nn.ModuleList([_conv_bn_act(c, out_channels, 1) for c in used_ch])` — 1×1 Conv + BN + SiLU statt nur Conv (Z. 206–208). | ✅ (mit Hinweis) |
| 6. Extra P6, P7 via 3×3 stride-2 + BN + SiLU? | `self.extra_p6 = _conv_bn_act(out_channels, out_channels, 3, s=2)` und analog `extra_p7` (Z. 214–215) — 3×3, stride 2, padding 1, BN, SiLU. | ✅ |

INFO: Die Aufgabenstellung erwartet „1×1 Conv für C3, C4, C5 → 256 Kanäle“.
Implementiert ist `_conv_bn_act(c, out_channels, 1)` = Conv 1×1 + BN + SiLU.
Das ist konsistent mit dem RT-DETR-Original (Zhao et al. 2024, Sec. 4.2),
weicht aber leicht von einer reinen 1×1-Conv-Lesart ab. Kein Befund, aber im
Methodenkapitel der BA explizit als „Conv-BN-Act-Block“ bezeichnen.

CCFM-Forward-Pfad (Z. 152–164) implementiert: f5↑→f4 → f4_td; f4_td↑→f3 → f3_td;
dann bottom-up f3_td↓→f4_td → f4_out; f4_out↓→f5 → f5_out. Reihenfolge
und Concat+1×1+RepConv-Refinement entsprechen RT-DETR Fig. 5.

## Aufgabe 4: MambaNeck

**Datei:** `necks/mamba_neck.py`

| Prüfpunkt | Befund | Status |
|---|---|---|
| 1. VSS-Blocks pro Stufe (P3, P4, P5)? | `vss_per_level = nn.ModuleList([nn.Sequential(*[VSSBlock(...) for _ in range(num_vss_blocks)]) for _ in range(len(used_ch))])` — eine Sequence mit `num_vss_blocks` (default 2) VSS-Blöcken pro genutzter Pyramidenstufe. `len(used_ch)=3` (C3, C4, C5 → P3, P4, P5). | ✅ Z. 258–265 |
| 2. 4 Cross-Scan-Richtungen? | `SS2D.K = 4` (Z. 103). Forward bildet `x_row`, `x_col`, `x_row.flip(-1)`, `x_col.flip(-1)` = row-fwd, col-fwd, row-rev, col-rev (Z. 157–164), läuft 4 unabhängige Scans, summiert zurück nach Re-Permutation in (B, d, H, W) (Z. 165–175). | ✅ |
| 3. Top-Down via upsample + add + VSS (kein Concat)? | `up = F.interpolate(outs[i+1], size=feats[i].shape[-2:], mode='nearest')` und `outs[i] = self.vss_per_level[i](feats[i] + up)` (Z. 290–292) — additive FPN-Fusion, kein Concat. | ✅ |
| 4. Asserts? | `num_outs == 5` (Z. 246), `start_level == 1` (Z. 247), `len(in_channels) == 4` (Z. 248), Forward-Assert auf Input-Länge (Z. 280–282). Keine Assert auf konkrete Spatial-Shapes (nur auf Anzahl). | ✅ |
| 5. 1×1 Input-Projection für C3, C4, C5 → 256? | `self.input_proj = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in used_ch])` — bare 1×1 Conv ohne BN/SiLU (Z. 255–257). | ✅ |
| 6. d_state, expand=2, d_conv konfigurierbar? | Konstruktor-Args: `d_state: int = 16`, `expand: int = 2`, `d_conv: int = 3` (Z. 240–242). Werden an jeden VSSBlock weitergereicht (Z. 260–263). | ✅ |

INFO: AifiNeck verwendet `_conv_bn_act` (Conv+BN+SiLU) für die Input-Projection,
MambaNeck verwendet bare `nn.Conv2d` für die Input-Projection. Diese
Asymmetrie ist architektonisch motiviert (VSSBlock besitzt sein eigenes
LayerNorm im Eingangspfad und braucht keine zusätzliche BN-Normalisierung
vor dem Block) und stellt keinen Ceteris-Paribus-Verstoß dar — nur das
*Neck* unterscheidet sich, das ist genau die Vorgabe. Kein Befund;
Dokumentationswert für die BA.

VSSBlock-Pre-LN-Residual (Z. 198–213): LayerNorm → 2-fach In-Projection
(`a, z = chunk(2)`), DWConv1d über die HW-Sequenz, SS2D auf `a`, Gate
`a * silu(z)`, Out-Projection, Residual-Add. Entspricht VMamba Fig. 2c
(Liu et al. 2024).

## Aufgabe 5: Notebook-Sync

**Methodik:** Alle 11 Zellen pro Notebook wurden über SHA-256 ihres
`source`-Strings verglichen. Cells mit identischem Hash sind byte-identisch.

### Cell-Map (Hash-Präfix der ersten 12 Zeichen)

| # | Typ | FPN | AIFI | MAMBA | Status |
|---|---|---|---|---|---|
| 0 | markdown | `daf91f8e3978` | `daf91f8e3978` | `daf91f8e3978` | identisch ✅ |
| 1 | code (Imports) | `9ccad9076158` | `9ccad9076158` | `9ccad9076158` | identisch ✅ |
| 2 | code (Install) | `4147807eb142` | `4147807eb142` | `9bf5ed31a8a9` | MAMBA weicht ab ✅ (Ausnahme erlaubt) |
| 3 | code (Setup 1b) | `8a12a0e4382c` | `8a12a0e4382c` | `b49df39d1b5c` | MAMBA weicht ab ⚠️ WICHTIG |
| 4 | code (Daten-Drive) | `5c610de4860a` | `5c610de4860a` | `5c610de4860a` | identisch ✅ |
| 5 | code (Hardware-Check) | `47d051df83a6` | `47d051df83a6` | `47d051df83a6` | identisch ✅ |
| 6 | code (Config / NECK) | `d4ebe7111494` | `55bb3cab6e90` | `89e582a36672` | nur `NECK="..."` differs ✅ (Ausnahme erlaubt) |
| 7 | code (Training-Loop) | `9e0ba06cfea1` | `9e0ba06cfea1` | `9e0ba06cfea1` | identisch ✅ |
| 8 | code (Eval-Loop) | `8f6005b3ebd1` | `8f6005b3ebd1` | `8f6005b3ebd1` | identisch ✅ |
| 9 | code (Validierung) | `e15ae4bb150c` | `e15ae4bb150c` | `e15ae4bb150c` | identisch ✅ |
| 10 | code (Konvergenz) | `52ccbba55fbe` | `52ccbba55fbe` | `52ccbba55fbe` | identisch ✅ |

### Diff-Details

**Cell 2 (Install) — FPN/AIFI vs. MAMBA** (erlaubte Ausnahme):
```
+# 5. Mamba CUDA-Kernel
+os.environ['CUDA_HOME'] = '/usr/local/cuda'
+os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'  # A100 = sm_80
+pip(['git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0', '--no-build-isolation'])
+pip(['git+https://github.com/state-spaces/mamba.git@v2.2.2', '--no-build-isolation'])
 ...
+from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
+print('mamba-ssm 2.2.2 OK')
```
Das ist exakt die in CLAUDE.md vorgesehene Ausnahme („mamba-ssm only in
02_train_mamba.ipynb“). Konsistent.

**Cell 3 (Setup 1b) — FPN/AIFI vs. MAMBA** (Sync-Verletzung, WICHTIG):
```
 import mmdet
 print(f'mmdet: {mmdet.__version__}')
+import mamba_ssm
+print(f'mamba-ssm: {mamba_ssm.__version__}')
 print('Setup OK.')
```
Die CLAUDE.md-Regel sagt: *„Setup, imports, training loop, eval loop und
convergence extraction are identical.“* Die zwei zusätzlichen Zeilen im
Mamba-Notebook sind funktional eine Verifikations-Print-Anweisung und
verändern keine Trainingslogik, aber sie sind ein **strikter Verstoß** gegen
die Sync-Regel. Empfehlung:
- entweder die zwei Zeilen in Cell 2 (Install) verschieben (dort sind sie
  durch die Install-Ausnahme ohnehin abgedeckt — Cell 2 enthält bereits einen
  `from mamba_ssm.ops...` Import + Print am Ende), oder
- Sync-Regel in CLAUDE.md explizit um „Setup darf einen mamba-ssm-Import-Smoke
  enthalten“ erweitern.

**Cell 6 (Config) — FPN vs. AIFI vs. MAMBA** (erlaubte Ausnahme):
```
-# Zelle 3 - Config (FPN)
-NECK = "fpn"
+# Zelle 3 - Config (AIFI)
+NECK = "aifi"

-# Zelle 3 - Config (FPN)
-NECK = "fpn"
+# Zelle 3 - Config (MAMBA)
+NECK = "mamba"
```
Reine `NECK`-Variable + Kommentar-Header. Genau die in CLAUDE.md erlaubte
Differenz. Konsistent.

### Inhaltliche Gleichheit

| Prüfpunkt | FPN | AIFI | MAMBA |
|---|---|---|---|
| Seed-Liste `[42, 123, 456, 789, 1024, 2048, 3407, 4096, 5555, 7777]` | ✅ | ✅ | ✅ |
| Drive-Pattern `MyDrive/ba/{NECK}/seed_{seed}` (via `DRIVE_BASE` + `/seed_{seed}`) | ✅ | ✅ | ✅ |
| `torch==2.5.1+cu124` Constraint im pip-Install | ✅ | ✅ | ✅ |

Alle drei Notebooks enthalten denselben Seed-Literal, dasselbe Drive-Layout
(`{DRIVE_BASE}/seed_{seed}` und `{DRIVE_BASE}/seed_{seed}/run_meta.json`)
und denselben PyTorch-Pin. Die Trainings-Loop- und Eval-Loop-Zellen sind
byte-identisch über alle drei Varianten — die Notebook-Sync-Regel ist
**funktional** vollständig erfüllt; einzig die zwei Verifikations-Zeilen
in Cell 3 des Mamba-Notebooks sind formal abweichend.
