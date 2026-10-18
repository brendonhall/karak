# main.tex number map

Sources:

- Manuscript: `/home/brendon/Dropbox/Projects/izawa/NWA4587_LPSC26/paper/main.tex` (1188 lines).
- Macros: `/home/brendon/Projects/science/nwa4587-karak-paper/results/numbers.tex` (875 `\newcommand` lines; same values in `numbers.json`).
- Sensitivity JSON: `/home/brendon/Dropbox/Projects/izawa/NWA4587_LPSC26/paper/sensitivity/`. The folder holds seven JSON files, not three: `baseline_results.json`, `hdbscan_results.json`, `knn_impact_results.json`, `strategy_results.json`, `tiled_robustness_results.json`, `timing_results.json`, `timing_wrapper_result.json`. All of them come from the old (published) pipeline run (`n_mineral_pixels = 12495787`; phase percentages such as plagioclase 28.76 match the old Table phases, not the regenerated macros).

Status rule: MATCH means the printed digits equal the macro value rounded to the printed precision. Rows marked "(derived)" have no single macro but the value follows from macros by simple arithmetic; they count under the status shown.

## Macro format notes (generator output versus manuscript form)

| Macro family | Generator form | Manuscript form | What the implementer needs |
|---|---|---|---|
| Pixel counts (`...Pixels`, `...PixelCount`, `...NIncluded`, `NMineralPixels`, `PyroxeneNPixels...`) | bare integer, `3614171` | `3{,}614{,}171` in tables; `12.5~million`, `4.64~million`, `37.6~million` in text | wrap in `\num{}` (siunitx is loaded, line 10, set `group-separator={,}`) or add `...Fmt` / `...Millions` variants to the generator |
| TIMA accuracies (`TimaSplitOverallAccuracy` 0.8412, `TimaCombinedOverallAccuracy` 0.9909) | fraction, 4 dp | percent, 1 dp (`84.1\%`, `99.1\%`) | a percent variant; 15.9 (discordant) also needs `100 - 84.12` |
| TIMA Spearman p (`6.25e-06`, `6.72e-05`) | e-notation | `$p < 0.001$` | a threshold formatter |
| Colormap errors (`ColormapJetMae` 1.39e-03, `ColormapJetMaxAe` 1.56e-02) | fraction in e-notation | percent (`0.14\%`, `1.6\%`) | a percent variant |
| `TablesParamsPCAVarianceKept` 0.95 | fraction | `95\%` | percent variant |
| `TablesParamsNormalisation` `zscore` | config token | `z-score` | display mapping |
| `TablesParamsDownsampleFactor` 2 | integer | `$2\times$`, "factor of two" | fine inside `$\nwa...\times$` |
| `TablesParamsRarePhaseSubsample` `500,000` | already has a comma | `500{,}000` | plain comma prints with a space in math mode; use `{,}` or `\num{500000}` |
| `TablesParamsMinimumClustersPerTile` 3 | integer | word "three" (line 406) | keep the word, or rewrite as a digit |
| `TablesPhasesXenotimePct` 0.00 | 2 dp | `$<$0.01` | conditional format for values under 0.005 |
| `LamellaeGrainSizeVsSpacingSpearmanP` 0.000 | 3 dp | `$p = 0.004$` | threshold formatter (`$p < 0.001$`) |
| `LamellaeNyquistLimitUm` 6.0, `LamellaePixelSizeUm` 3.0 | 1 dp | `\SI{6}{...}`, `\SI{3}{...}` | 0 dp variant or accept `6.0` |
| Lamellae spacings (`...MedianUm` 45 etc.) | integer µm | `\SI{51}{\micro\metre}` | works inside `\SI{\nwa...}{...}` |
| Phase-table percents (`TablesPhases...Pct`) | 2 dp | 2 dp in Table phases, 1 dp (2 dp below 1 %) in Section 3.1 text | a 1-dp variant for lines 562-570 |
| `TimaSplitSpearmanR` 0.952 | 3 dp | 2 dp (`0.95`) | a 2-dp variant |
| `PyroxenePctOneX...` 48.0 / 52.0 | 1 dp | 1 dp | fine |

Missing macros the text needs (none exists in `numbers.tex`): number of mineral phases (13), number of rare phases from pass 2, xenotime area percent (0.0017 % = 213 / 12495787), total 1x pyroxene pixels (18.5 M = augite + pigeonite + undefined), discordant percent, count of TIMA-matched phases (11), excluded grain count (272 - 154 = 118), the 6475 x 3990 downsampled shape, PCA component count (9) and its variance (96 %), valid-polygon area fraction (27 %), boundary noise share (28 %), over-split cosine (0.95), lamellae grain-size thresholds (500 px, 2000 px), structure-tensor sigma (1.5 px), and every knn-impact and sensitivity value.

Phase-count note: the regenerated run has 15 phases with pixels (`\nwaTablesNPhasesWithPixels` = 15, `\nwaWorkflowNPhasesFinal` = 15). It has no "Pyroxene (unres.)" phase (old label 2, 92 px). The 13 minerals (15 minus epoxy and weathering assemblage) still hold. Labels 13 and 14 swap: the regenerated run has pigeonite = 13, augite = 14.

## Table A. Pipeline-derived numbers

### Abstract, highlights, introduction, Figure 1

| Line | Text | Meaning | Macro = current value | Status |
|---|---|---|---|---|
| 113 | `13 mineral phases` | number of mineral phases | none (15 phases with pixels minus epoxy and weathering = 13) | NO MACRO (value still 13) |
| 115 | `0.002\%` | xenotime area fraction | none; `\nwaXenotimeNPixelsTwoX` 213 / `\nwaTablesNMineralPixels` 12495787 = 0.0017 % | NO MACRO (value still 0.002) |
| 116 | `84.1\%` | TIMA overall agreement, split pyroxenes | `\nwaTimaSplitOverallAccuracy` = 0.8412 | MATCH (needs percent format) |
| 116 | `$r_s = 0.95$` | TIMA modal Spearman r, split | `\nwaTimaSplitSpearmanR` = 0.952 | MATCH (needs 2 dp) |
| 116 | `$p < 0.001$` | TIMA Spearman p, split | `\nwaTimaSplitSpearmanP` = 6.25e-06 | MATCH (needs threshold format) |
| 117 | `99.1\%` | TIMA overall agreement, combined pyroxene | `\nwaTimaCombinedOverallAccuracy` = 0.9909 | MATCH (needs percent format) |
| 135 | `0.002\%` | rare-phase floor (xenotime area) | none (as line 115) | NO MACRO (value still 0.002) |
| 232 | `13 mineral phases` | number of mineral phases | none | NO MACRO (value still 13) |
| 252 | `13 unnamed clusters` | clusters before naming and pyroxene split | `\nwaWorkflowNClustersBeforePyx` = 13 | MATCH |

### Methods (Sections 2.3 to 2.6)

| Line | Text | Meaning | Macro = current value | Status |
|---|---|---|---|---|
| 314 | `4096` | colormap samples on [0, 1] | none; `\nwaColormapLutStep` = 0.000244 (= 1/4095, consistent) | NO MACRO |
| 334 | `$6475 \times 3990$` | shape after 2x downsample | none (12950 x 7980 / `\nwaTablesParamsDownsampleFactor` 2) | NO MACRO |
| 334 | `factor of two` | downsample factor | `\nwaTablesParamsDownsampleFactor` = 2 | MATCH (word form) |
| 341 | `approximately 27\%` | valid-polygon share of full image | none; source unknown (karak memory notes a 48.1 % valid fraction on a different image frame) | NO MACRO (unknown) |
| 345 | `below 100 pixels` | min object size | `\nwaTablesParamsMinimumObjectSizePx` = 100 | MATCH |
| 349 | `$\sigma_\text{spatial} = 1.0$` | bilateral spatial sigma | `\nwaTablesParamsBilateralSigmaSpatialPx` = 1.0 | MATCH |
| 350 | `automatic $\sigma_\text{colour}$` | bilateral colour sigma | `\nwaTablesParamsBilateralSigmaColour` = auto | MATCH |
| 372 | `95\%` | PCA variance target | `\nwaTablesParamsPCAVarianceKept` = 0.95 | MATCH (needs percent format) |
| 372 | `nine components` | PCA components kept | none; `hdbscan_results.json` `metadata.n_pca_components` = 9 (also `strategy_results.json`, `baseline_results.json`) | NO MACRO |
| 372 | `(96\%)` | PCA cumulative variance | none; `hdbscan_results.json` `metadata.pca_cumulative_variance` = 96.2 | NO MACRO |
| 385 | `$\sim$28\%` | mixed boundary pixels (noise share) | none; unknown (knn_impact `noise_fraction_pct` = 10.12; tiled_robustness `noise_pct_pre_knn` = 22.83; neither is 28) | NO MACRO (unknown) |
| 389 | `$\sim$12.5~million` | mineral pixels at 2x | `\nwaTablesNMineralPixels` = 12495787 | MATCH (needs millions format) |
| 395 | `$1024 \times 1024$` | tile size | `\nwaTablesParamsTileSizePx` = 1024 | MATCH |
| 397 | `\texttt{min\_cluster\_size}~$= 100$` | HDBSCAN min cluster size | `\nwaTablesParamsHDBSCANMinClusterSize` = 100 | MATCH |
| 397 | `\texttt{min\_samples}~$= 25$` | HDBSCAN min samples | `\nwaTablesParamsHDBSCANMinSamples` = 25 | MATCH |
| 402 | `0.88` | registry merge cosine | `\nwaTablesParamsTileMergeCosineThreshold` = 0.88 | MATCH |
| 406 | `fewer than three clusters` | min clusters per tile | `\nwaTablesParamsMinimumClustersPerTile` = 3 | MATCH (word form) |
| 415 | `of 50` | rare-phase min cluster size | `\nwaTablesParamsRarePhaseMinClusterSize` = 50 | MATCH |
| 417 | `$k = 5$` | kNN neighbours | `\nwaTablesParamsKNNNeighbours` = 5 | MATCH |
| 419 | `$\sim$200 pixels` | smallest detected phase | `\nwaXenotimeNPixelsTwoX` = 213 (`\nwaXenotimeLargestComponentPx` = 200) | MATCH (approximate) |
| 420 | `$\sim$0.002\%` | smallest phase area | none (213 / 12495787) | NO MACRO (value still 0.002) |
| 431 | `0.95` | over-split flag cosine | none | NO MACRO |
| 438 | `the 16 phases` | phases in Table phases | `\nwaTablesNPhasesWithPixels` = 15 | DIFFERS (16 vs 15) |
| 439 | `13 of them minerals` | mineral phases | none | NO MACRO (value still 13) |
| 454 | `$= 0.54$` | augite mean Ca/(Ca+Mg) | `\nwaPyroxeneRatioMeanAugite` = 0.54 (the GMM component mean `\nwaPyroxeneGmmMeanAugite` = 0.53) | MATCH (with RatioMean; text says "the GMM separates ... mean", so pick the macro on purpose) |
| 455 | `($= 0.30$)` | pigeonite mean Ca/(Ca+Mg) | `\nwaPyroxeneRatioMeanPigeonite` = 0.30 (`\nwaPyroxeneGmmMeanPigeonite` = 0.31) | MATCH (same caveat) |
| 463 | `minimum grain size 500 pixels` | lamellae segmentation param | none | NO MACRO |
| 464 | `$\sigma = 1.5$~px` | structure-tensor sigma | none | NO MACRO |
| 483 | `12.5~million mineral pixels` | mineral pixels at 2x | `\nwaTablesNMineralPixels` = 12495787 | MATCH (needs millions format) |

### Validation and results (Sections 2.7, 3.1 to 3.4)

| Line | Text | Meaning | Macro = current value | Status |
|---|---|---|---|---|
| 526 | `(1919, 1051)` | TIMA offset dx, dy | `\nwaTimaAlignmentDx` = 1919, `\nwaTimaAlignmentDy` = 1051 | MATCH |
| 557 | `16 phases` | phases identified | `\nwaTablesNPhasesWithPixels` = 15 | DIFFERS (16 vs 15) |
| 558 | `13 mineral phases` | minerals | none | NO MACRO (value still 13) |
| 560 | `92-pixel unresolved-pyroxene residual` | old label 2 | none; the regenerated run has no such phase (`\nwaPyroxeneNUndefinedOneX` = 130 is a 1x count, not this) | NO MACRO (phase gone; sentence and the "two non-mineral classes and a residual" count need a rewrite) |
| 562 | `plagioclase (28.8\%)` | area % | `\nwaTablesPhasesPlagioclasePct` = 28.92 | DIFFERS (28.8 vs 28.9) |
| 562 | `pigeonite (19.3\%)` | area % | `\nwaTablesPhasesPigeonitePct` = 19.32 | MATCH |
| 562 | `augite (17.9\%)` | area % | `\nwaTablesPhasesAugitePct` = 17.76 | DIFFERS (17.9 vs 17.8) |
| 563 | `65.9\%` | sum of three dominant | none (derived 28.92 + 19.32 + 17.76 = 66.00) | DIFFERS (65.9 vs 66.0, derived) |
| 565 | `weathering assemblage (14.6\%)` | area % | `\nwaTablesPhasesWeatheringAssemblagePct` = 14.14 | DIFFERS (14.6 vs 14.1) |
| 565 | `epoxy (8.5\%)` | area % | `\nwaTablesPhasesEpoxyPct` = 8.73 | DIFFERS (8.5 vs 8.7) |
| 567 | `Fe-Ti-Cr spinel (4.9\%)` | area % | `\nwaTablesPhasesFeTiCrSpinelPct` = 4.91 | MATCH |
| 567 | `ilmenite (2.8\%)` | area % | `\nwaTablesPhasesIlmenitePct` = 2.90 | DIFFERS (2.8 vs 2.9) |
| 567 | `ferroan olivine (2.6\%)` | area % | `\nwaTablesPhasesFerroanOlivinePct` = 2.58 | MATCH |
| 568 | `merrillite (0.34\%)` | area % | `\nwaTablesPhasesMerrillitePct` = 0.34 | MATCH |
| 568 | `calcite (0.15\%)` | area % | `\nwaTablesPhasesCalcitePct` = 0.18 | DIFFERS (0.15 vs 0.18) |
| 568 | `silica polymorph (0.10\%)` | area % | `\nwaTablesPhasesSilicaPolymorphPct` = 0.10 | MATCH |
| 569 | `chlorapatite (0.05\%)` | area % | `\nwaTablesPhasesChlorapatitePct` = 0.06 | DIFFERS (0.05 vs 0.06) |
| 570 | `(0.03\%)` | Fe oxyhydroxide area % | `\nwaTablesPhasesFeOxyhydroxidePct` = 0.03 | MATCH |
| 570 | `Zn-bearing phase (0.01\%)` | area % | `\nwaTablesPhasesZnBearingPhasePct` = 0.02 | DIFFERS (0.01 vs 0.02) |
| 570 | `213 pixels` | xenotime pixels | `\nwaTablesPhasesXenotimePixels` = 213 | MATCH |
| 570 | `$<$0.01\%` | xenotime area | `\nwaTablesPhasesXenotimePct` = 0.00 | MATCH (needs `<0.01` format) |
| 586 | `the 13 mineral phases` (Fig. 5 caption) | fingerprinted minerals | none (numbers.tex has fingerprints for 13 phases) | NO MACRO (value still 13) |
| 606 | `(1919, 1051)` | TIMA offset | `\nwaTimaAlignmentDx` / `Dy` = 1919 / 1051 | MATCH |
| 611 | `eleven matched phases` | phases in TIMA modal comparison | none (11 `TimaSplitAbundance...` phases) | NO MACRO (value still 11) |
| 611 | `$r_s = 0.95$` | Spearman r, split | `\nwaTimaSplitSpearmanR` = 0.952 | MATCH |
| 612 | `($p < 0.001$)` | Spearman p, split | `\nwaTimaSplitSpearmanP` = 6.25e-06 | MATCH |
| 612 | `33.1\%` | plagioclase pipeline % of compared area | `\nwaTimaSplitAbundancePlagioclasePipelinePct` = 33.3 | DIFFERS (33.1 vs 33.3) |
| 613 | `36.2\%` | plagioclase TIMA % | `\nwaTimaSplitAbundancePlagioclaseTimaPct` = 36.2 | MATCH |
| 614 | `21.0\%` | augite pipeline % | `\nwaTimaSplitAbundanceAugitePipelinePct` = 21.0 | MATCH |
| 615 | `22.8\%` | pigeonite pipeline % | `\nwaTimaSplitAbundancePigeonitePipelinePct` = 22.8 | MATCH |
| 615 | `14.6\%` | augite TIMA % | `\nwaTimaSplitAbundanceAugiteTimaPct` = 14.6 | MATCH |
| 615 | `34.6\%` | pigeonite TIMA % | `\nwaTimaSplitAbundancePigeoniteTimaPct` = 34.6 | MATCH |
| 616 | `43.8\%` | combined pyroxene pipeline % | `\nwaTimaCombinedAbundancePigeonitePipelinePct` = 43.8 (combined class is keyed "Pigeonite") | MATCH |
| 616 | `49.3\%` | combined pyroxene TIMA % | `\nwaTimaCombinedAbundancePigeoniteTimaPct` = 49.9 (split TIMA 14.6 + 34.6 = 49.2) | DIFFERS (49.3 vs 49.9) |
| 618 | `84.1\%` | overall agreement, split | `\nwaTimaSplitOverallAccuracy` = 0.8412 | MATCH |
| 618 | `37.6~million compared pixels` | pixels in comparison, split | `\nwaTimaSplitNIncluded` = 37730564 (combined `\nwaTimaCombinedNIncluded` = 37770693) | DIFFERS (37.6 vs 37.7) |
| 619 | `exceed 0.90` | F1 floor claim | (calcite F1 = 0.90 equals, does not exceed, in both runs) | MATCH (wording issue only) |
| 620 | `plagioclase (0.99)` | F1 split | `\nwaTimaSplitPerPhasePlagioclaseFOne` = 0.99 | MATCH |
| 621 | `Fe-Ti-Cr spinel (0.98)` | F1 | `\nwaTimaSplitPerPhaseFeTiCrSpinelFOne` = 0.98 | MATCH |
| 621 | `merrillite (0.98)` | F1 | `\nwaTimaSplitPerPhaseMerrilliteFOne` = 0.98 | MATCH |
| 621 | `ilmenite (0.96)` | F1 | `\nwaTimaSplitPerPhaseIlmeniteFOne` = 0.96 | MATCH |
| 622 | `(0.94)` | ferroan olivine F1 | `\nwaTimaSplitPerPhaseFerroanOlivineFOne` = 0.94 | MATCH |
| 622 | `chlorapatite (0.91)` | F1 | `\nwaTimaSplitPerPhaseChlorapatiteFOne` = 0.91 | MATCH |
| 622 | `calcite (0.90)` | F1 | `\nwaTimaSplitPerPhaseCalciteFOne` = 0.90 | MATCH |
| 624 | `augite F1~$= 0.60$` | F1 | `\nwaTimaSplitPerPhaseAugiteFOne` = 0.60 | MATCH |
| 624 | `pigeonite $0.75$` | F1 | `\nwaTimaSplitPerPhasePigeoniteFOne` = 0.74 | DIFFERS (0.75 vs 0.74) |
| 629 | `99.1\%` | overall agreement, combined | `\nwaTimaCombinedOverallAccuracy` = 0.9909 | MATCH |
| 633 | `F1~$= 0$` | silica and Fe oxyhydroxide F1 | `\nwaTimaSplitPerPhaseSilicaPolymorphFOne` = 0.00, `\nwaTimaSplitPerPhaseFeOxyhydroxideFOne` = 0.00 | MATCH |
| 634 | `93\%` | pipeline silica pixels in TIMA-unclassified | `\nwaTimaRareSilicaPolymorphTopPct` = 91.9 (`...TopTimaName` = Unknown1) | DIFFERS (93 vs 92) |
| 639 | `(73\%)` | Fe oxyhydroxide outside TIMA classified area | `\nwaTimaRareFeOxyhydroxideTopPct` = 70.1 (`...TopTimaName` = Background) | DIFFERS (73 vs 70) |
| 659 | `concordant (84.1\%)` | Fig. 4 caption | `\nwaTimaSplitOverallAccuracy` = 0.8412 | MATCH |
| 659 | `discordant (15.9\%)` | Fig. 4 caption | none (derived 100 - 84.12 = 15.88) | MATCH (derived) |
| 662 | `99.1\%` | Fig. 4 caption | `\nwaTimaCombinedOverallAccuracy` = 0.9909 | MATCH |
| 672 | `$520 \times 520$~px` | Fig. pyroxene-zoom window | `\nwaTimaFigFiveWindowSize` = 520 | MATCH |
| 672 | `$1.56 \times 1.56$~mm` | same window in mm | none (520 x `\nwaLamellaePixelSizeUm` 3.0 µm) | MATCH (derived) |
| 683 | `three accessory phases` | pass-2 rare phases (xenotime, Zn, Fe oxyhydroxide) | none | NO MACRO (count; check against the new run's pass-2 output) |
| 685 | `213 pixels` | xenotime pixels at 2x | `\nwaXenotimeNPixelsTwoX` = 213 | MATCH |
| 686 | `($<$0.01\% area)` | xenotime area | `\nwaTablesPhasesXenotimePct` = 0.00 | MATCH (needs `<0.01` format) |
| 693 | `of 50` | rare-phase min cluster size | `\nwaTablesParamsRarePhaseMinClusterSize` = 50 | MATCH |
| 697 | `764 pixels` | Zn-bearing pixels | `\nwaTablesPhasesZnBearingPhasePixels` = 2912 | DIFFERS (764 vs 2912) |
| 698 | `0.01\%` | Zn-bearing area | `\nwaTablesPhasesZnBearingPhasePct` = 0.02 | DIFFERS (0.01 vs 0.02) |
| 698 | `3{,}975 pixels` | Fe oxyhydroxide pixels | `\nwaTablesPhasesFeOxyhydroxidePixels` = 4178 | DIFFERS (3975 vs 4178) |
| 698 | `0.03\%` | Fe oxyhydroxide area | `\nwaTablesPhasesFeOxyhydroxidePct` = 0.03 | MATCH |
| 709 | `The 213-pixel xenotime cluster` | Fig. 7 caption | `\nwaXenotimeNPixelsTwoX` = 213 | MATCH |
| 718 | `4.64~million` | pyroxene pixels at 2x | `\nwaPyroxeneNPixelsTwoX` = 4633456 | DIFFERS (4.64 vs 4.63) |
| 718 | `37.1\% of mineral area` | pyroxene share at 2x | `\nwaPyroxenePctOfMineralTwoX` = 37.1 | MATCH |
| 724 | `18.6~million` | pyroxene pixels at 1x | none (derived 8904186 + 9629508 + 130 = 18533824) | DIFFERS (18.6 vs 18.5, derived) |
| 727 | `0.54` | augite mean ratio | `\nwaPyroxeneRatioMeanAugite` = 0.54 | MATCH |
| 727 | `8.9~million pixels` | augite 1x pixels | `\nwaPyroxeneNPixelsOneXAugite` = 8904186 | MATCH (needs millions format) |
| 727 | `48.1\%` | augite share of 1x pyroxene | `\nwaPyroxenePctOneXAugite` = 48.0 | DIFFERS (48.1 vs 48.0) |
| 727 | `(0.30;` | pigeonite mean ratio | `\nwaPyroxeneRatioMeanPigeonite` = 0.30 | MATCH |
| 728 | `9.6~million pixels` | pigeonite 1x pixels | `\nwaPyroxeneNPixelsOneXPigeonite` = 9629508 | MATCH (needs millions format) |
| 728 | `51.9\%` | pigeonite share | `\nwaPyroxenePctOneXPigeonite` = 52.0 | DIFFERS (51.9 vs 52.0) |

### Lamellae (Section 3.4, Fig. 6 and Fig. 8 captions, Section 4.2)

| Line | Text | Meaning | Macro = current value | Status |
|---|---|---|---|---|
| 748 | `128` | grains segmented | `\nwaLamellaeNGrainsTotal` = 272 | DIFFERS (128 vs 272) |
| 749 | `the 59 grains` | grains analysed | `\nwaLamellaeNGrainsAnalyzed` = 154 | DIFFERS (59 vs 154) |
| 750 | `$\geq$2000 pixels` | analysis size threshold | none | NO MACRO |
| 751 | `remaining 69` | grains excluded | none (derived 272 - 154 = 118) | DIFFERS (69 vs 118, derived) |
| 752 | `0.63` | mean coherence | `\nwaLamellaeCoherenceMean` = 0.62 | DIFFERS (0.63 vs 0.62) |
| 753 | `\SI{51}{\micro\metre}` | median spacing | `\nwaLamellaeSpacingMedianUm` = 45 | DIFFERS (51 vs 45) |
| 754 | `\SIrange{29}{73}` (low) | spacing Q25 | `\nwaLamellaeSpacingQTwoFiveUm` = 36 | DIFFERS (29 vs 36) |
| 754 | `\SIrange{29}{73}` (high) | spacing Q75 | `\nwaLamellaeSpacingQSevenFiveUm` = 63 | DIFFERS (73 vs 63) |
| 754 | `\SIrange{18}{912}` (low) | spacing min | `\nwaLamellaeSpacingMinUm` = 18 | MATCH |
| 754 | `\SIrange{18}{912}` (high) | spacing max | `\nwaLamellaeSpacingMaxUm` = 220 | DIFFERS (912 vs 220) |
| 756 | `\SI{6}{\micro\metre}` | Nyquist limit | `\nwaLamellaeNyquistLimitUm` = 6.0 | MATCH (0 dp) |
| 756 | `\SI{3}{\micro\metre\per pixel}` | pixel size | `\nwaLamellaePixelSizeUm` = 3.0 | MATCH (0 dp) |
| 758 | `$\sim$94\textdegree{}` | circular mean orientation | `\nwaLamellaeOrientationCircularMeanDeg` = 33.7 | DIFFERS (94 vs 33.7) |
| 758 | `$\bar{R} = 0.28$` | Rayleigh R-bar | `\nwaLamellaeOrientationRayleighRBar` = 0.04 | DIFFERS (0.28 vs 0.04) |
| 758 | `$p = 0.008$` | Rayleigh p | `\nwaLamellaeOrientationRayleighPValue` = 0.80 (`\nwaLamellaeOrientationUniform` = True) | DIFFERS (0.008 vs 0.80; the "statistically significant preferred direction" claim on lines 757-760 no longer holds) |
| 761 | `$\rho = 0.37$` | grain area vs spacing Spearman | `\nwaLamellaeGrainSizeVsSpacingSpearmanRho` = 0.32 | DIFFERS (0.37 vs 0.32) |
| 762 | `$p = 0.004$` | that p value | `\nwaLamellaeGrainSizeVsSpacingSpearmanP` = 0.000 | DIFFERS (0.004 vs <0.001; needs threshold format) |
| 775 | `0.54 and 0.30` | Fig. 6 class means | `\nwaPyroxeneRatioMeanAugite` = 0.54, `\nwaPyroxeneRatioMeanPigeonite` = 0.30 | MATCH |
| 788 | `($94^\circ$)` | Fig. 8 circular mean | `\nwaLamellaeOrientationCircularMeanDeg` = 33.7 | DIFFERS (94 vs 33.7) |
| 789 | `(\SI{6}{\micro\metre}` | Fig. 8 Nyquist | `\nwaLamellaeNyquistLimitUm` = 6.0 | MATCH |
| 789 | `median (\SI{51}{\micro\metre}` | Fig. 8 median | `\nwaLamellaeSpacingMedianUm` = 45 | DIFFERS (51 vs 45) |
| 938 | `\SI{6}{\micro\metre} (2 pixels)` | Nyquist floor | `\nwaLamellaeNyquistLimitUm` = 6.0 | MATCH |
| 939 | `All measured spacings ... exceed this floor` | no spacing at the floor | `\nwaLamellaeSpacingNNearNyquist` = 0, `\nwaLamellaeSpacingMinUm` = 18 | MATCH |

Note: memory file `lamellae-spacing-axis-bias.md` records that spacing = P/|sin 2θ| in the old code biases the 45 µm median; the user decision on that is pending.

### Sensitivity (Section 3.5) and limitations (Section 4.2); all from the old run's sensitivity JSON

| Line | Text | Meaning | Macro = current value | Status |
|---|---|---|---|---|
| 800 | `min\_cluster\_size~$= 100$`, `min\_samples~$= 25$` | global run params | `\nwaTablesParamsHDBSCANMinClusterSize` = 100, `...MinSamples` = 25; also `strategy_results.json` `hdbscan_params` | MATCH |
| 800 | `500{,}000 subsample` | global run subsample | none; `strategy_results.json` `hdbscan_params.subsample_n` = 500000 (same value as `\nwaTablesParamsRarePhaseSubsample`) | NO MACRO |
| 801 | `four minerals` | phases missed by global | none; `strategy_results.json` `n_tiled_only` = 5 (4 minerals + unresolved pyroxene) | NO MACRO |
| 803 | `$\leq$2.6\% of area` | largest missed phase (olivine) | none; `strategy_results.json` `matches[label 11].tiled_pct` 2.59 | NO MACRO |
| 804 | `ten classes` | matched classes | none; `strategy_results.json` `n_matched` = 10 | NO MACRO |
| 805 | `0.98` | mean cosine | none; `strategy_results.json` `mean_cosine_similarity` = 0.9823 | NO MACRO |
| 805 | `0.87--1.00` | cosine range | none; `min_cosine_similarity` = 0.8745, max 1.0 (`matches`) | NO MACRO |
| 814 | `50 to 5{,}000` | sweep range | none; `hdbscan_results.json` `runs[].min_cluster_size` 50..5000 | NO MACRO |
| 815 | `500{,}000 subsample`, `min\_samples~$= 25$` | sweep fixed values | none; `hdbscan_results.json` `metadata.subsample_n`, `metadata.min_samples` | NO MACRO |
| 816 | `from 10 to 6` | cluster counts | none; `hdbscan_results.json` `runs[].n_clusters` 10,10,9,9,7,6,6 | NO MACRO |
| 818 | `13.5--14.1\%` | noise range | none; `runs[].noise_pct` 13.53..14.14 | NO MACRO |
| 819 | `cosine similarity 1.00` | consecutive-run cosine | none; `hdbscan_results.json` `stability[].mean_similarity` = 1.0 | NO MACRO |
| 822 | `The value of 100` | pipeline min cluster size | `\nwaTablesParamsHDBSCANMinClusterSize` = 100 | MATCH |
| 828 | `nine PCA components` (line 827), `500{,}000-pixel subsample` | baseline setup | none; `baseline_results.json` `n_pca_components` = 9, `subsample_n` = 500000 | NO MACRO |
| 828 | `$k = 13$ (matching the pipeline's phase count)` | baseline k | none; `baseline_results.json` `runs[].k`; 13 = `\nwaWorkflowNClustersBeforePyx` | NO MACRO |
| 829 | `$k = 16$` | baseline k (old 16-phase count) | none; `baseline_results.json`; the regenerated count is 15 (`\nwaTablesNPhasesWithPixels`) | NO MACRO (the parenthetical rationale ties it to the old count) |
| 831 | `$\geq 0.80$` | baseline match threshold | none; analysis threshold in `baseline_comparison.py` | NO MACRO |
| 833 | `53.6\%` | lowest baseline agreement | none; `baseline_results.json` kmeans k=16 `pixel_agreement_with_pipeline` = 0.5358 | NO MACRO |
| 833 | `74.0\%` | highest baseline agreement | none; gmm k=13 = 0.7402 | NO MACRO |
| 909 | `$\sim$0.14\%` | jet colormap MAE | `\nwaColormapJetMae` = 1.39e-03 | MATCH (needs percent format) |
| 910 | `$\sim$1.6\%` | jet worst-case error (cyan) | `\nwaColormapJetMaxAe` = 1.56e-02 (`\nwaColormapJetRegionsCyanMaxAe` = 1.56e-02) | MATCH (needs percent format) |
| 925 | `10.1\%` | HDBSCAN noise fraction | none; `knn_impact_results.json` `noise_fraction_pct` = 10.12 (also `strategy_results.json` `tiled_noise_pct`) | NO MACRO |
| 926 | `1.1 percentage points` | max abundance change without kNN | none; `knn_impact_results.json` `max_abs_delta_pp` = 1.1154 | NO MACRO |
| 927 | `(mean 0.31)` | mean change | none; `mean_abs_delta_pp` = 0.3125 | NO MACRO |
| 928 | `44\% pyroxene` | share of reassigned noise | none; `phases[Pyroxene (combined)].share_of_reassigned_noise_pct` = 44.38 | NO MACRO |
| 928 | `25\% weathering assemblage` | share | none; `phases[Weathering Assemblage]` = 24.55 | NO MACRO |
| 928 | `23\% plagioclase` | share | none; `phases[Plagioclase]` = 22.76 | NO MACRO |

### Conclusions

| Line | Text | Meaning | Macro = current value | Status |
|---|---|---|---|---|
| 987 | `$\sim$0.002\%` | rare-phase floor | none (213 / 12495787) | NO MACRO (value still 0.002) |
| 995 | `13` (mineral phases) | minerals | none | NO MACRO (value still 13) |
| 997 | `213 pixels` | xenotime | `\nwaXenotimeNPixelsTwoX` = 213 | MATCH |
| 997 | `$<$0.01\% area` | xenotime area | `\nwaTablesPhasesXenotimePct` = 0.00 | MATCH (needs `<0.01` format) |
| 998 | `= 0.54` | augite ratio | `\nwaPyroxeneRatioMeanAugite` = 0.54 | MATCH |
| 999 | `0.30` | pigeonite ratio | `\nwaPyroxeneRatioMeanPigeonite` = 0.30 | MATCH |
| 1000 | `84.1\%` | overall agreement | `\nwaTimaSplitOverallAccuracy` = 0.8412 | MATCH |
| 1001 | `$r_s = 0.95$` | Spearman | `\nwaTimaSplitSpearmanR` = 0.952 | MATCH |
| 1001 | `$p < 0.001$` | Spearman p | `\nwaTimaSplitSpearmanP` = 6.25e-06 | MATCH |
| 1002 | `99.1\%` | combined agreement | `\nwaTimaCombinedOverallAccuracy` = 0.9909 | MATCH |

### Table params (lines 1075-1101)

| Line | Text | Meaning | Macro = current value | Status |
|---|---|---|---|---|
| 1082 | `100~px` | header trim | `\nwaTablesParamsHeaderTrimPx` = 100 | MATCH |
| 1083 | `$2\times$` | downsample factor | `\nwaTablesParamsDownsampleFactor` = 2 | MATCH |
| 1084 | `100~px` | min object size | `\nwaTablesParamsMinimumObjectSizePx` = 100 | MATCH |
| 1085 | `1.0~px` | bilateral sigma spatial | `\nwaTablesParamsBilateralSigmaSpatialPx` = 1.0 | MATCH |
| 1086 | `auto` | bilateral sigma colour | `\nwaTablesParamsBilateralSigmaColour` = auto | MATCH |
| 1087 | `z-score` | normalization | `\nwaTablesParamsNormalisation` = zscore | MATCH (display form differs) |
| 1088 | `95\%` | PCA variance target | `\nwaTablesParamsPCAVarianceKept` = 0.95 | MATCH (needs percent format) |
| 1089 | `1024~px` | tile size | `\nwaTablesParamsTileSizePx` = 1024 | MATCH |
| 1090 | `0.88` | registry merge cosine | `\nwaTablesParamsTileMergeCosineThreshold` = 0.88 | MATCH |
| 1091 | `3` | min clusters per tile | `\nwaTablesParamsMinimumClustersPerTile` = 3 | MATCH |
| 1092 | `100` | HDBSCAN min cluster size | `\nwaTablesParamsHDBSCANMinClusterSize` = 100 | MATCH |
| 1093 | `25` | HDBSCAN min samples | `\nwaTablesParamsHDBSCANMinSamples` = 25 | MATCH |
| 1094 | `50` | rare-phase min cluster size | `\nwaTablesParamsRarePhaseMinClusterSize` = 50 | MATCH |
| 1095 | `500{,}000` | rare-phase subsample | `\nwaTablesParamsRarePhaseSubsample` = 500,000 | MATCH (comma form) |
| 1096 | `5` | kNN k | `\nwaTablesParamsKNNNeighbours` = 5 | MATCH |
| 1097 | `0.95` | over-split flag cosine | none | NO MACRO |
| 1098 | `GMM-2 on Ca/(Ca+Mg)` | pyroxene split | `\nwaTablesParamsPyroxeneSplitFeatureOneX` = Ca/(Ca+Mg) (feature only; "GMM-2" not in a macro) | MATCH (feature part) |

The generator also exports three params that Table params does not list: `\nwaTablesParamsOlivineRule` (`Fe-K > 0.6 & Ca < 0.10`), `\nwaTablesParamsPhosphateGMMFeatures` (`Cl,Na,Mg,F`), `\nwaTablesParamsWeatheringGMMFeatures` (`Ca/(Ca+Mg),BSE,Fe-K`). These are the regenerated paper-flow split steps; the table may need rows for them. The `&` in the olivine rule needs escaping before it can sit in a tabular cell.

### Table phases (lines 1104-1133) and Table concordance (lines 1160-1185)

Table concordance "Pixels" = pipeline pixels of that phase in the overlap (TP + FP), so it maps to `...PixelCount`. Table correspondence (lines 1136-1157) has no number cells.

| Line | Text | Meaning | Macro = current value | Status |
|---|---|---|---|---|
| 1115 | `3` | Plagioclase label | `\nwaTablesPhasesPlagioclaseLabel` = 3 | MATCH |
| 1115 | `28.76` | Plagioclase area % | `\nwaTablesPhasesPlagioclasePct` = 28.92 | DIFFERS (text 28.76, macro 28.92) |
| 1115 | `3{,}593{,}809` | Plagioclase pixels (2x) | `\nwaTablesPhasesPlagioclasePixels` = 3614171 | DIFFERS (text 3{,}593{,}809, macro 3614171) |
| 1116 | `14` | Pigeonite label | `\nwaTablesPhasesPigeoniteLabel` = 13 | DIFFERS (text 14, macro 13) |
| 1116 | `19.26` | Pigeonite area % | `\nwaTablesPhasesPigeonitePct` = 19.32 | DIFFERS (text 19.26, macro 19.32) |
| 1116 | `2{,}407{,}260` | Pigeonite pixels (2x) | `\nwaTablesPhasesPigeonitePixels` = 2414187 | DIFFERS (text 2{,}407{,}260, macro 2414187) |
| 1117 | `13` | Augite label | `\nwaTablesPhasesAugiteLabel` = 14 | DIFFERS (text 13, macro 14) |
| 1117 | `17.88` | Augite area % | `\nwaTablesPhasesAugitePct` = 17.76 | DIFFERS (text 17.88, macro 17.76) |
| 1117 | `2{,}234{,}352` | Augite pixels (2x) | `\nwaTablesPhasesAugitePixels` = 2219269 | DIFFERS (text 2{,}234{,}352, macro 2219269) |
| 1118 | `12` | Weathering Assemb. label | `\nwaTablesPhasesWeatheringAssemblageLabel` = 12 | MATCH |
| 1118 | `14.64` | Weathering Assemb. area % | `\nwaTablesPhasesWeatheringAssemblagePct` = 14.14 | DIFFERS (text 14.64, macro 14.14) |
| 1118 | `1{,}829{,}811` | Weathering Assemb. pixels (2x) | `\nwaTablesPhasesWeatheringAssemblagePixels` = 1766957 | DIFFERS (text 1{,}829{,}811, macro 1766957) |
| 1119 | `4` | Epoxy label | `\nwaTablesPhasesEpoxyLabel` = 4 | MATCH |
| 1119 | `8.45` | Epoxy area % | `\nwaTablesPhasesEpoxyPct` = 8.73 | DIFFERS (text 8.45, macro 8.73) |
| 1119 | `1{,}055{,}840` | Epoxy pixels (2x) | `\nwaTablesPhasesEpoxyPixels` = 1090297 | DIFFERS (text 1{,}055{,}840, macro 1090297) |
| 1120 | `5` | Fe-Ti-Cr Spinel label | `\nwaTablesPhasesFeTiCrSpinelLabel` = 5 | MATCH |
| 1120 | `4.89` | Fe-Ti-Cr Spinel area % | `\nwaTablesPhasesFeTiCrSpinelPct` = 4.91 | DIFFERS (text 4.89, macro 4.91) |
| 1120 | `610{,}907` | Fe-Ti-Cr Spinel pixels (2x) | `\nwaTablesPhasesFeTiCrSpinelPixels` = 613557 | DIFFERS (text 610{,}907, macro 613557) |
| 1121 | `0` | Ilmenite label | `\nwaTablesPhasesIlmeniteLabel` = 0 | MATCH |
| 1121 | `2.84` | Ilmenite area % | `\nwaTablesPhasesIlmenitePct` = 2.90 | DIFFERS (text 2.84, macro 2.90) |
| 1121 | `354{,}787` | Ilmenite pixels (2x) | `\nwaTablesPhasesIlmenitePixels` = 361896 | DIFFERS (text 354{,}787, macro 361896) |
| 1122 | `11` | Ferroan Olivine label | `\nwaTablesPhasesFerroanOlivineLabel` = 11 | MATCH |
| 1122 | `2.59` | Ferroan Olivine area % | `\nwaTablesPhasesFerroanOlivinePct` = 2.58 | DIFFERS (text 2.59, macro 2.58) |
| 1122 | `323{,}636` | Ferroan Olivine pixels (2x) | `\nwaTablesPhasesFerroanOlivinePixels` = 322635 | DIFFERS (text 323{,}636, macro 322635) |
| 1123 | `15` | Merrillite label | `\nwaTablesPhasesMerrilliteLabel` = 15 | MATCH |
| 1123 | `0.34` | Merrillite area % | `\nwaTablesPhasesMerrillitePct` = 0.34 | MATCH |
| 1123 | `42{,}977` | Merrillite pixels (2x) | `\nwaTablesPhasesMerrillitePixels` = 42957 | DIFFERS (text 42{,}977, macro 42957) |
| 1124 | `6` | Calcite label | `\nwaTablesPhasesCalciteLabel` = 6 | MATCH |
| 1124 | `0.15` | Calcite area % | `\nwaTablesPhasesCalcitePct` = 0.18 | DIFFERS (text 0.15, macro 0.18) |
| 1124 | `18{,}261` | Calcite pixels (2x) | `\nwaTablesPhasesCalcitePixels` = 22915 | DIFFERS (text 18{,}261, macro 22915) |
| 1125 | `1` | Silica polymorph label | `\nwaTablesPhasesSilicaPolymorphLabel` = 1 | MATCH |
| 1125 | `0.10` | Silica polymorph area % | `\nwaTablesPhasesSilicaPolymorphPct` = 0.10 | MATCH |
| 1125 | `12{,}279` | Silica polymorph pixels (2x) | `\nwaTablesPhasesSilicaPolymorphPixels` = 12682 | DIFFERS (text 12{,}279, macro 12682) |
| 1126 | `16` | Chlorapatite label | `\nwaTablesPhasesChlorapatiteLabel` = 16 | MATCH |
| 1126 | `0.05` | Chlorapatite area % | `\nwaTablesPhasesChlorapatitePct` = 0.06 | DIFFERS (text 0.05, macro 0.06) |
| 1126 | `6{,}824` | Chlorapatite pixels (2x) | `\nwaTablesPhasesChlorapatitePixels` = 6961 | DIFFERS (text 6{,}824, macro 6961) |
| 1127 | `8` | Fe Oxyhydroxide label | `\nwaTablesPhasesFeOxyhydroxideLabel` = 8 | MATCH |
| 1127 | `0.03` | Fe Oxyhydroxide area % | `\nwaTablesPhasesFeOxyhydroxidePct` = 0.03 | MATCH |
| 1127 | `3{,}975` | Fe Oxyhydroxide pixels (2x) | `\nwaTablesPhasesFeOxyhydroxidePixels` = 4178 | DIFFERS (text 3{,}975, macro 4178) |
| 1128 | `9` | Zn-bearing label | `\nwaTablesPhasesZnBearingPhaseLabel` = 9 | MATCH |
| 1128 | `0.01` | Zn-bearing area % | `\nwaTablesPhasesZnBearingPhasePct` = 0.02 | DIFFERS (text 0.01, macro 0.02) |
| 1128 | `764` | Zn-bearing pixels (2x) | `\nwaTablesPhasesZnBearingPhasePixels` = 2912 | DIFFERS (text 764, macro 2912) |
| 1129 | `10` | Xenotime label | `\nwaTablesPhasesXenotimeLabel` = 10 | MATCH |
| 1129 | `$<$0.01` | Xenotime area % | `\nwaTablesPhasesXenotimePct` = 0.00 | MATCH (macro prints 0.00; text needs `$<$0.01`, so keep a conditional or a format rule) |
| 1129 | `213` | Xenotime pixels (2x) | `\nwaTablesPhasesXenotimePixels` = 213 | MATCH |
| 1130 | `2`, `$<$0.01`, `92` | Pyroxene (unres.): label, area %, pixels | none (regenerated run has no such phase; 15 phases with pixels) | NO MACRO |
| 1172 | `0.99` | Plagioclase Precision (split) | `\nwaTimaSplitPerPhasePlagioclasePrecision` = 0.99 | MATCH |
| 1172 | `0.99` | Plagioclase Recall (split) | `\nwaTimaSplitPerPhasePlagioclaseRecall` = 0.99 | MATCH |
| 1172 | `0.99` | Plagioclase F1 (split) | `\nwaTimaSplitPerPhasePlagioclaseFOne` = 0.99 | MATCH |
| 1172 | `13{,}960{,}619` | Plagioclase PixelCount (split) | `\nwaTimaSplitPerPhasePlagioclasePixelCount` = 14039603 | DIFFERS (text 13{,}960{,}619, macro 14039603) |
| 1173 | `0.88` | Pigeonite (low-Ca) Precision (split) | `\nwaTimaSplitPerPhasePigeonitePrecision` = 0.87 | DIFFERS (text 0.88, macro 0.87) |
| 1173 | `0.65` | Pigeonite (low-Ca) Recall (split) | `\nwaTimaSplitPerPhasePigeoniteRecall` = 0.65 | MATCH |
| 1173 | `0.75` | Pigeonite (low-Ca) F1 (split) | `\nwaTimaSplitPerPhasePigeoniteFOne` = 0.74 | DIFFERS (text 0.75, macro 0.74) |
| 1173 | `9{,}607{,}890` | Pigeonite (low-Ca) PixelCount (split) | `\nwaTimaSplitPerPhasePigeonitePixelCount` = 9596420 | DIFFERS (text 9{,}607{,}890, macro 9596420) |
| 1174 | `0.49` | Augite (high-Ca) Precision (split) | `\nwaTimaSplitPerPhaseAugitePrecision` = 0.48 | DIFFERS (text 0.49, macro 0.48) |
| 1174 | `0.78` | Augite (high-Ca) Recall (split) | `\nwaTimaSplitPerPhaseAugiteRecall` = 0.78 | MATCH |
| 1174 | `0.60` | Augite (high-Ca) F1 (split) | `\nwaTimaSplitPerPhaseAugiteFOne` = 0.60 | MATCH |
| 1174 | `8{,}843{,}093` | Augite (high-Ca) PixelCount (split) | `\nwaTimaSplitPerPhaseAugitePixelCount` = 8824364 | DIFFERS (text 8{,}843{,}093, macro 8824364) |
| 1175 | `0.98` | Fe-Ti-Cr Spinel Precision (split) | `\nwaTimaSplitPerPhaseFeTiCrSpinelPrecision` = 0.98 | MATCH |
| 1175 | `0.98` | Fe-Ti-Cr Spinel Recall (split) | `\nwaTimaSplitPerPhaseFeTiCrSpinelRecall` = 0.98 | MATCH |
| 1175 | `0.98` | Fe-Ti-Cr Spinel F1 (split) | `\nwaTimaSplitPerPhaseFeTiCrSpinelFOne` = 0.98 | MATCH |
| 1175 | `2{,}370{,}405` | Fe-Ti-Cr Spinel PixelCount (split) | `\nwaTimaSplitPerPhaseFeTiCrSpinelPixelCount` = 2377618 | DIFFERS (text 2{,}370{,}405, macro 2377618) |
| 1176 | `0.99` | Ilmenite Precision (split) | `\nwaTimaSplitPerPhaseIlmenitePrecision` = 0.99 | MATCH |
| 1176 | `0.93` | Ilmenite Recall (split) | `\nwaTimaSplitPerPhaseIlmeniteRecall` = 0.93 | MATCH |
| 1176 | `0.96` | Ilmenite F1 (split) | `\nwaTimaSplitPerPhaseIlmeniteFOne` = 0.96 | MATCH |
| 1176 | `1{,}407{,}537` | Ilmenite PixelCount (split) | `\nwaTimaSplitPerPhaseIlmenitePixelCount` = 1431304 | DIFFERS (text 1{,}407{,}537, macro 1431304) |
| 1177 | `0.93` | Ferroan Olivine Precision (split) | `\nwaTimaSplitPerPhaseFerroanOlivinePrecision` = 0.94 | DIFFERS (text 0.93, macro 0.94) |
| 1177 | `0.95` | Ferroan Olivine Recall (split) | `\nwaTimaSplitPerPhaseFerroanOlivineRecall` = 0.95 | MATCH |
| 1177 | `0.94` | Ferroan Olivine F1 (split) | `\nwaTimaSplitPerPhaseFerroanOlivineFOne` = 0.94 | MATCH |
| 1177 | `1{,}202{,}878` | Ferroan Olivine PixelCount (split) | `\nwaTimaSplitPerPhaseFerroanOlivinePixelCount` = 1199293 | DIFFERS (text 1{,}202{,}878, macro 1199293) |
| 1178 | `0.97` | Merrillite Precision (split) | `\nwaTimaSplitPerPhaseMerrillitePrecision` = 0.97 | MATCH |
| 1178 | `0.99` | Merrillite Recall (split) | `\nwaTimaSplitPerPhaseMerrilliteRecall` = 0.99 | MATCH |
| 1178 | `0.98` | Merrillite F1 (split) | `\nwaTimaSplitPerPhaseMerrilliteFOne` = 0.98 | MATCH |
| 1178 | `162{,}397` | Merrillite PixelCount (split) | `\nwaTimaSplitPerPhaseMerrillitePixelCount` = 162258 | DIFFERS (text 162{,}397, macro 162258) |
| 1179 | `0.88` | Chlorapatite Precision (split) | `\nwaTimaSplitPerPhaseChlorapatitePrecision` = 0.88 | MATCH |
| 1179 | `0.95` | Chlorapatite Recall (split) | `\nwaTimaSplitPerPhaseChlorapatiteRecall` = 0.93 | DIFFERS (text 0.95, macro 0.93) |
| 1179 | `0.91` | Chlorapatite F1 (split) | `\nwaTimaSplitPerPhaseChlorapatiteFOne` = 0.91 | MATCH |
| 1179 | `17{,}710` | Chlorapatite PixelCount (split) | `\nwaTimaSplitPerPhaseChlorapatitePixelCount` = 17980 | DIFFERS (text 17{,}710, macro 17980) |
| 1180 | `0.93` | Calcite Precision (split) | `\nwaTimaSplitPerPhaseCalcitePrecision` = 0.92 | DIFFERS (text 0.93, macro 0.92) |
| 1180 | `0.87` | Calcite Recall (split) | `\nwaTimaSplitPerPhaseCalciteRecall` = 0.89 | DIFFERS (text 0.87, macro 0.89) |
| 1180 | `0.90` | Calcite F1 (split) | `\nwaTimaSplitPerPhaseCalciteFOne` = 0.90 | MATCH |
| 1180 | `64{,}962` | Calcite PixelCount (split) | `\nwaTimaSplitPerPhaseCalcitePixelCount` = 78363 | DIFFERS (text 64{,}962, macro 78363) |
| 1181 | `0.00` | Silica polymorph Precision (split) | `\nwaTimaSplitPerPhaseSilicaPolymorphPrecision` = 0.00 | MATCH |
| 1181 | `0.00` | Silica polymorph Recall (split) | `\nwaTimaSplitPerPhaseSilicaPolymorphRecall` = 0.00 | MATCH |
| 1181 | `0.00` | Silica polymorph F1 (split) | `\nwaTimaSplitPerPhaseSilicaPolymorphFOne` = 0.00 | MATCH |
| 1181 | `1{,}791` | Silica polymorph PixelCount (split) | `\nwaTimaSplitPerPhaseSilicaPolymorphPixelCount` = 1795 | DIFFERS (text 1{,}791, macro 1795) |
| 1182 | `0.00` | Fe Oxyhydroxide Precision (split) | `\nwaTimaSplitPerPhaseFeOxyhydroxidePrecision` = 0.00 | MATCH |
| 1182 | `0.00` | Fe Oxyhydroxide Recall (split) | `\nwaTimaSplitPerPhaseFeOxyhydroxideRecall` = 0.00 | MATCH |
| 1182 | `0.00` | Fe Oxyhydroxide F1 (split) | `\nwaTimaSplitPerPhaseFeOxyhydroxideFOne` = 0.00 | MATCH |
| 1182 | `1{,}461` | Fe Oxyhydroxide PixelCount (split) | `\nwaTimaSplitPerPhaseFeOxyhydroxidePixelCount` = 1566 | DIFFERS (text 1{,}461, macro 1566) |
| 1166 | `F1~$= 0.99$` | combined pyroxene F1 (caption) | `\nwaTimaCombinedPerPhasePigeoniteFOne` = 0.99 | MATCH |
| 1166 | `84.1\%` | caption | `\nwaTimaSplitOverallAccuracy` = 0.8412 | MATCH |
| 1166 | `99.1\%` | caption | `\nwaTimaCombinedOverallAccuracy` = 0.9909 | MATCH |

## Table B. Numbers that are not pipeline results

| Line | Text | Source |
|---|---|---|
| 112, 231, 284, 996 | `20 element maps` / `20-element` / `20 element intensity maps` | instrument (acquisition) |
| 113, 292 | `$12{,}950 \times 7{,}980$` | input image shape after the 100-row trim (13050 - 100); no macro; could become one if the generator reads the load shape |
| 139, 1076, 1105 | `NWA~4587-2` | sample name |
| 243 | `$\approx$24~$\times$~39~mm` | derived: 7980 x 3 µm = 23.9 mm, 12950 x 3 µm = 38.9 mm |
| 270 | `NWA~011` | literature |
| 278 | `\SI{25}{\kilo\volt}`, `\SI{8.24}{\nano\ampere}` | instrument |
| 279 | `\SI{15}{\milli\metre}` | instrument |
| 280 | `\SI{3}{\micro\metre}`, `\SI{9}{\micro\metre}` | instrument (pixel and dot spacing) |
| 289 | `$13{,}050 \times 7{,}980$` | instrument (export size) |
| 291 | `100 rows` | instrument banner; equals `\nwaTablesParamsHeaderTrimPx` = 100 (could use the macro) |
| 294 | `\SI{2.0}{\kilo\electronvolt}` | literature (line energies) |
| 295 | `\SI{3.6}{\kilo\electronvolt}` | literature |
| 316 | `$256 \times 256 \times 256$` | algorithm (8-bit RGB) |
| 318 | `$0.0$` | algorithm (black mapped to zero) |
| 332, 342, 371, 483, 557 | `$C = 19$`, `all 19`, `19-channel`, `19 channels` | input choice (20 maps minus Fe La) |
| 357-358 | `$\mu = 0$`, `$\sigma = 1$` | definition of z-score |
| 385, 922 | `$\sim$1--3~\si{\micro\metre}` | literature (EDS interaction volume) |
| 411 | `$<$1\% area` | definition of "rare" |
| 445, 720 | `1--2 pixels wide` | observation; no macro |
| 462 | `8-connectivity` | algorithm setting |
| 482 to 486 | `239~min`, `10.0~GB`, `232~min`, `194~min`, `under 8~min` | timing run: `timing_results.json` `wall_total_min` = 238.91, `peak_rss_gb` = 9.98, `stages[cluster].duration_s` = 13910.9 (231.8 min), preprocessing stages 49.5 + 6.0 + 152.4 + 91.2 s = 5.0 min; 194 min Pass-1 is not in the JSON (unknown; `timing_run.log` shows Pass 2 ending at 3:49:38 elapsed) |
| 489 | `default seed 42` | config; `baseline_results.json` / `hdbscan_results.json` `random_state` = 42 |
| 493 | `2--23\%` | replicate noise range: `tiled_robustness_results.json` `conditions.threshold_0.88_natural.noise_pct_pre_knn` = 22.83 (upper end); the 2 % end is not in the JSON (unknown) |
| 494 | `11 phases` | `tiled_robustness_results.json` `n_phases` = 11 |
| 495 | `$\geq 0.996$` | replicate cosine; not in any JSON (unknown) |
| 496 | `$\sim$2.5 percentage points` | replicate abundance agreement; not in any JSON (unknown) |
| 516 | `53.1 to 74.2~wt\%` | literature (lastra2016) |
| 523, 604 | `3.0~\textmu m/px`, `$3.0$~\textmu m` | instrument |
| 524 | `$12{,}950 \times 7980$` | input image shape |
| 524 | `$14{,}736 \times 10{,}800$` | TIMA image shape (input) |
| 711 | `\SI{200}{\micro\metre}` | figure scale bar |
| 834 | `4--66~s` | `baseline_results.json` `runs[].elapsed_s` = 3.6, 3.7, 49.2, 65.7 |
| 845 | `2.4~million mineral pixels` | NWA 4587-3 transfer run; no JSON in the sensitivity folder |
| 848 | `four 1024-px tiles`, `three fell below` | NWA 4587-3 transfer run |
| 851 | `\texttt{tile\_size} to 512`, `$\sim$15 tiles` | NWA 4587-3 transfer run |
| 852 | `10 phases` | NWA 4587-3 transfer run |
| 853 | `cosine 0.84--0.999` | NWA 4587-3 transfer run |
| 854 | `0.95`, `0.98`, `0.98` | NWA 4587-3 transfer run |
| 938 | `(2 pixels)` | definition of Nyquist |
| 1025 | `Python 3.12` | software |
| 1030 | `32~GB RAM` | hardware (`timing_results.json` `hardware`) |
| 1031 | `$\sim$10--12~GB RAM` | hardware estimate; `timing_results.json` `peak_rss_gb` = 9.98 |
| 1032 | `2026` | software metadata |

## Figures, tables, preamble

`\includegraphics` lines (8):

| Line | File | Width |
|---|---|---|
| 241 | `figures/fig1_workflow.pdf` | `\textwidth` |
| 585 | `figures/fig5_fingerprints.pdf` | `\textwidth` |
| 646 | `figures/fig3_phase_comparison.pdf` | `\textwidth` |
| 657 | `figures/fig4_agreement_map.pdf` | `\textwidth` |
| 669 | `figures/fig_pyroxene_tima.pdf` | `\textwidth` |
| 706 | `figures/fig7_xenotime_callout.pdf` | `\columnwidth` |
| 770 | `figures/fig6_pyroxene_refinement.pdf` | `\textwidth` |
| 783 | `figures/fig8_lamellae.pdf` | `\textwidth` |

Table environments (4):

| Lines | Env | Label | Number cells |
|---|---|---|---|
| 1075-1101 | `table` | `tbl:params` | rows 1082-1098 |
| 1104-1133 | `table*` | `tbl:phases` | rows 1115-1130 |
| 1136-1157 | `table*` | `tbl:correspondence` | none |
| 1160-1185 | `table*` | `tbl:concordance` | caption line 1166, rows 1172-1182 |

Preamble (lines 1-160): `\usepackage` lines 8-12 (natbib, booktabs, siunitx, xcolor, graphicx); `\newcommand{\note}` line 15 and `\newcommand{\TODO}` line 16. There is no `\input` line. Put `\input{numbers}` after line 16 (before `\begin{document}` on line 18). Add `\sisetup{group-separator={,}}` there too if `\num{\nwa...}` is used for pixel counts.
