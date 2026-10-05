# VALLIS-3C 1.6.0 — model architecture and traceability


This document is the human-readable map of the scientific architecture distributed with VALLIS-3C 1.6.0. It links the formulation described in the companion manuscript to the packaged model artifacts, runtime implementation, and scientific validation data.


It is intentionally not a second training manifest. Exact fitted coefficients, checksums, eligibility counts, residual-mode dimensions, and model-specific numerical parameters remain authoritative in the JSON manifests distributed beside each production artifact.


## 1. Architecture at a glance


VALLIS-3C separates spectral amplitude from temporal organization and then recombines them during synthesis:


```text
earthquake scenario + site
        |
        v
conditional 3C Fourier spectra
  - CU source/path reference
  - site amplification
  - persistent local spectral departure
  - Major/Intermediate and Vertical/EAS ratios
        |
        +--> stochastic spectral residual sampling
        |
        v
realization-specific Major / Intermediate / Vertical target FAS
        |
        +-----------------------------+
        |                             |
        v                             v
Unified H/V duration model     ExtraTrees texture selector
  - horizontal D_H              - scenario and site
  - 8 V/H duration ratios       - realization-specific 3C FAS
  - residual duration ranks     - empirical 3C temporal support
        |                             |
        +--------------+--------------+
                       |
                       v
             phase-diffusion carrier
                       |
                       v
             two-pass complex transport
                       |
                       v
          smooth FAS correction toward target
                       |
                       v
             zero-lag covariance correction
                       |
                       v
          completed Major / Intermediate / Vertical motions
```


The spectral center, duration model, and time-frequency selector are learned or calibrated independently. Dependence in the generated motions is introduced through shared scenario/site conditioning, shared spectral quantities, empirical three-component textures, common synthesis operations, and the ensemble duration-mapping procedure. VALLIS-3C does not use a single monolithic neural network or a single fully joint residual model.


## 2. Release variants


The release manifest is:


`models/release/model_manifest.json`


Three model variants are distributed under the same `final_hv` architecture:


| Variant | Purpose | Directory |
| --- | --- | --- |
| `FULL` | Production model trained with the complete release population | `models/release/variants/full/` |
| `OOS2017` | Strict validation model excluding the 19 September 2017 event from fitted models and eligible donor populations | `models/release/variants/oos_2017/` |
| `OOS2012_2017` | Strict validation model excluding the 20 March 2012 and 19 September 2017 events from fitted models and eligible donor populations | `models/release/variants/oos_2012_2017/` |


The graphical production interface exposes `FULL`. OOS2017 and OOS2012_2017 are retained for the corresponding withheld-event validations.


## 3. Model inventory


| Scientific block | Model / algorithm | Main conditioning variables | Production artifact | Main implementation |
| --- | --- | --- | --- | --- |
| Conditional horizontal spectral center at CU | Source-specific hierarchical Fourier model with Matérn-5/2 source/path residual structure and centered periodic azimuth term | `Mw`, `Rrup`, `Ztor/depth`, source type, path azimuth | `variants/<variant>/spectral_model.joblib` | `spectral_core/src/gm_parametric_3c/fas_spectral.py` |
| Site amplification and persistent local spectral departure | Broad `Ts`-conditioned amplification plus normalized-frequency local texture conditioned in `XY-Ts` space | `Ts`, conditioned site coordinates | Embedded in `spectral_model.joblib` | `site_texture.py`, `site_texture_base.py` |
| 3C spectral ratios | Learned Major/Intermediate and Vertical/EAS layers | scenario and site spectral state | Embedded in `spectral_model.joblib` | `fas_functional_hierarchical.py` |
| Spectral realization variability | Between-event curves, record functional modes, component-ratio modes, empirical fine-scale 3C residual exemplar | source type, scenario, site, selected residual draws | Embedded spectral artifact + `models/release/empirical_fas/empirical_fas_library.npz` | `fas_carrier.py`, `runtime/vallis3c_runtime/empirical_fas.py` |
| Unified horizontal and vertical duration model | Source-specific portable multioutput Random Forest | `Mw`, `ln(Rrup)`, depth, `ln(Ts)` | `temporal/duration_clock_final_hv_portable.npz` | `runtime/vallis3c_runtime/duration_clock.py` |
| Time-frequency texture selector | Source-specific ExtraTrees ensemble | scenario/site variables + 12 FAS nodes for each of M, I, V | `temporal/selector_portable.npz` | `runtime/vallis3c_runtime/selector.py` |
| Empirical temporal texture | Processed 3C energy/persistence texture constructed from the research database | selected donor from conditional ExtraTrees support | `models/release/textures/` | selector + structural carrier runtime |
| Stochastic carrier and synthesis | Phase-diffusion synthesis + analytic band transport | target FAS, duration scales, selected texture | runtime | `engine.py`, `surface_carrier.py`, `synthesis.py`, `numerics.py` |


### 3.1 Record-level calibration and runtime-support populations


The release distinguishes event-station membership by scientific role rather than treating the complete database as a single interchangeable population. The machine-readable crosswalk is `validation/model_training_usage_manifest.csv`. Each record can carry one or more of the following codes:


| Code | Scientific role | Records | Events | Source split |
| --- | --- | ---: | ---: | --- |
| `F` | Conditional spectral/site-FAS population, including the broad `Ts`-conditioned site term and persistent local spectral departure | 2,224 | 76 | 1,148 interplate / 1,076 intraslab |
| `C` | CU reference-FAS calibration | 68 | 68 | 44 interplate / 24 intraslab |
| `R` | Major/Intermediate and Vertical/EAS spectral-ratio calibration | 2,195 | 47 | 1,124 interplate / 1,071 intraslab |
| `S` | ExtraTrees time-frequency selector training population | 1,623 | 42 | 839 interplate / 784 intraslab |
| `T` | Production-eligible time-frequency texture donors | 1,623 | 42 | 839 interplate / 784 intraslab |
| `D` | Unified horizontal and vertical duration-model calibration | 1,623 | 42 | 839 interplate / 784 intraslab |


The `R` population is exactly the subset belonging to the 47 earthquakes represented at two or more spectrally eligible stations. The 2,224-record spectral population contains 29 one-station earthquakes; excluding those 29 singleton observations gives 2,195 records from 47 events. Singleton events remain part of `F` and, for CU observations, `C` where applicable.


The `S`, `T` and `D` populations contain the same 1,623 event-station observations while identifying different scientific roles: selector fitting, empirical texture donation and duration-model calibration. The runtime admits only complete three-component texture donors.


A single cell in the supplementary event-station matrix can therefore contain several codes, for example `FCRSTD` when the same observation contributes to all six roles.


The temporal catalog stores record identifiers, preprocessed relative/archive paths, and SHA256 values for the full temporal catalog; the `S`, `T`, and `D` flags identify the 1,623 observations used by the temporal models. Three early CU observations (1968-02-03, 1976-02-01, and 1978-11-29) contribute to the spectral CU calibration (`F` and `C`) but have no usable three-component temporal representation and therefore do not carry `S`, `T`, or `D`. The 2025-08-02/LI58 metadata use `Ts = 1.83 s`. The source record is `INCOMPLETE` because its end is truncated, so it is not included in selector training, production texture donation, or unified duration-model training.


The supplementary `Table S1.xlsx` mirrors these codes in each event-station cell and provides a human-readable legend. The CSV manifest is the authoritative record-level audit because it can be checked programmatically against the packaged spectral and temporal artifacts.


## 4. Conditional three-component Fourier model


### 4.1 CU reference motion


The horizontal reference quantity is EAS,


`EAS = sqrt((F_M^2 + F_I^2)/2)`,


and the conditional center is calibrated separately for interplate and intermediate-depth intraslab earthquakes.


The production architecture identifier is:


`gm_parametric_3c.fas.vallis3c`


The current FULL spectral artifact reports:


- 2,224 training records from 76 earthquakes;
- 68 CU earthquakes used by the CU reference layer;
- 44 interplate and 24 intraslab CU events;
- frequency support from 0.1 to 20 Hz.


The interplate CU model uses no explicit source-spectrum prior. The intraslab model uses the fixed regional omega-squared normalization from García et al. (2004) only as a weak shape prior; source, path, depth, and azimuth dependence remain data-driven.


The source/path layer is implemented in:


`spectral_core/src/gm_parametric_3c/fas_spectral.py`


The fitted release report is:


`models/release/variants/full/spectral_training_report.json`


### 4.2 Site amplification and local spectral structure


The site layer separates:


1. broad amplification conditioned on dominant site period `Ts`; and
2. a persistent local spectral departure represented in normalized frequency `u = f Ts`.


The production local texture uses conditioned spatial coordinates and `Ts`; it is not a stochastic spatial covariance field. The normalized-frequency support is `u = 0.35-4.0`, applied fully over `0.50-3.0` and tapered toward the identity near the support edges.


The FULL release site-texture audit reports:


- 63 stations contributing to the persistent local-texture validation;
- 1,912 residual records;
- leave-one-station-out profile RMSE ≈ 0.149;
- peak correlation ≈ 0.858;
- peak-location/amplitude diagnostic MAE ≈ 0.141.


The compact released validation is:


`validation/01_spectral_model/site_amplification_loso.csv`


The implementation is in:


`spectral_core/src/gm_parametric_3c/site_texture.py`


### 4.3 Component spectra and residual variability


The horizontal EAS center is combined with learned frequency-dependent Major/Intermediate and Vertical/EAS ratios to reconstruct the three component spectra.


Realization variability is added after the conditional center through:


- one complete between-event EAS residual curve per simulated earthquake realization;
- frequency-correlated record-level functional modes;
- functional component-ratio modes;
- one eligible empirical three-component fine-scale spectral residual exemplar.


The event curve may be shared across sites for a common simulated earthquake realization. Record-level draws are site-specific; VALLIS-3C does not impose an explicit cross-site covariance kernel on those stochastic record-level modes.


Key code:


- `fas_functional_hierarchical.py`
- `fas_carrier.py`
- `runtime/vallis3c_runtime/empirical_fas.py`


## 5. Unified horizontal and vertical duration model


The production duration artifact is:


`models/release/variants/<variant>/temporal/duration_clock_final_hv_portable.npz`


with manifest:


`duration_clock_final_hv_manifest.json`


The runtime feature vector is:


`[Mw, ln(Rrup), depth_used, ln(Ts)]`.


The portable model is source-specific and contains one horizontal log-duration target plus eight logarithmic V/H duration-ratio outputs corresponding to synthesis-bank indices 31–38, whose center frequencies span approximately 5.5–13.6 Hz.


For the FULL release, the horizontal duration clock uses 512-tree Random Forests with:


- INSLAB: `min_samples_leaf = 15`, 784 eligible records, 19 events;
- INTERPLATE: `min_samples_leaf = 40`, 839 eligible records, 23 events.


Event-grouped OOF performance reported by the production manifest is:


| Source | OOF RMSE in ln(D_H) | event-balanced mean log bias |
| --- | ---: | ---: |
| INSLAB | 0.224 | -0.003 |
| INTERPLATE | 0.201 | -0.002 |


These values are also provided in:


`validation/02_duration_model/duration_oof_summary.csv`


### Upper-tail behavior


The Random Forest is not used as an unconstrained extrapolator outside dense high-magnitude support.


- For interplate scenarios, magnitude is capped at the calibrated model limit before forest evaluation and the upper-tail duration follows a nonnegative site-period dependence extracted from the trained forest.
- For sparse intraslab upper-tail scenarios, the duration anchor is additionally constrained by a data-derived magnitude-distance floor.


The complete inference rule is implemented in:


`runtime/vallis3c_runtime/duration_clock.py`


## 6. ML-guided three-component time-frequency texture selection


The production selector is:


`models/release/variants/<variant>/temporal/selector_portable.npz`


The corresponding training metadata are stored in:


`models/release/variants/<variant>/temporal/training_manifest.json`


For the FULL release, the selector uses:


- `ExtraTreesRegressor`;
- separate source-specific ensembles;
- 128 trees per source;
- `min_samples_leaf = 3`;
- event-balanced training;
- 44 input features;
- 54 temporal descriptors.


The fitted FULL selector uses 1,623 eligible records from 42 earthquakes: 784 INSLAB records from 19 events and 839 INTERPLATE records from 23 events. OOS2017 uses 1,568 eligible records from 41 earthquakes after excluding 19 September 2017. OOS2012_2017 uses 1,515 eligible records from 40 earthquakes: 729 INSLAB records from 18 events and 786 INTERPLATE records from 22 events after excluding both validation events from fitting and donor support.

### 6.1 Inputs


The 44 inputs are:


- 8 scenario/site quantities:
  - `Mw`
  - `ln(Rrup)`
  - source depth
  - `ln(Ts)`
  - conditioned `X,Y`
  - event longitude and latitude
- 12 logarithmically spaced FAS ordinates for each of Major, Intermediate, and Vertical between 0.2 and 8 Hz.


### 6.2 Targets and runtime role


The 54 training targets summarize temporal morphology rather than waveform samples:


- component energy-duration descriptors;
- relative energy-accumulation times;
- real and imaginary parts of narrowband complex persistence at 0.2, 0.5, 1, 2, 4, and 8 Hz for the three components.


The ExtraTrees predictions are not inverted into an accelerogram. At runtime, the leaves reached by the conditioned query define a local empirical support. One eligible record-indexed three-component texture is selected from that support using event-balanced weights.


The selected object is a processed temporal-energy texture used by the synthesis model; it does not contain the original acceleration time history or recorded phase history.


The runtime implementation is:


`runtime/vallis3c_runtime/selector.py`


## 7. Phase-diffusion synthesis


VALLIS-3C 1.6.0 uses one phase-diffusion carrier for synthesis.


The selected empirical texture defines the three-component bandwise energy organization. New stochastic phase trajectories are generated independently; no recorded phase trajectory is transferred.


The synthesis then performs:


1. 42-band analytic decomposition / carrier construction over 0.1-20 Hz;
2. preliminary synthesis at the conditional horizontal duration center;
3. ensemble ranking of preliminary horizontal durations;
4. mapping of those ranks to the empirical source-specific duration-residual distribution;
5. one globally adjusted time scale per realization;
6. band-dependent one-sided Vertical H/V duration transport;
7. regeneration with the same stochastic carrier and analytic phase rotation;
8. smooth FAS correction toward the realization-specific target;
9. zero-lag covariance correction and Major/Intermediate reassignment.


Because covariance correction follows the smooth spectral correction, completed-history FAS agreement is evaluated on the generated records rather than being exact by construction.


Primary runtime files:


- `runtime/vallis3c_runtime/engine.py`
- `runtime/vallis3c_runtime/surface_carrier.py`
- `runtime/vallis3c_runtime/synthesis.py`
- `runtime/vallis3c_runtime/numerics.py`


## 8. FULL and OOS variants use the same architecture


The OOS variants are not separate methods. They use the same VALLIS-3C architecture fitted on explicitly reduced populations. OOS2017 excludes the 19 September 2017 Puebla–Morelos earthquake. OOS2012_2017 excludes both that event and the 20 March 2012 Ometepec earthquake from every data-dependent model and eligible donor population used in the strict validations.


Model paths:


- FULL: `models/release/variants/full/`
- OOS2017: `models/release/variants/oos_2017/`
- OOS2012_2017: `models/release/variants/oos_2012_2017/`


The installation audit verifies that each OOS target event is absent from its donor catalog and empirical-FAS donor pool.


## 9. Validation crosswalk


| Architecture block / claim | Compact validation data | Manuscript role |
| --- | --- | --- |
| CU conditional spectral center | `validation/01_spectral_model/` | Section 2.2 / Table 1; external benchmarks and leave-one-event-out baseline tests |
| Persistent local site amplification | `validation/01_spectral_model/site_amplification_loso.csv` | Site-layer LOSO audit supporting Section 2.2 |
| Horizontal duration prediction | `validation/02_duration_model/` | Section 2.3; grouped OOF archive, baselines and residual diagnostics |
| Complete generator with all learned/data-driven stages excluded from 2017 | `validation/03_oos_2017/` | Sections 3.1-3.2 / Figs. 4-7; strict OOS metrics and station-bootstrap summary |
| Complete generator with all learned/data-driven stages excluded from 2012 and 2017 | `validation/04_oos_2012/` | Supplementary strict OOS network metrics and station-bootstrap summary for 2012 |
| Temporal variability and engineering comparison with regulatory motions | `validation/05_sasid/` | Section 3.3 / Table 2 / Figs. 8-10 |


`VALIDATION_REPORT.json` at repository root is different: it is an installation/runtime integrity check, not the scientific validation dataset.


## 10. Where the exact fitted parameters live


For release reproducibility, use the packaged manifests rather than copying values from this overview:


- spectral architecture and training report:
  `models/release/variants/<variant>/spectral_training_report.json`
- duration clock:
  `models/release/variants/<variant>/temporal/duration_clock_final_hv_manifest.json`
- selector / temporal training metadata:
  `models/release/variants/<variant>/temporal/training_manifest.json`
- top-level variant mapping:
  `models/release/model_manifest.json`
- texture-part mapping:
  `models/release/textures/texture_manifest.json`


These files, together with `SHA256SUMS.txt`, are the authoritative release-level record of the fitted artifacts.


## 11. Data provenance


VALLIS-3C does not include the original RAII-UNAM or RACM/CIRES ground-motion recordings. Learned/statistical models and processed spectral and temporal research libraries are packaged for reproducibility. Ground-motion source attribution is documented in `DATA_PROVENANCE.md`.


For a first technical audit of the software, the recommended reading order is:


1. companion manuscript, Sections 2.2-2.6;
2. this `MODEL_ARCHITECTURE.md`;
3. `validation/README.md`;
4. the relevant production manifest;
5. the runtime/source file listed in the tables above.
