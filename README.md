# VALLIS-3C 1.6.0


Self-contained software for three-component ground-motion simulation at sites in the Basin of Mexico.


## Software overview


This distribution contains the self-contained **VALLIS-3C 1.6.0** application and its scientific runtime. It does not require model or dataset paths outside this directory.


Main directories:


- `runtime/` — VALLIS-3C 1.6.0 simulation runtime;
- `spectral_core/` — packaged FAS model core;
- `models/` — scientific models, compact empirical runtime assets and strict out-of-sample variants;
- `validation/` — compact scientific validation datasets corresponding to the manuscript benchmarks and figures;
- `MODEL_ARCHITECTURE.md` — human-readable map from manuscript formulation to packaged models, runtime code, and validation evidence;
- `core/` — orchestration, spatial conditioning and scientific output services;
- `sim/` and `app/` — signal processing and user interface;
- `outputs/` — user-generated results.


## Release status

VALLIS-3C 1.6.0 is the public software release of the method described in the companion manuscript. The original ground-motion recordings used for model development are not included in the software distribution; their sources and access routes are documented in `DATA_PROVENANCE.md`.

The VALLIS-3C software is released under GPL-3.0-only.

Repository: [github.com/NoctStark/VALLIS-3C](https://github.com/NoctStark/VALLIS-3C)


## Scientific workflow


VALLIS-3C generates suites of three-component ground motions for a specified earthquake scenario and site. The workflow combines scenario-dependent spectral amplitudes, site effects, a unified horizontal and vertical duration model, phase-diffusion synthesis, and correlated component-to-component variability.


Available variants under the same `final_hv` architecture:


- `FULL` — complete training set;
- `OOS2017` — excludes 19 September 2017;
- `OOS2012_2017` — excludes 20 March 2012 and 19 September 2017.


The distribution contains FULL, OOS2017 and OOS2012_2017; the production interface exposes FULL only. The two OOS variants are retained exclusively for reproducible validation. OOS2017 excludes every 19 September 2017 record, while OOS2012_2017 excludes every 20 March 2012 and 19 September 2017 record from its fitted models and eligible record populations.


## Model architecture and traceability


See `MODEL_ARCHITECTURE.md` for the technical map of the conditional spectral model, site-amplification layer, unified horizontal and vertical duration model, ExtraTrees time-frequency selector, phase-diffusion synthesis, all three model variants, and the validation files associated with each block. Exact fitted coefficients and model-specific numerical settings remain authoritative in the manifests distributed beside each model artifact.


Record-level membership of the calibration and runtime-support populations is auditable in `validation/model_training_usage_manifest.csv`. The compact usage codes are: `F` = spectral/site-FAS population (including the persistent local spectral-departure layer), `C` = CU reference-FAS calibration, `R` = three-component spectral-ratio calibration, `S` = ExtraTrees time-frequency selector training population, `T` = production-eligible time-frequency texture donor, and `D` = unified horizontal and vertical duration-model calibration. The temporal models use 1,623 eligible FULL records (784 INSLAB; 839 INTERPLATE). OOS2017 uses 1,568 eligible records after exclusion of 19 September 2017. OOS2012_2017 uses 1,515 eligible records (729 INSLAB; 786 INTERPLATE) after excluding both validation events. The manifest preserves the temporal preprocessed path and SHA256 wherever that record enters the temporal catalog; the original ground-motion recordings are not included in the repository.


## Synthesis workflow


VALLIS-3C uses one production carrier: **phase-diffusion synthesis**. It is fixed internally and is not exposed as a user-selectable setting.


The ML selector uses a processed time-frequency texture library constructed from the research database. The library contains the temporal descriptors required by the selector and does not contain the original acceleration time histories. Fresh band phases then evolve stochastically at a diffusion rate fixed by the bandwidth of the native logarithmic filter bank. The generated carrier continues through the unified horizontal and vertical duration model, duration quantile mapping, complex transport, covariance correction, output conditioning, FAS/response-spectrum calculation and export.

The temporal selector and empirical donor pool use the same eligibility population. The runtime admits only complete three-component donors. Physical validation is provided by `validation/03_oos_2017/`, `validation/04_oos_2012/` and `validation/05_sasid/`.


The structural texture library contains 1,629 unique records: the exact union reachable by the FULL, OOS2017 and OOS2012_2017 selectors. It is distributed as eight independently readable HDF5 parts below 95 MiB each. Each individual file is below GitHub's 100 MiB hard limit, but the binary runtime assets are large and should be handled deliberately when the public repository strategy is finalized. The runtime discovers the part list from `models/release/textures/texture_manifest.json`.


## Output conditioning


- Output time step `dt`.
- Configurable zero padding before and after each motion; the default is 20 s at each end.
- Optional zero-phase Butterworth bandpass.
- Standard edge conditioning is applied automatically to minimize end effects and residual drift.


## Duration model


Horizontal and vertical durations are generated jointly from the earthquake and site scenario using the unified horizontal and vertical duration model distributed with VALLIS-3C. The vertical component is represented through frequency-dependent V/H duration ratios.


## Installation and startup on Windows


Run `INSTALL_VALLIS.cmd` once before the first launch. The installer uses an existing 64-bit CPython 3.12, 3.13 or 3.14 installation, creates one shared private environment at `%LOCALAPPDATA%\VALLIS-3C\1.6.0\venv`, installs the exact versions in `requirements.txt`, checks the environment and executes the packaged numerical validation. Administrator privileges are not required. Internet access is required during the initial installation. After installation, double-click `VALLIS-3C.cmd` to start the application.


Run `INSTALL_VALLIS.cmd` again whenever the private environment needs verification or repair. The installer immediately reuses a complete environment and does not reinstall or redownload packages. Copies of the same VALLIS-3C version share this environment, so extracting the program again does not duplicate the dependencies. If a supported Python installation is unavailable, install 64-bit Python 3.12, 3.13 or 3.14 from python.org and rerun the installer. The private environment does not alter or replace the user's other Python environments.


For installation transparency, `VALLIS-3C.cmd` only starts the already-created private environment. `INSTALL_VALLIS.cmd` does not request elevation, install Python, modify `PATH` or the registry, create services, or add startup tasks. It stores one version-scoped environment under `LOCALAPPDATA`, asks pip to install binary distributions from `requirements.txt` without retaining a download cache, checks package consistency, and runs the numerical self-test.


Optionally run `%LOCALAPPDATA%\VALLIS-3C\1.6.0\venv\Scripts\python tools\validate_installation.py` from the application folder to verify FULL, OOS2017 and OOS2012_2017 through the phase-diffusion synthesis flow. The local machine-readable result is stored in `outputs/VALIDATION_REPORT.json`; the report at repository root is the reference self-test for this version.


User preferences are stored outside the repository at `%LOCALAPPDATA%\VALLIS-3C\1.6.0\settings.json`. The packaged `config/default_settings.json` contains only release defaults, so changing application preferences does not modify a Git working tree.


## Scientific validation data


The `validation/` directory contains the compact numerical products used to audit the scientific results reported in the companion manuscript: the regional spectral benchmark, the out-of-fold duration-clock check, strict out-of-sample evaluations for 20 March 2012 and 19 September 2017, and the SASID engineering comparison. The original RAII-UNAM and RACM/CIRES ground-motion recordings are not included in the validation directory. See `validation/README.md` for the file-level description.


`VALIDATION_REPORT.json` is a separate installation/runtime integrity test and should not be interpreted as the scientific validation dataset.


## Numerical and component conventions


Elastic pseudo-spectral acceleration is evaluated with the Generalized Single-Step Single-Solve (GSSSS) structural-dynamics algorithm described by Zhou and Tamma (2004), *International Journal for Numerical Methods in Engineering*, 59(5), 597–668, https://doi.org/10.1002/nme.873.


Major and Intermediate follow the horizontal principal-axis convention of Rezaeian and Der Kiureghian (2012): the orthogonal horizontal component with the larger Arias intensity is Major and the other is Intermediate. This convention is used in both model development and generated accelerograms and does not prescribe the orientation of either axis relative to the source. Reference: *Earthquake Engineering & Structural Dynamics*, 41(2), 335–353, https://doi.org/10.1002/eqe.1132.


Displayed smoothed FAS use the logarithmic weighting function of Konno and Ohmachi (1998), *Bulletin of the Seismological Society of America*, 88(1), 228–241, https://doi.org/10.1785/bssa0880010228. The weighted averages use every native FFT ordinate in the configured frequency band and are evaluated at 320 logarithmically spaced output frequencies. Components of one realization share the same frequency grid and smoothing weights.


## Reproducible environment


The dependency installer supports 64-bit CPython `3.12`, `3.13` and `3.14` and reproduces the exact package versions in `requirements.txt`. The reference release used CPython `3.12.10`; its complete captured environment is recorded in `environment.json`. Every installation runs the packaged numerical validation because the released scientific models use joblib serialization. Python versions outside the supported interval are not used to create the private environment.


## Calibration domain


The application checks Mw, Rrup, Ztor (depth to the top of rupture, shown as **Depth** in the interface), Ts, source type, event coordinates when used, path angle, and the site training-support hull before generation. Within-domain scenarios are marked `IN SUPPORT`. Extrapolations are marked explicitly and require user acceptance; they are never executed silently.


## Data provenance


Raw ground-motion recordings from the RAII-UNAM network, operated by the Institute of Engineering at UNAM, were accessed through the RAII-UNAM website. Raw ground-motion recordings from the Mexico City Accelerographic Network (RACM), operated by CIRES, were requested directly from CIRES. The original ground-motion recordings are not included in the VALLIS-3C repository. The repository contains the software, fitted models, processed research libraries, generated simulations, and numerical validation products used by the study. See `DATA_PROVENANCE.md` for source attribution and the SOS-CDMX zoning references.


## License and scientific use


Citation metadata for this software release are provided in `CITATION.cff`. Cite the archived, version-specific software record when a persistent archive identifier is available.


This unpacked directory is the intended GitHub repository content. The versioned ZIP archive and its SHA256 checksum are distribution assets for the GitHub Release and corresponding Zenodo software record; the ZIP is not stored inside the Git repository. Local environments, generated outputs, original ground-motion recordings, private working files, and development-only rebuild material are not distribution content.


Copyright © 2026 Joel D. Cruz-Arguelles.


VALLIS-3C is distributed under the **GNU General Public License v3.0 only** (`GPL-3.0-only`). See `LICENSE` for the software terms, `DATA_PROVENANCE.md` for the origin of the ground-motion data used in model development, `THIRD_PARTY_NOTICES.md` for dependency notices, and `DISCLAIMER.md` for the scientific and engineering limitations of use.


The software is intended for research and methodological evaluation. It is not a certified seismic-hazard or building-code compliance product and does not replace professional site-specific engineering assessment.
