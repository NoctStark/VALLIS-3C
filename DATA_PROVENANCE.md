# Data sources and research artifacts

## Ground-motion data sources

The ground-motion database used to develop and validate VALLIS-3C combines records from two Mexico City strong-motion networks:

- **RAII-UNAM.** Raw ground-motion recordings from the Red Acelerográfica del Instituto de Ingeniería (RAII-UNAM), operated by the Institute of Engineering at Universidad Nacional Autónoma de México (UNAM), were accessed through the RAII-UNAM network website: https://aplicaciones.iingen.unam.mx/AcelerogramasRSM/Inicio.aspx
- **RACM / CIRES.** Raw ground-motion recordings from the Mexico City Accelerographic Network (RACM), operated by the Centro de Instrumentación y Registro Sísmico, A.C. (CIRES), were requested directly from CIRES. Network information is available at https://www.cires.org.mx/racm_n.php

The original ground-motion recordings are **not included** in the VALLIS-3C software repository or release archive.

## What VALLIS-3C distributes

The repository contains the research products required to run and audit the method, including:

- source code and configuration files;
- fitted spectral and duration models;
- processed spectral and time-frequency libraries used by the runtime;
- model manifests and record-use crosswalks;
- generated VALLIS ground motions used in validation;
- derived spectra, durations, intensity measures, nonlinear-response summaries, and comparison tables used in the manuscript.

These processed libraries are runtime/model artifacts and do not contain the original RAII-UNAM or RACM/CIRES acceleration time histories.

## Data-provider acknowledgement

Publications using VALLIS-3C should acknowledge the organizations that maintain the ground-motion networks used for model development: the Instituto de Ingeniería, UNAM (RAII-UNAM) and the Centro de Instrumentación y Registro Sísmico, A.C. (CIRES/RACM).

RAII-UNAM acknowledgement used by the database:

> Los datos sísmicos fueron proporcionados por la Red Acelerográfica del Instituto de Ingeniería (RAII-UNAM), producto de las labores de instrumentación y procesamiento de la Unidad de Instrumentación Sísmica. Los datos son distribuidos a través del Sistema de Base de Datos Acelerográficos en web: http://aplicaciones.iingen.unam.mx/AcelerogramasRSM/

## Record-level model-use audit

`validation/model_training_usage_manifest.csv` provides the release-level event-station crosswalk for the distinct calibration and runtime-support populations used by VALLIS-3C 1.6.0. The codes are `F` (spectral/site-FAS and persistent local spectral departure), `C` (CU reference FAS), `R` (three-component spectral ratios), `S` (ExtraTrees time-frequency selector training population), `T` (production-eligible time-frequency texture donor), and `D` (unified horizontal and vertical duration-model calibration).

The temporal models use 1,623 eligible FULL observations. OOS2017 uses 1,568 eligible observations and excludes every 19 September 2017 record. OOS2012_2017 uses 1,515 eligible observations and excludes every 20 March 2012 and 19 September 2017 record.

The `R` population contains 2,195 observations from the 47 earthquakes having at least two spectrally eligible stations; the 29 one-station earthquakes in the full 2,224-record spectral population are not used for this component-ratio calibration.

For records present in the temporal catalog, the manifest preserves the preprocessed relative path, archive path, and record SHA-256 used by the packaged workflow. These identifiers support reproducibility and audit of the processed research database; they are not copies of the original ground-motion time histories.

## Mexico City zoning sources

The site-zoning presentation and map interpretation are based on:

- Secretaría de Obras y Servicios de la Ciudad de México (SOS-CDMX) (2023). *Normas Técnicas Complementarias para Diseño por Sismo*. Reglamento de Construcciones de la Ciudad de México. Gaceta Oficial de la Ciudad de México, Mexico City [in Spanish].
- Secretaría de Obras y Servicios de la Ciudad de México (SOS-CDMX) (2023). *Normas Técnicas Complementarias para el Diseño y Construcción de Cimentaciones*. Reglamento de Construcciones de la Ciudad de México. Gaceta Oficial de la Ciudad de México, Mexico City [in Spanish].

The inclusion of regulatory references and data-provider names indicates provenance and acknowledgement only and does not imply endorsement of VALLIS-3C.
