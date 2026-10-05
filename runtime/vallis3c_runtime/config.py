from __future__ import annotations
from dataclasses import dataclass,asdict
import copy,json
from pathlib import Path
import numpy as np

PRODUCT_VERSION='1.6.0'
ENGINE_FAMILY='VALLIS-3C'
ARCHITECTURE='final_hv'

class ConfigurationError(ValueError):
    pass

ROOT=Path(__file__).resolve().parents[1]

@dataclass
class VallisConfig:
    dt_s: float=0.01
    max_duration_s: float=1200.0
    exclude_same_event: bool=False
    strict_oos: bool=False
    model_variant: str='full'
    spectral_correction: bool=True
    analytic_rotation: bool=True
    complete_donor_only: bool=True
    architecture: str=ARCHITECTURE
    output_components: str='three_component'

    def validate(self):
        if not np.isfinite(self.dt_s) or not (0.001<=self.dt_s<0.025):
            raise ConfigurationError('dt_s must be >=0.001 and <0.025 s for the fixed 20 Hz band.')
        if not np.isfinite(self.max_duration_s) or self.max_duration_s<=0:
            raise ConfigurationError('max_duration_s must be a positive numerical safety ceiling.')
        if self.architecture!=ARCHITECTURE:
            raise ConfigurationError('Only the unified horizontal and vertical duration architecture is supported in VALLIS-3C 1.6.0.')
        if self.model_variant not in ('full','oos2017','oos2012_2017'):
            raise ConfigurationError('Unsupported model_variant')
        if self.output_components not in ('three_component','horizontal_only'):
            raise ConfigurationError("output_components must be 'three_component' or 'horizontal_only'")
        self.strict_oos=self.model_variant!='full'
        # Same-event donors are part of the packaged sampling population.
        # Enforce the donor policy for every persisted configuration.
        self.exclude_same_event=False
        for key in ('exclude_same_event','strict_oos','spectral_correction','analytic_rotation','complete_donor_only'):
            if not isinstance(getattr(self,key),bool):
                raise ConfigurationError(f'{key} must be bool')
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls,d=None):
        d=copy.deepcopy(d or {})
        variant=str(d.get('model_variant') or ('oos2017' if bool(d.get('strict_oos',False)) else 'full')).strip().lower()
        clean={k:v for k,v in d.items() if k in cls.__dataclass_fields__}
        clean['architecture']=ARCHITECTURE
        clean['model_variant']=variant
        clean['strict_oos']=variant!='full'
        clean['exclude_same_event']=False
        return cls(**clean).validate()

FAS_DEFAULTS={
    'fas_input_mode':'model',
    'fas_variability_source':'surface_hybrid',
    'fas_detail_strength':1.0,
    'fas_residual':True,
    'fas_resonance_texture':True,
    'fas_resonance_texture_residual':True,
    'fas_resonance_texture_residual_scale':1.0,
    'fas_event_mode':'independent_realization',
    'fas_mean_mode':'default',
    'fas_site_mode':'ts_only',
    'fas_empirical_on_exact_input':False,
    'fas_donor_k':30,
    'fas_donor_source_match':True,
}

FAS_RUNTIME_INVARIANTS={
    'fas_residual_scale':1.0,
    'fas_resonance_texture_site_strength':1.0,
    'fas_broad_strength':1.0,
    'fas_split_decades':0.18,
}
FAS_PRODUCTION_FIXED=FAS_RUNTIME_INVARIANTS

def fas_preset():
    path=ROOT/'configs/fas_defaults.json'
    d=json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else {}
    d.update(FAS_DEFAULTS)
    d.update(FAS_RUNTIME_INVARIANTS)
    return d
