"""VALLIS-3C 1.6.0 host adapter."""
from __future__ import annotations
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]

CORE_SRC=ROOT/'spectral_core'/'src'
if not (CORE_SRC/'gm_parametric_3c').is_dir():
    raise FileNotFoundError('Packaged VALLIS-3C spectral core missing')
if str(CORE_SRC) in sys.path:
    sys.path.remove(str(CORE_SRC))
sys.path.insert(0,str(CORE_SRC))

from . import VallisEngine,VallisConfig,ModelFASProvider,SuppliedFASProvider,SurfaceCarrierEngine
from .config import fas_preset,FAS_RUNTIME_INVARIANTS

VARIANTS={
    'full':{
        'heldout':set(),
        'fas':Path('release/variants/full/spectral_model.joblib'),
        'temporal':Path('release/variants/full/temporal'),
    },
    'oos2017':{
        'heldout':{'2017-09-19'},
        'fas':Path('release/variants/oos_2017/spectral_model.joblib'),
        'temporal':Path('release/variants/oos_2017/temporal'),
    },
    'oos2012_2017':{
        'heldout':{'2012-03-20','2017-09-19'},
        'fas':Path('release/variants/oos_2012_2017/spectral_model.joblib'),
        'temporal':Path('release/variants/oos_2012_2017/temporal'),
    },
}

def _install_fas_override(engine,model,variant):
    seen=set();stack=[engine];changed=0
    while stack:
        obj=stack.pop()
        if obj is None or id(obj) in seen:continue
        seen.add(id(obj))
        for name in ('fas_mean','fas_residual'):
            if hasattr(obj,name):
                setattr(obj,name,model);changed+=1
        if hasattr(obj,'resonance_texture_model'):
            setattr(obj,'resonance_texture_model',getattr(model,'site_texture_model',None));changed+=1
        if hasattr(obj,'active_fas_architecture'):
            setattr(obj,'active_fas_architecture','VALLIS3C')
        for name in ('engine','base_engine','base','wrapped_engine'):
            child=getattr(obj,name,None)
            if child is not None and child is not obj:stack.append(child)
    if not changed:
        raise RuntimeError(f'Could not install {variant} spectral model into the VALLIS-3C carrier')
    try:setattr(engine,'variant',variant)
    except Exception:pass

def _load_variant_fas(engine,final_models_root,variant,heldout):
    from gm_parametric_3c.fas_spectral import FASSpectralModel
    spec=VARIANTS[variant]
    fas_path=final_models_root/spec['fas']
    if not fas_path.is_file():
        raise FileNotFoundError(f'{variant} spectral model missing: {fas_path}')
    model=FASSpectralModel.load(fas_path)
    meta=dict(getattr(model,'artifact_metadata',{}) or {})
    report=dict(getattr(model,'training_report',{}) or {})
    exc=set(map(str,report.get('excluded_event_ids',meta.get('excluded_event_ids',meta.get('excluded_events',[]))) or []))
    if exc != set(heldout):
        raise RuntimeError(f'{variant} FAS provenance mismatch: expected {sorted(heldout)}, got {sorted(exc)}')
    tex=getattr(model,'site_texture_model',None)
    prov=dict(getattr(tex,'provenance',{}) or {}) if tex is not None else {}
    tex_exc=set(map(str,prov.get('excluded_event_ids',[]) or []))
    if tex is None or tex_exc != set(heldout):
        raise RuntimeError(f'{variant} site-texture provenance mismatch: expected {sorted(heldout)}, got {sorted(tex_exc)}')
    _install_fas_override(engine,model,variant)
    return engine

def _build_surface_engine(variant:str):
    from gm_parametric_3c.fas_spectral import FASSpectralModel
    final_models_root=ROOT/'models'
    spec=VARIANTS[variant]
    fas_path=final_models_root/spec['fas']
    if not fas_path.is_file():raise FileNotFoundError(f'{variant} spectral model missing: {fas_path}')
    model=FASSpectralModel.load(fas_path)
    empirical_fas_asset=ROOT/'models'/'release'/'empirical_fas'/'empirical_fas_library.npz'
    surface=SurfaceCarrierEngine(model,empirical_fas_asset,variant=variant)
    _load_variant_fas(surface,final_models_root,variant,set(spec['heldout']))
    return surface

def prepare_transfer_engine(engine=None,variant='full'):
    variant=str(variant or 'full').strip().lower()
    if variant not in VARIANTS:raise ValueError(f'Unsupported model variant: {variant}')
    final_models_root=ROOT/'models'
    if engine is None:return _build_surface_engine(variant)
    return _load_variant_fas(engine,final_models_root,variant,set(VARIANTS[variant]['heldout']))

def simulate_vallis3c_sync(model_dir,request,*,progress=None,cancel=None,**_):
    if 'scenario' not in request:raise ValueError('request.scenario is required')
    request=dict(request)
    cfg=VallisConfig.from_dict(request.get('vallis_config')).validate()
    cfg.exclude_same_event=False
    requested_components=str(request.get('afmp_output_components',cfg.output_components)).strip().lower()
    if requested_components in {'horizontal_only','three_component'}:
        cfg.output_components=requested_components
    variant=str(cfg.model_variant)
    if variant not in VARIANTS:raise ValueError(f'Unsupported model variant: {variant}')
    spec=VARIANTS[variant];heldout_events=set(spec['heldout'])
    request['strict_oos2017']=variant=='oos2017'
    request['strict_oos_events']=sorted(heldout_events)
    request['vallis_config']=cfg.to_dict()

    fas_options=fas_preset()
    fas_options.update(dict(request.get('fas_options',{}) or {}))
    fas_options.update(FAS_RUNTIME_INVARIANTS)
    for k in ('fas_input_mode','exact_fas','city_exact_rock','fas_empirical_on_exact_input','fas_event_seed'):
        if k in request:fas_options[k]=request[k]
    mode=str(fas_options.get('fas_input_mode','model')).strip().lower()
    if mode not in {'model','exact_event'}:raise ValueError(f'Unsupported FAS mode: {mode}')

    if mode=='model':
        fas_options.update(FAS_RUNTIME_INVARIANTS)
        event_conditioning=str(request.get('event_conditioning','')).strip().lower()
        event_mode='record_only' if event_conditioning=='conditional_median_record_only' else str(fas_options.get('fas_event_mode','independent_realization')).strip().lower()
        if event_mode not in {'independent_realization','shared_batch','record_only'}:
            event_mode='independent_realization'
        fas_options['fas_event_mode']=event_mode
        es=fas_options.get('fas_event_seed')
        if event_mode in {'independent_realization','shared_batch'} and es is not None:
            es=int(es)
            if es<0:raise ValueError('fas_event_seed must be nonnegative')
            fas_options['fas_event_seed']=es
        else:
            fas_options.pop('fas_event_seed',None)

    scenario=dict(request['scenario']);caps={}
    final_models_root=ROOT/'models'
    if mode=='exact_event':
        exact=request.get('exact_fas') or fas_options.get('exact_fas')
        if not exact:raise ValueError('Selected-event FAS requires exact_fas M/I/V.')
        provider=SuppliedFASProvider(exact,mode=('city_event_rock_x_site' if bool(request.get('city_exact_rock')) else 'selected_event_exact'))
        fas_options['fas_empirical_on_exact_input']=False
    else:
        from core.path_angle import apply_source_path_theta
        scenario=apply_source_path_theta(scenario,variant=variant);request['scenario']=scenario
        surface=_build_surface_engine(variant)
        initial_caps=dict(surface.capabilities())
        caps={
            'active_fas_architecture':'VALLIS3C',
            'training_exclusions':sorted(heldout_events),
            'spectral_model_path':str(final_models_root/spec['fas']),
            'bootstrap_training_exclusions':list(initial_caps.get('training_exclusions',[]) or []),
        }
        if heldout_events:fas_options['excluded_donor_event_ids']=sorted(heldout_events)
        provider=ModelFASProvider(surface)

    temporal_dir=final_models_root/spec['temporal']
    if not temporal_dir.is_dir():
        raise FileNotFoundError(f'{variant} temporal model directory missing: {temporal_dir}')

    engine=VallisEngine(temporal_dir,provider)
    try:
        payload=engine.generate_one(
            scenario,n_realizations=request.get('n_realizations',1),seed=request.get('seed',0),
            config=cfg.to_dict(),fas_options=fas_options,progress=progress,cancel=cancel
        )
        for item in payload.get('metadata',[]):
            item['model_variant']=variant
            item['strict_oos_events']=sorted(heldout_events)
            item['fas_input_mode']=mode
            if 'source_path_theta_deg' in scenario:
                item.update(
                    source_path_theta_deg=scenario['source_path_theta_deg'],
                    source_path_theta_origin=scenario.get('source_path_theta_origin'),
                    source_path_theta_convention=scenario.get('source_path_theta_convention')
                )
        if not bool(request.get('defer_output_postprocess',False)):
            from sim.signal_ops import stabilize_3c_output_displacement
            import numpy as np
            dt=float(payload.get('dt_s',request.get('dt_s',0.01)))
            corrected,diagnostics=stabilize_3c_output_displacement(payload['accelerations'],dt,payload.get('metadata'))
            payload['accelerations']=corrected
            payload['time_s']=np.arange(corrected.shape[-1],dtype=float)*dt
            payload['output_displacement_stabilization']={
                'enabled':True,'stage':'adapter_before_consumer','tail_padding_s':10.0,'diagnostics':diagnostics
            }
            limited=sum(item.get('reason')!='ok' for item in diagnostics)
            if limited:
                payload.setdefault('warnings',[]).append(
                    f'Displacement-control integration was limited in {limited} component(s); review diagnostics before using PGD.'
                )
        else:
            payload['output_displacement_stabilization']={'enabled':False,'stage':'deferred_to_application_output_conditioning'}
        audit=dict(payload.get('model_audit',{}))
        audit.update({
            'product_version':'1.6.0',
            'conditional_fas_architecture':('VALLIS3C' if mode=='model' else 'supplied_exact_event'),
            'conditional_fas_variant':variant,
            'model_variant':variant,
            'output_components':cfg.output_components,
            'release_model_pair':True,
            'fallback_used':False,
            'models_root':str(final_models_root),
            'strict_oos_events':sorted(heldout_events),
            'conditional_fas_capabilities':caps,
            'fas_event_mode':fas_options.get('fas_event_mode'),
            'fas_event_seed':fas_options.get('fas_event_seed'),
            'carrier':'phase_diffusion',
            'texture_selector':engine.selector.selector_id,
            'texture_selector_artifact':engine.selector.artifact_name,
            'texture_selector_training_records':engine.selector.training_records,
            'texture_selector_min_samples_leaf':engine.selector.min_samples_leaf,
        })
        payload['model_audit']=audit
        return payload
    finally:
        engine.close()

simulate_vallis_sync=simulate_vallis3c_sync
