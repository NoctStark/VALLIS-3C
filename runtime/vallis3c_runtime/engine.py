from __future__ import annotations
import copy
import numpy as np,pandas as pd
from .config import VallisConfig,ConfigurationError
from .selector import PortableSelector
from .duration_clock import FinalHVDurationMapper,rank_probabilities
from .hf_clock import VerticalRatioBank
from .numerics import COMPONENTS,match_smooth,decorrelate,metrics
from .synthesis import transport_complex
from .providers import array_hash
from .structural_carriers import StructuralTextureStore

class VallisEngine:
    def __init__(self,model_dir,fas_provider):
        self.selector=PortableSelector(model_dir);self.fas_provider=fas_provider
        from pathlib import Path
        root=Path(model_dir)
        if not (root/'duration_clock_final_hv_portable.npz').is_file():raise FileNotFoundError('Unified duration-model artifact missing')
        if not (root/'hf_vertical_clock_exactbank_5_15.csv').is_file():raise FileNotFoundError('Vertical donor-ratio bank missing')
        self.duration_mapper=FinalHVDurationMapper(root)
        self.vertical_ratio_bank=VerticalRatioBank(root,self.selector)
        self.structural_store=StructuralTextureStore(root,self.selector)
    def close(self):
        self.structural_store.close()
    @staticmethod
    def _d5h(metric):
        c=metric['components'];return float(np.sqrt(c['major']['D5_95_s']*c['intermediate']['D5_95_s']))
    def _finalize(self,y,cfg,f,fas):
        if y.shape[-1]*cfg.dt_s>cfg.max_duration_s:raise ValueError('Support exceeds safety ceiling')
        n_components=y.shape[0];requested_fas=np.asarray(fas)[:n_components]
        if cfg.spectral_correction:y=match_smooth(y,cfg.dt_s,f,requested_fas)
        y,cov=decorrelate(y);perm=list(range(n_components))
        if np.sum(y[1]**2)>np.sum(y[0]**2):perm=[1,0]+list(range(2,n_components));y=y[perm]
        yf=requested_fas[perm].copy()
        if not np.isfinite(y).all():raise ValueError('Nonfinite output')
        met=metrics(y,cfg.dt_s,f,yf);return y,yf,cov,perm,met,self._d5h(met)
    def _metadata(self,scenario,i,r,seed,selection,row,fas,perm,cov,y,met,scale,duration_mapping,hf_extra,cfg,carrier_audit):
        selected=self.selector.catalog.iloc[selection['indices']]
        return {'version':'1.6.0','runtime_configuration':'final-hv-phase-diffusion','method':'transport_complex','architecture':duration_mapping.get('architecture','final_hv'),'scenario':scenario,
            'carrier':'phase_diffusion','carrier_audit':dict(carrier_audit or {}),
            'texture_selector':self.selector.selector_id,'texture_selector_artifact':self.selector.artifact_name,
            'texture_selector_training_records':self.selector.training_records,'texture_selector_min_samples_leaf':self.selector.min_samples_leaf,
            'scenario_index':i,'realization_index':r,'synthesis_seed_entropy':[int(seed),i,r,6000],
            'temporal_scale':float(scale),'duration_mapping':duration_mapping,'final_hv_vertical_transport':hf_extra,
            'complete_donor_only':bool(selection['complete_donor_only']),'tree_reselected_for_complete_donor':bool(selection['tree_reselected_for_complete_donor']),
            'selector_tree':selection['tree'],'selector_node':selection['node'],'texture_record_id':str(row.record_id),
            'texture_event_id':str(row.event_id),'carrier_representation':'vallis3c.phase_diffusion.texture_carrier.v1',
            'training_source_sha256':str(row.sha256),'texture_boundary_warning':bool(row.boundary_warning),
            'texture_temporal_eligible':bool(row.temporal_donor_eligible) if 'temporal_donor_eligible' in row.index else (not bool(row.boundary_warning)),
            'texture_eligibility_mode':selection.get('donor_eligibility_mode','energy'),
            'member_record_ids':selected.record_id.astype(str).tolist(),'member_weights':selection['weights'].tolist(),
            'member_events':selected.event_id.astype(str).tolist(),'imputed_features':selection['imputed_features'],
            'input_features':selection['features'].tolist(),'feature_names':self.selector.feature_names,
            'predicted_log_durations':selection['predicted_targets'][:3].tolist(),'requested_fas_sha256':array_hash(fas),
            'output_to_requested_component_map':perm,'fas_exact_by_construction':False,'rejection_used':False,
            'covariance_correction':cov,'unpadded_npts':y.shape[-1],'postprocess_applied':False,'metrics_native':met}
    def _generate_scenario(self,scenario,i,n_realizations,seed,cfg,target,cancel,progress,done,total):
        max_support_retries=50
        n_components=2 if cfg.output_components=='horizontal_only' else 3
        if cfg.architecture!='final_hv':raise ConfigurationError('Only the unified horizontal and vertical duration model is available in VALLIS-3C 1.6.0')
        mapper=self.duration_mapper
        target_vertical_ratios=mapper.vertical_target_ratios(scenario) if n_components==3 else {}

        def is_support_limit(exc):
            return isinstance(exc,ValueError) and ('safety ceiling' in str(exc) or 'Support exceeds' in str(exc))

        def prepare(r,attempt):
            if cancel and cancel():raise InterruptedError('VALLIS-3C cancelled')
            entropy=[int(seed),i,r,6000]+([attempt] if attempt else [])
            rng=np.random.default_rng(np.random.SeedSequence(entropy));fas=target.amplitudes[r,i];f=target.frequencies_hz
            selection=self.selector.select(scenario,f,fas,rng,cfg.exclude_same_event,cfg.complete_donor_only,'energy')
            idx=selection['donor_index'];row=self.selector.catalog.iloc[idx]
            carrier_entropy=[int(seed),i,r,9017]+([attempt] if attempt else [])
            carrier_rng=np.random.default_rng(np.random.SeedSequence(carrier_entropy))
            a,carrier_audit=self.structural_store.signal(str(row.record_id),cfg.dt_s,carrier_rng)
            carrier_audit['seed_entropy']=carrier_entropy
            center=mapper.center_duration(scenario);base_scale=float(center/mapper.donor_duration(str(row.record_id)))
            state=copy.deepcopy(rng.bit_generator.state)
            # Pass 1 deliberately excludes vertical H/V band transport so the horizontal rank clock is unchanged.
            y0=transport_complex(a,cfg.dt_s,base_scale,rng,cfg.analytic_rotation,cfg.max_duration_s,n_components=n_components)
            _,_,_,_,_,d5h0=self._finalize(y0,cfg,f,fas)
            cond=mapper.condition(scenario,f,fas,cfg.exclude_same_event)
            return {'r':r,'fas':fas,'f':f,'selection':selection,'row':row,'a':a,'base_scale':base_scale,
                    'rng_state':state,'current_D5H_s':d5h0,'condition':cond,'seed_entropy':entropy,'support_retries':attempt,'carrier_audit':carrier_audit}

        def next_supported(r,first_attempt=0):
            for attempt in range(first_attempt,max_support_retries+1):
                try:return prepare(r,attempt)
                except ValueError as exc:
                    if not is_support_limit(exc):raise
            raise RuntimeError(f'VALLIS-3C could not find support ≤ {cfg.max_duration_s:g} s for realization {r} after {max_support_retries+1} seeds.')

        pre=[next_supported(r) for r in range(n_realizations)]
        q=rank_probabilities(np.log([x['current_D5H_s'] for x in pre]));out=[]
        for item,qq in zip(pre,q):
            while True:
                if cancel and cancel():raise InterruptedError('VALLIS-3C cancelled')
                try:
                    cond=item['condition'];desired=mapper.desired_duration(cond,float(qq))
                    scale=float(item['base_scale']*desired/item['current_D5H_s'])
                    rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(item['rng_state'])
                    rel={} if n_components!=3 else self.vertical_ratio_bank.one_sided_relative_scales(target_vertical_ratios,str(item['row'].record_id))
                    y=transport_complex(item['a'],cfg.dt_s,scale,rng,cfg.analytic_rotation,cfg.max_duration_s,rel,n_components=n_components)
                    y,yf,cov,perm,met,final_d5h=self._finalize(y,cfg,item['f'],item['fas'])
                    break
                except ValueError as exc:
                    if not is_support_limit(exc):raise
                    item=next_supported(item['r'],item['support_retries']+1)
            dm={'policy':'conditional_D5_95_quantile_mapping','architecture':cfg.architecture,'passes':1,'population_quantile':float(qq),
                'current_D5H_s':float(item['current_D5H_s']),'desired_D5H_s':float(desired),'final_D5H_s':float(final_d5h),
                'center_D5H_s':float(cond['center_D5H_s']),'support_sigma_ln':float(cond['support_sigma_ln']),
                'sigma_oos_ln':float(cond['sigma_oos_ln']),'alpha':float(cond['alpha']),'support_count':int(cond['support_count']),
                'base_temporal_scale':float(item['base_scale']),'final_temporal_scale':scale}
            vals=np.asarray(list(rel.values()),float) if rel else np.ones(1)
            band_indices=[int(k) for k in self.vertical_ratio_bank.band_indices]
            band_centers_hz=[float(x) for x in self.vertical_ratio_bank.band_centers_hz]
            hf={'integrated_in_final_hv_clock':bool(n_components==3),'configured':bool(n_components==3),
                'band_indices':band_indices,'band_centers_hz':band_centers_hz,
                'band_hz':[band_centers_hz[0],band_centers_hz[-1]],'one_sided_only':True,
                'target_ratios_by_band_index':{str(k):float(v) for k,v in target_vertical_ratios.items()},
                'relative_scale_mean':float(vals.mean()),'relative_scale_min':float(vals.min()),'relative_scale_max':float(vals.max()),
                'relative_scales_by_band_index':{str(k):float(v) for k,v in rel.items()}}
            md=self._metadata(scenario,i,item['r'],seed,item['selection'],item['row'],item['fas'],perm,cov,y,met,scale,dm,hf,cfg,item.get('carrier_audit'))
            md['output_components']=cfg.output_components
            md['synthesis_seed_entropy']=item['seed_entropy']
            md['support_limit_retries']=int(item['support_retries'])
            md['rejection_used']=bool(item['support_retries'])
            out.append((y,yf,md));done+=1
            if progress:progress(done,total)
        return out,done
    def generate(self,scenarios,n_realizations=1,seed=0,config=None,fas_options=None,progress=None,cancel=None):
        cfg=config if isinstance(config,VallisConfig) else VallisConfig.from_dict(config);cfg.validate();s=pd.DataFrame(scenarios).reset_index(drop=True);n_realizations=int(n_realizations)
        if n_realizations<1 or int(seed)<0:raise ConfigurationError('Need positive realization count and nonnegative seed')
        for scenario in s.to_dict('records'):self.selector.assert_validation_scope(scenario,cfg.strict_oos)
        if cfg.strict_oos and hasattr(self.fas_provider,'engine') and getattr(self.fas_provider.engine,'variant','full')=='full':raise ConfigurationError('Strict validation cannot use FULL-trained FAS artifacts')
        if cancel and cancel():raise InterruptedError('VALLIS-3C cancelled')
        target=self.fas_provider.resolve(s,n_realizations,int(seed),fas_options);results=[[] for _ in range(len(s))];done=0;total=len(s)*n_realizations
        for i,scenario in enumerate(s.to_dict('records')):results[i],done=self._generate_scenario(scenario,i,n_realizations,seed,cfg,target,cancel,progress,done,total)
        payloads=[]
        for i in range(len(s)):
            n_components=2 if cfg.output_components=='horizontal_only' else 3
            count=max(x[0].shape[-1] for x in results[i]);a=np.zeros((n_realizations,n_components,count))
            for r,(y,_,_) in enumerate(results[i]):a[r,:,:y.shape[-1]]=y
            payloads.append({'accelerations':a,'time_s':np.arange(count)*cfg.dt_s,'dt_s':cfg.dt_s,'units':'cm/s2',
              'component_names':COMPONENTS[:n_components],'target_frequencies_hz':target.frequencies_hz.copy(),'target_fas':np.stack([x[1] for x in results[i]]),
              'requested_target_fas':target.amplitudes[:,i].copy(),'metadata':[x[2] for x in results[i]],'fas_provider':target.metadata,
              'config':cfg.to_dict(),'fas_exact_by_construction':False,'scenario':s.iloc[i].to_dict()})
        return payloads
    def generate_one(self,scenario,**kwargs):return self.generate([scenario],**kwargs)[0]
