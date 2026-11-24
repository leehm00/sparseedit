"""Independent source/semantic and saved-trajectory candidate admission."""
from pathlib import Path
import math
import protocol as p

def validate_cells(rows,tags,arms,repetitions):
 measured=[r for r in rows if not r.get('warmup')];keys=[(r.get('case_tag'),r.get('arm'),r.get('repetition')) for r in measured];expected={(t,a,j) for t in tags for a in arms for j in range(repetitions)}
 if len(keys)!=len(expected) or set(keys)!=expected or any(r.get('status')!='PASS' or r.get('native_execution') is not True for r in measured):raise ValueError('complete unique native source/arm/repetition coverage required')

def validate_trace_application(md,trace):
 p.validate_trace(trace);rows=md.get('step_records') or [];summary=md.get('summary') or {};config=md.get('run_config') or {}
 if len(rows)!=28 or any(summary.get(k)!=v for k,v in {'transformer_compute_steps':28,'transformer_skipped_steps':0,'dense_steps':9,'sparse_steps':19}.items()):raise ValueError('actual frozen compute work differs')
 if config.get('latent_dtype')!='torch.bfloat16' or config.get('true_cfg_scale')!=4. or config.get('cfg_execution_plan',{}).get('selected')!='serial':raise ValueError('actual BF16 serial CFG4 differs')
 for got,want in zip(rows,trace['steps']):
  for k in ('step','phase','kv_action'):
   if got.get(k)!=want[k]:raise ValueError('actual trace chronology differs')
  if got.get('cfg_branch_count')!=2 or got.get('transformer_computed') is not True:raise ValueError('actual CFG/compute differs')
  if want['phase']=='sparse' and (got.get('indices')!=want['indices'] or got.get('active_token_count')!=460):raise ValueError('actual source-specific mask/index differs')
 return True

def compare_trajectories(left,right):
 import torch
 differences=[];exact=True
 for i in range(28):
  a=torch.load(Path(left)/f'step_{i:02d}.pt',map_location='cpu',weights_only=True);b=torch.load(Path(right)/f'step_{i:02d}.pt',map_location='cpu',weights_only=True)
  for k in ('positive_compact','negative_compact','guided_velocity','latents'):
   x,y=a[k],b[k]
   if x.shape!=y.shape or x.dtype!=y.dtype or not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('saved numerical trajectory shape/dtype/finite differs')
   same=torch.equal(x,y);exact&=same;delta=x.double()-y.double();differences.append({'step':i,'field':k,'exact':same,'unequal':int((x!=y).sum()),'max_abs':float(delta.abs().max()),'relative_l2':float(torch.linalg.vector_norm(delta)/max(float(torch.linalg.vector_norm(x.double())),1e-30))})
 return {'exact':exact,'fields':differences}

def admit_cal3(rows,tags):
 import json
 lookup={(r['case_tag'],r['arm']):r for r in rows if not r['warmup']};admissions={};eligible=[]
 for c in p.CANDIDATES:
  checks=[];failure=[]
  for tag in tags:
   base=lookup.get((tag,'standard_fastops_eager'));candidate=lookup.get((tag,c))
   if not base or base['status']!='PASS':raise ValueError('all3 fastops norm oracle cases must pass')
   if not candidate or candidate['status']!='PASS':failure.append(candidate or {'case_tag':tag,'failure':'missing candidate'});continue
   backend=candidate['request']['runtime_metadata']['standard_backend'];base_backend=base['request']['runtime_metadata']['standard_backend'];comparison=compare_trajectories(Path(base['audit_directory'])/'base',Path(candidate['audit_directory'])/'base');exact=comparison['exact'] and candidate['output_sha256']==base['output_sha256']
   local=True
   if c=='standard_ff_compile':
    oracle=lookup.get((tag,'standard_ff_oracle'))
    if not oracle or oracle['status']!='PASS':local=False
    else:
     proof=compare_trajectories(Path(base['audit_directory'])/'base',Path(oracle['audit_directory'])/'base');local=proof['exact'] and oracle['output_sha256']==base['output_sha256']
     for record in (candidate,oracle):
      units=record['request']['runtime_metadata']['standard_backend']['ff_units'];local &= len(units)==38 and all(u['local_calls']==224 and u['local_failures']==0 and u['production_calls']==56 for u in units)
   if c=='standard_sparse_graph':exact &= backend['cache_steps']==base_backend['cache_steps']
   try:lane=p.numerical_lane(c,exact=exact,local_pass=local)
   except ValueError as e:failure.append({'case_tag':tag,'failure':str(e),'trajectory':comparison,'local_pass':local});continue
   checks.append({'case_tag':tag,'arithmetic_lane':lane,'trajectory':comparison,'png_exact':candidate['output_sha256']==base['output_sha256'],'cache_exact':backend['cache_steps']==base_backend['cache_steps'],'local_pass':local})
  if failure:admissions[c]={'status':'FAILED','failures':failure,'passed_cases':checks}
  else:
   eligible.append(c);admissions[c]={'status':'PASS','arithmetic_lane':'ff_arithmetic_change' if any(x['arithmetic_lane']=='ff_arithmetic_change' for x in checks) else 'exact','cases':checks,'semantic_pass':True,'capacity_pass':True,'execution_pass':True}
 if 'standard_fastops_eager' not in eligible:raise ValueError('no admitted baseline fastops fallback')
 return {'candidate_admissions':admissions,'eligible_candidates':eligible,'selection_scope':'quality independent fixed local-gate feasibility; timing selection follows'}
