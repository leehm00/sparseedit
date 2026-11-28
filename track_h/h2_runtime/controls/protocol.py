"""Fixed source-paired standard backend protocol; no framework imports."""
import base64,hashlib,json,math,statistics
CANDIDATES=('standard_fastops_eager','standard_ff_compile','standard_sparse_graph')
ARMS=('direct_eager','standard','full')
STANDARD_DISABLED=('sparseedit_layer_fusion','sparseedit_stage_fusion','sparseedit_share_condition','sparseedit_share_positions','sparseedit_share_joint','sparseedit_request_position_reuse','sparseedit_condition_bank_shared','sparseedit_condition_plan_mode','sparseedit_condition_prepare_graph','sparseedit_prewarm_ratios')
COMPILE_OPTIONS={'backend':'inductor','mode':None,'dynamic':False,'fullgraph':False,'options':{'triton.cudagraphs':False}}
PROTOCOL={'kind':'standard_backend_reference','schema_version':1,'native':{'model':'FLUX.1-Kontext-dev','height':768,'width':768,'dtype':'bfloat16','steps':28,'warmup':8,'refresh':23,'seed':110,'true_cfg_scale':4.,'guidance_scale':1.,'negative_prompt':' ','cfg_execution':'serial'},'N':2304,'K':460,'candidates':list(CANDIDATES),'compile':COMPILE_OPTIONS,'formal_sources':48,'repetitions':2,'formal_measured':288,'candidate_timing_sources':3,'candidate_timing_repetitions':2,'tie_fraction':.01,'tie_order':list(CANDIDATES),'arithmetic_gate':{'bfloat16':[.016,1e-5],'float32':[1.3e-6,1e-5]},'timer':'synchronized native Omni.generate; all internal binding/staging/capture/lazy compile charged; model loading/input resize/PNG/excluded worker warmups/scoring separate'}
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
PROTOCOL_SHA256=digest(PROTOCOL)
def trace_digest(t):return digest({k:v for k,v in t.items() if k!='sha256'})
def validate_trace(t):
 if t.get('kind')!='flux_sparseedit_fixed_mask_trace' or t.get('schema_version')!=1 or t.get('sha256')!=trace_digest(t):raise ValueError('frozen trace schema/content digest differs')
 checks={'generated_token_count':2304,'reference_token_count':2304,'grid_height':48,'grid_width':48,'active_token_count':460,'num_inference_steps':28,'ratio':.2,'warmup_steps':8,'refresh_interval':15,'true_cfg':True,'true_cfg_scale':4.,'selector_mode':'forward','selector_protocol':'paper'}
 if any(t.get('config',{}).get(k)!=v for k,v in checks.items()):raise ValueError('frozen trace configuration differs')
 if len(t.get('steps',[]))!=28:raise ValueError('complete28 trace required')
 for i,row in enumerate(t['steps']):
  phase='warmup' if i<8 else 'refresh_full' if i==23 else 'sparse';action='store_full' if i in (7,23) else 'none' if i<8 else 'update_active'
  if row.get('step')!=i or row.get('phase')!=phase or row.get('kv_action')!=action:raise ValueError('trace action chronology differs')
  ix=row.get('indices')
  if phase=='sparse':
   if not isinstance(ix,list) or len(ix)!=460 or any(type(x) is not int or not 0<=x<2304 for x in ix) or ix!=sorted(set(ix)):raise ValueError('unique sorted exact-K indices required')
   if 'active_mask_b64' in row:
    packed=base64.b64decode(row['active_mask_b64'],validate=True)
    if len(packed)!=288:raise ValueError('packed mask size differs')
    decoded=[j for j in range(2304) if packed[j//8] & (1<<(7-j%8))]
    if decoded!=ix:raise ValueError('mask/index contents differ')
  elif ix is not None:raise ValueError('dense step carries sparse indices')
 return t

def options(candidate,trace,audit=False):
 if candidate not in CANDIDATES+('direct_eager','full'):raise ValueError('undeclared backend candidate')
 x={'flux_acceleration_method':'sparseedit','sparseedit_selector_protocol':'paper','sparseedit_paper_mode':'forward','sparseedit_ratio':.2,'sparseedit_warmup_steps':8,'sparseedit_refresh_interval':15,'sparseedit_avd_enabled':False,'sparseedit_online_reuse':'off','sparseedit_cfg_execution':'serial','sparseedit_replay_trace':str(trace),'sparseedit_collect_trace':True,'sparseedit_return_metadata':True,'sparseedit_defer_diagnostics':not audit,'sparseedit_collect_audit':audit,'sparseedit_runtime_variant':'baseline','sparseedit_direct_ops':'eager' if candidate=='direct_eager' else 'qkv','sparseedit_standard_backend':candidate if candidate in CANDIDATES else 'off','sparseedit_standard_audit':'off','sparseedit_standard_audit_dir':'','sparseedit_graph_fallback':False,'sparseedit_template_capacity':1}
 x.update({k:False for k in STANDARD_DISABLED});x.update(sparseedit_condition_plan_mode='off',sparseedit_prewarm_ratios=None)
 if candidate=='full':
  x.update(sparseedit_runtime_variant='graph_reuse',sparseedit_direct_ops='off',sparseedit_template_capacity=3,sparseedit_layer_fusion=True,sparseedit_stage_fusion=True,sparseedit_share_condition=True,sparseedit_share_positions=True,sparseedit_share_joint=True,sparseedit_request_position_reuse=True,sparseedit_condition_bank_shared=True,sparseedit_condition_plan_mode='reuse')
 return x

def numerical_lane(candidate,*,exact,local_pass):
 if candidate not in CANDIDATES:raise ValueError('unknown candidate')
 if exact:return 'exact'
 if candidate=='standard_ff_compile' and local_pass:return 'ff_arithmetic_change'
 raise ValueError('candidate numerical admission failed')
def complete_wall(request,pre_cost,overlap):
 if any(type(x) not in (float,int) or not math.isfinite(x) or x<0 for x in (request,pre_cost)) or overlap:raise ValueError('finite nonoverlapping wall costs required')
 return request+pre_cost

def select_candidate(rows,tags):
 if len(tags)!=3 or len(set(tags))!=3:raise ValueError('first3 source identities required')
 means={};lanes={};failures={}
 for c in CANDIDATES:
  rr=[r for r in rows if r.get('candidate')==c]
  # Failures are explicit candidates, not missing successful cells.
  if rr and any(r.get('status')=='FAILED' for r in rr):failures[c]=rr;continue
  if len(rr)!=6 or {(r.get('case_tag'),r.get('repetition')) for r in rr}!={(t,j) for t in tags for j in (0,1)}:raise ValueError('six complete candidate measurements required')
  for r in rr:
   if r.get('warmup') is not False or r.get('native_execution') is not True or r.get('status')!='PASS' or not all(r.get(k) is True for k in ('semantic_pass','numerical_pass','capacity_pass','execution_pass')):raise ValueError('ineligible candidate timing row')
   complete_wall(r['complete_request_seconds'],0.,False)
  means[c]=statistics.mean(r['complete_request_seconds'] for r in rr);lanes[c]=rr[0].get('arithmetic_lane','exact')
 if not means:raise ValueError('no supported standard path')
 minimum=min(means.values());tied=[c for c in CANDIDATES if c in means and means[c]<=1.01*minimum];chosen=tied[0]
 return {'selected':chosen,'arithmetic_lane':lanes[chosen],'means':means,'minimum':minimum,'tie_candidates':tied,'failures':failures,'quality_used':False,'rule':'mean six complete requests; <=1.01*Tmin; eager/FF/graph'}

def schedule(phase,tags,selected=None,eligible=None):
 size={'cal3':3,'calibration_timing':3,'formal288':48}
 if phase not in size or len(tags)!=size[phase] or len(set(tags))!=len(tags):raise ValueError('fixed source count/order required')
 if phase=='formal288' and selected not in CANDIDATES:raise ValueError('frozen selected path required')
 if eligible is not None and (not eligible or len(set(eligible))!=len(eligible) or any(c not in CANDIDATES for c in eligible)):raise ValueError('only declared admitted timing candidates allowed')
 rows=[];seen=set()
 def add(t,arm,rep,audit):
  if (arm,rep) not in seen:rows.append(dict(case_tag=tags[0],arm=arm,repetition=rep,warmup=True,audit=False));seen.add((arm,rep))
  rows.append(dict(case_tag=t,arm=arm,repetition=rep,warmup=False,audit=audit))
 if phase=='cal3':
  for t in tags:
   for arm in ('standard_fastops_eager','standard_ff_oracle','standard_ff_compile','standard_sparse_graph'):add(t,arm,0,True)
 else:
  arms=tuple(eligible or CANDIDATES) if phase=='calibration_timing' else ARMS
  for rep in (0,1):
   for i,t in enumerate(tags):
    order=arms[i%len(arms):]+arms[:i%len(arms)]
    if rep:order=order[::-1]
    for arm in order:add(t,arm,rep,False)
 return rows
