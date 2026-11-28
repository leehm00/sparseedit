"""Standard reference adapters: full raw-KV copying, ordinary torch graphs, FF compile."""
from __future__ import annotations
import copy,hashlib,json,math,time,types
from pathlib import Path
import torch

MAX_AUDIT_BYTES=256*1024**2
COMPILE_KW={'dynamic':False,'fullgraph':False,'options':{'triton.cudagraphs':False}}

def tensor_bytes(x):return x.numel()*x.element_size()
def pair_bytes(pair):return sum(tensor_bytes(x) for x in pair)
def validate_disjoint_caches(caches):
 seen=set()
 for branch,c in caches.items():
  for pair in c.layers.values():
   for x in pair:
    ptr=x.untyped_storage().data_ptr()
    if ptr in seen:raise ValueError('CFG/cache storage aliases')
    seen.add(ptr)

def cache_branch(caches, cache):
 for branch, candidate in caches.items():
  if candidate is cache:return branch
 return None

def register_cache_branch(caches,cache):
 branch=cache_branch(caches,cache)
 if branch is not None:return branch
 if len(caches)>=2:raise ValueError('unexpected third CFG cache')
 branch='positive' if not caches else 'negative';caches[branch]=cache;return branch

def local_accuracy(expected,actual):
 out={'pass':False,'shape':list(expected.shape),'dtype':str(expected.dtype),'device':str(expected.device),'layout':str(expected.layout),'stride':list(expected.stride()),'elements':expected.numel(),'unequal':0,'nonfinite':0,'max_abs':0.,'relative_l2':0.,'max_tolerance_ratio':0.,'tolerance_exceedances':0}
 if expected.shape!=actual.shape or expected.dtype!=actual.dtype or expected.device!=actual.device or expected.layout!=actual.layout or expected.layout!=torch.strided or expected.dtype not in (torch.bfloat16,torch.float32):out['reason']='shape/dtype/device/layout';return out
 rtol,atol=(.016,1e-5) if expected.dtype==torch.bfloat16 else (1.3e-6,1e-5);out.update(rtol=rtol,atol=atol,actual_stride=list(actual.stride()));ss=rr=0.;passed=True
 e,a=expected.reshape(-1),actual.reshape(-1)
 for start in range(0,e.numel(),262144):
  x,y=e[start:start+262144],a[start:start+262144];finite=torch.isfinite(x)&torch.isfinite(y);bad=int((~finite).sum().item());out['nonfinite']+=bad
  if bad:passed=False;continue
  try:torch.testing.assert_close(y,x,rtol=rtol,atol=atol,equal_nan=False,check_device=True,check_dtype=True,check_layout=True)
  except AssertionError:passed=False
  delta=(y.float()-x.float()).abs();out['unequal']+=int((x!=y).sum().item());out['tolerance_exceedances']+=int((~torch.isclose(y,x,rtol=rtol,atol=atol,equal_nan=False)).sum().item())
  if delta.numel():out['max_abs']=max(out['max_abs'],float(delta.max().item()));out['max_tolerance_ratio']=max(out['max_tolerance_ratio'],float((delta/(atol+rtol*x.float().abs())).max().item()))
  ss+=float((delta.double().square()).sum().item());rr+=float((x.double().square()).sum().item())
 out['relative_l2']=math.sqrt(ss)/max(math.sqrt(rr),1e-30);out['pass']=passed;return out

def exact_pair(expected,actual):
 if len(expected)!=len(actual) or any(x.shape!=y.shape or x.dtype!=y.dtype or not torch.equal(x,y) for x,y in zip(expected,actual)):raise ValueError('candidate-produced cache transition differs from clone/scatter oracle')

def require_compile_coverage(value):
 if not value.get('regions') or not any(r.get('operators') for r in value['regions']) or value.get('production_calls',0)<1 or value.get('lazy_failures',0):raise ValueError('actual nonempty compiled production execution required')

class GraphCacheOwner:
 """Two full sets: original cache becomes read staging; captured clone outputs are W."""
 def __init__(self,cache):
  self.cache=cache;self.reads={};self.outputs={};self.fresh={};self.staging_bytes=0;self.refresh_bytes=0;self.original_store=cache.store_full;self.original_update=cache.update_active;self.attached=False;self.record_fresh=True
 def attach(self):
  owner=self
  def store(c,s,l,k,v):
   owner.original_store(s,l,k,v)
   if (s,l) in owner.reads:
    transient=c.layers[s,l]
    for dst,src in zip(owner.reads[s,l],transient):dst.copy_(src)
    owner.refresh_bytes+=pair_bytes(transient);c.layers[s,l]=owner.reads[s,l]
  def update(c,s,l,k,v,indices):
   if owner.record_fresh:owner.fresh[s,l]=(k,v,indices)
   return owner.original_update(s,l,k,v,indices)
  self.cache.store_full=types.MethodType(store,self.cache);self.cache.update_active=types.MethodType(update,self.cache);self.attached=True
 def seed_reads(self):
  if self.reads:raise ValueError('read staging already seeded')
  self.reads=dict(self.cache.layers)
  if not self.reads:raise ValueError('full cache required before graph capture')
 def capture_outputs(self):
  self.outputs=dict(self.cache.layers)
  if self.outputs.keys()!=self.reads.keys():raise ValueError('graph cache coverage differs')
  if any(x.data_ptr()==y.data_ptr() for key in self.reads for x,y in zip(self.reads[key],self.outputs[key])):raise ValueError('clone outputs alias read staging')
 def restore_trials(self):self.cache.layers=dict(self.reads)
 def bind_previous(self):
  for key,pair in self.cache.layers.items():
   for dst,src in zip(self.reads[key],pair):
    if dst.data_ptr()!=src.data_ptr():dst.copy_(src);self.staging_bytes+=tensor_bytes(src)
 def publish(self):self.cache.layers=dict(self.outputs)
 def audit_replay(self):
  if self.fresh.keys()!=self.outputs.keys():raise ValueError('all dual/single layers required')
  for key,(k,v,ix) in self.fresh.items():
   # One layer's oracle transient only; no third complete cache set.
   want=tuple(old.clone().scatter_(1,ix[:,:,None,None].expand_as(new),new) for old,new in zip(self.reads[key],(k,v)))
   exact_pair(want,self.outputs[key])
 def inventory(self):return {'full_sets':2 if self.reads and self.outputs else 1,'read_bytes':sum(pair_bytes(p) for p in self.reads.values()),'output_bytes':sum(pair_bytes(p) for p in self.outputs.values()),'full_staging_copy_bytes':self.staging_bytes,'refresh_staging_copy_bytes':self.refresh_bytes,'refresh_transient_scope':'one baseline cloned layer pair; released after full copy into graph read staging'}
 def close(self):
  if self.attached:self.cache.store_full=self.original_store;self.cache.update_active=self.original_update
  self.reads.clear();self.outputs.clear();self.fresh.clear();self.attached=False

class FFCompileUnit:
 def __init__(self,module,original,name,*,compile_factory=None,emit=None,context=None,mode='candidate'):
  self.module=module;self.original=original;self.name=name;self.emit=emit;self.context=context or (lambda:{});self.mode=mode;self.regions=[];self.production_calls=0;self.lazy_failures=0;self.local_calls=0;self.local_failures=0;self.compiler_seconds=0.
  factory=compile_factory or torch.compile
  def backend(gm,inputs,**kw):
   started=time.perf_counter();result=torch._inductor.compile(gm,inputs,options=kw.get('options',{'triton.cudagraphs':False}));self.compiler_seconds+=time.perf_counter()-started
   self.regions.append({'operators':[str(n.target) for n in gm.graph.nodes if n.op in ('call_function','call_method','call_module')],'nodes':len(list(gm.graph.nodes))})
   def counted(*a,**k):
    self.compiled_executions=getattr(self,'compiled_executions',0)+1
    return result(*a,**k)
   return counted
  self.production=factory(original,backend=backend,**COMPILE_KW)
  self.diagnostic=None;self.factory=factory;self.backend=backend
 def _diagnostics(self,x):
  if self.diagnostic is None:
   first=self.module.net[0].proj.forward;second=self.module.net[2].forward
   gelu=lambda z:torch.nn.functional.gelu(z,approximate='tanh')
   self.diagnostic=[(label,fn,self.factory(fn,backend='inductor',**COMPILE_KW)) for label,fn in [('linear1',first),('gelu_tanh',gelu),('linear2',second)]]
  current=x
  for label,eager,compiled in self.diagnostic:
   expected=eager(current);actual=compiled(current);self._check(label,expected,actual);current=expected
 def _check(self,label,expected,actual):
  value=local_accuracy(expected,actual);value.update(self.context(),ff=self.name,boundary=label);self.local_calls+=1;self.local_failures+=not value['pass']
  if self.emit:self.emit(value)
  if not value['pass']:raise ValueError('FF local public Torch numerical gate failed: '+self.name+'/'+label)
 def __call__(self,x):
  if self.emit and not bool(torch.isfinite(x).all()):raise ValueError('nonfinite actual FF input')
  try:actual=self.production(x)
  except BaseException:self.lazy_failures+=1;raise
  self.production_calls+=1
  if self.emit:
   self._check('production_ff',self.original(x),actual);self._diagnostics(x)
  return self.original(x) if self.mode=='oracle' else actual
 def receipt(self):return {'ff':self.name,'regions':self.regions,'production_calls':self.production_calls,'local_calls':self.local_calls,'local_failures':self.local_failures,'lazy_failures':self.lazy_failures,'backend_compile_seconds':self.compiler_seconds}

def scheduler_transition(scheduler,original,velocity,timestep,latents,audit,**kwargs):
 if not audit:return original(velocity,timestep,latents,**kwargs),None
 # The installed step hook owns the live pipeline, including CUDA weights.
 # Copy scheduler state independently and invoke the original method on it.
 reference = copy.copy(scheduler)
 reference.__dict__ = copy.deepcopy({key: value for key, value in vars(scheduler).items() if key != "step"})
 expected = original.__func__(reference, velocity, timestep, latents, **kwargs)
 result = original(velocity, timestep, latents, **kwargs)
 exact_pair(expected,result)
 before=getattr(reference,'step_index',None);after=getattr(scheduler,'step_index',None)
 if before!=after:raise ValueError('scheduler advancement differs')
 return result,{'exact':True,'step_index':after,'velocity_input':'actual candidate complete velocity'}

def clone_inputs(value):
 if isinstance(value,torch.Tensor):return value.detach().clone()
 if isinstance(value,dict):return {k:clone_inputs(v) for k,v in value.items()}
 if isinstance(value,(tuple,list)):return type(value)(clone_inputs(v) for v in value)
 if hasattr(value,'kv_cache'):
  result=copy.copy(value)
  for k,v in vars(value).items():
   if isinstance(v,torch.Tensor):setattr(result,k,clone_inputs(v))
  return result
 return value

def bind_inputs(static,live):
 if isinstance(static,torch.Tensor):
  if not isinstance(live,torch.Tensor) or static.shape!=live.shape or static.dtype!=live.dtype or static.device!=live.device or static.stride()!=live.stride():raise ValueError('unsupported graph tensor shape/dtype/device/stride binding')
  static.copy_(live)
 elif isinstance(static,dict):
  if static.keys()!=live.keys():raise ValueError('graph input schema changed')
  for k in static:
   if k!='_flux_acceleration_step':bind_inputs(static[k],live[k])
 elif hasattr(static,'kv_cache'):
  if static.kv_cache is not live.kv_cache or static.kv_action!=live.kv_action:raise ValueError('graph cache/action owner changed')
  for k in ('image_indices','full_img_ids'):
   if hasattr(static,k):bind_inputs(getattr(static,k),getattr(live,k))
 elif isinstance(static,(list,tuple)):
  if len(static)!=len(live):raise ValueError('graph sequence shape changed')
  for x,y in zip(static,live):bind_inputs(x,y)
 elif static!=live:raise ValueError('graph scalar configuration changed')

class CacheTransitionAudit:
 def __init__(self,cache):self.cache=cache;self.store=cache.store_full;self.update=cache.update_active;self.checks=0
 def attach(self):
  audit=self
  def store(c,s,l,k,v):
   audit.store(s,l,k,v);exact_pair((k,v),c.layers[s,l]);audit.checks+=1
  def update(c,s,l,k,v,ix):
   old=c.layers[s,l];result=audit.update(s,l,k,v,ix);expected=tuple(x.clone().scatter_(1,ix[:,:,None,None].expand_as(y),y) for x,y in zip(old,(k,v)));exact_pair(expected,result);audit.checks+=1;return result
  self.cache.store_full=types.MethodType(store,self.cache);self.cache.update_active=types.MethodType(update,self.cache)
 def close(self):self.cache.store_full=self.store;self.cache.update_active=self.update

class StandardBranchGraph:
 def __init__(self,branch,cache,predict,*,audit=False):
  self.branch=branch;self.owner=GraphCacheOwner(cache);self.owner.attach();self.owner.record_fresh=audit;self.predict=predict;self.audit=audit;self.static=None;self.graph=None;self.output=None;self.owned=None;self.replays=0;self.capture_seconds=0.;self.warmup_seconds=0.;self.pool_bytes=None;self.capture_reserved_delta=0;self.trial_reads_restored=False
 def run(self,kwargs):
  if self.graph is None:self.capture(kwargs)
  else:bind_inputs(self.static,kwargs)
  self.owner.bind_previous();self.graph.replay();self.owner.publish();self.owned.copy_(self.output);self.replays+=1
  if self.audit:self.owner.audit_replay()
  return self.owned
 def capture(self,kwargs):
  device=kwargs['hidden_states'].device
  if device.type!='cuda':raise ValueError('native standard CUDA graph requires CUDA device')
  self.owner.seed_reads();self.static=clone_inputs(kwargs);torch.cuda.synchronize(device);stream=torch.cuda.Stream(device=device);stream.wait_stream(torch.cuda.current_stream(device));started=time.perf_counter()
  with torch.cuda.stream(stream):
   for _ in range(2):self.owner.restore_trials();self.predict(**self.static)
  torch.cuda.current_stream(device).wait_stream(stream);torch.cuda.synchronize(device);self.warmup_seconds=time.perf_counter()-started;self.owner.restore_trials()
  graph=torch.cuda.CUDAGraph();pool=torch.cuda.graph_pool_handle();reserved=torch.cuda.memory_reserved(device);started=time.perf_counter()
  with torch.cuda.graph(graph,pool=pool):self.output=self.predict(**self.static)
  torch.cuda.synchronize(device);self.capture_seconds=time.perf_counter()-started;self.capture_reserved_delta=torch.cuda.memory_reserved(device)-reserved;self.owner.capture_outputs();self.owner.restore_trials();self.trial_reads_restored=True;self.graph=graph;self.owned=torch.empty_like(self.output)
  try:self.pool_bytes=sum(int(s['total_size']) for s in torch.cuda.memory_snapshot(mempool_id=pool,include_traces=False))
  except (TypeError,RuntimeError):self.pool_bytes=None
 def receipt(self):return {'branch':self.branch,'actual_cuda_graph':self.graph is not None,'replays':self.replays,'capture_trials':1 if self.graph is not None else 0,'warmup_trials':2 if self.graph is not None else 0,'capture_seconds':self.capture_seconds,'warmup_seconds':self.warmup_seconds,'pool_bytes':self.pool_bytes,'pool_bytes_source':'allocator_pool_snapshot' if self.pool_bytes is not None else 'unavailable','capture_reserved_delta_bytes':self.capture_reserved_delta,'owned_output_bytes':tensor_bytes(self.owned) if self.owned is not None else 0,'trial_reads_restored':self.trial_reads_restored,'cache':self.owner.inventory()}
 def close(self):self.graph=None;self.output=None;self.owned=None;self.static=None;self.owner.close()

class BackendSession:
 """Installed inside one synchronized request. Default off paths bypass this class."""
 def __init__(self,pipeline,config):
  self.pipeline=pipeline;self.config=config;self.candidate=config.sparseedit_standard_backend;self.audit_mode=config.sparseedit_standard_audit;self.audit=self.audit_mode!='off';self.directory=Path(config.sparseedit_standard_audit_dir) if self.audit else None
  self.graphs={};self.caches={};self.monitors=[];self.units=[];self.original_ff=[];self.calls=[];self.scheduler_checks=[];self.cache_steps=[];self.current={};self.memory=[];self.closed=False;self.status='RUNNING';self.error=None;self.local_records=0;self.local_bytes=0;self.ff_before={};self.restore=[]
  if self.candidate not in ('standard_fastops_eager','standard_ff_compile','standard_sparse_graph'):raise ValueError('unsupported standard candidate')
  disabled=('sparseedit_layer_fusion','sparseedit_stage_fusion','sparseedit_share_condition','sparseedit_share_positions','sparseedit_share_joint','sparseedit_request_position_reuse','sparseedit_condition_bank_shared','sparseedit_condition_prepare_graph')
  if config.sparseedit_runtime_variant!='baseline' or config.sparseedit_direct_ops!='qkv' or any(getattr(config,k) for k in disabled) or config.sparseedit_condition_plan_mode!='off' or config.sparseedit_online_reuse!='off' or config.sparseedit_avd_enabled:raise ValueError('standard reference inherited optimized runtime flags')
 def memory_snapshot(self,site):
  device=next(self.pipeline.transformer.parameters()).device
  if device.type=='cuda':self.memory.append({'site':site,'allocated':int(torch.cuda.memory_allocated(device)),'reserved':int(torch.cuda.memory_reserved(device)),'peak_allocated':int(torch.cuda.max_memory_allocated(device)),'peak_reserved':int(torch.cuda.max_memory_reserved(device))})
 def emit(self,row):
  data=json.dumps(row,sort_keys=True,allow_nan=False)+'\n';self.local_bytes+=len(data.encode());self.local_records+=1
  if self.local_bytes>MAX_AUDIT_BYTES:raise ValueError('compact FF audit disk cap exceeded')
  with (self.directory/'ff_local.jsonl').open('a') as f:f.write(data)
 def __enter__(self):
  if self.audit:self.directory.mkdir(parents=True,exist_ok=False)
  self.memory_snapshot('request_setup')
  self.predict=self.pipeline.predict_noise;self.scheduler=self.pipeline.scheduler.step
  self.pipeline.predict_noise=self.predict_noise;self.pipeline.scheduler.step=self.scheduler_step
  if self.candidate=='standard_ff_compile':
   bank=getattr(self.pipeline.transformer,'_standard_ff_compiled_units',None)
   if bank is None:bank={};self.pipeline.transformer._standard_ff_compiled_units=bank
   for i,block in enumerate(self.pipeline.transformer.transformer_blocks):
    for attr in ('ff','ff_context'):
     module=getattr(block,attr);name=f'dual.{i}.{attr}';original=module.forward
     if name not in bank:bank[name]=FFCompileUnit(module,original,name)
     unit=bank[name];unit.emit=self.emit if self.audit else None;unit.context=lambda:self.current;unit.mode='oracle' if self.audit_mode=='oracle' else 'candidate';self.ff_before[name]=(unit.production_calls,getattr(unit,'compiled_executions',0),unit.local_calls,unit.local_failures,unit.lazy_failures);self.units.append(unit);self.original_ff.append((module,original));module.forward=unit
   if len(self.units)!=2*len(self.pipeline.transformer.transformer_blocks):raise ValueError('dual FF coverage differs')
  return self
 def predict_noise(self,**kwargs):
  context=kwargs['joint_attention_kwargs']['_flux_acceleration_context'];step=kwargs['joint_attention_kwargs']['_flux_acceleration_step'];cache=context.kv_cache
  branch=cache_branch(self.caches,cache)
  if branch is None:
   branch=register_cache_branch(self.caches,cache)
   if self.audit and self.candidate!='standard_sparse_graph':monitor=CacheTransitionAudit(cache);monitor.attach();self.monitors.append(monitor)
  expected_step=len(self.calls)//2;expected_branch='positive' if len(self.calls)%2==0 else 'negative'
  if step!=expected_step or branch!=expected_branch:raise ValueError('actual serial CFG chronology differs')
  self.current={'step':step,'branch':branch,'input_trajectory':'norm_oracle' if self.audit_mode=='oracle' else 'candidate'}
  if self.candidate=='standard_sparse_graph' and context.kv_action=='update_active':
   if branch not in self.graphs:self.graphs[branch]=StandardBranchGraph(branch,cache,self.predict,audit=self.audit)
   self.memory_snapshot('before_sparse_graph' if step==8 else 'before_steady_replay' if step==9 else 'before_replay');result=self.graphs[branch].run(kwargs)
  else:result=self.predict(**kwargs)
  self.calls.append({'step':step,'branch':branch,'kv_action':context.kv_action,'input_shape':list(kwargs['hidden_states'].shape),'output_shape':list(result.shape),'graph_replayed':self.candidate=='standard_sparse_graph' and context.kv_action=='update_active'})
  return result
 def scheduler_step(self,velocity,timestep,latents,**kwargs):
  step=len(self.scheduler_checks)
  if self.audit:
   from .kv_evidence import hash_tensor
   validate_disjoint_caches(self.caches)
   roots={branch:{str(key):[hash_tensor(x).content_sha256 for x in pair] for key,pair in c.layers.items()} for branch,c in self.caches.items()}
   self.cache_steps.append({'step':step,'branches':roots})
  result,check=scheduler_transition(self.pipeline.scheduler,self.scheduler,velocity,timestep,latents,self.audit,**kwargs);self.scheduler_checks.append(check or {'step':step,'audit':False});return result
 def receipt(self):
  units=[]
  for u in self.units:
   value=u.receipt();before=self.ff_before[u.name];value.update(production_calls=u.production_calls-before[0],compiled_executions=getattr(u,'compiled_executions',0)-before[1],local_calls=u.local_calls-before[2],local_failures=u.local_failures-before[3],lazy_failures=u.lazy_failures-before[4]);units.append(value)
  return {'status':self.status,'candidate':self.candidate,'audit_mode':self.audit_mode,'error':self.error,'calls':self.calls,'scheduler_checks':self.scheduler_checks,'cache_steps':self.cache_steps,'cache_transition_checks':sum(m.checks for m in self.monitors),'graphs':{k:g.receipt() for k,g in self.graphs.items()},'ff_units':units,'ff_local_records':self.local_records,'ff_local_bytes':self.local_bytes,'memory':self.memory,'capture_scope':'request-scoped sparse CFG forward; full raw-KV staging + full clone/scatter; dense eager','arithmetic_group':'independent_QKV_fastnorm_rope','fallback':False}
 def __exit__(self,kind,error,tb):
  self.status='FAILED' if error else 'PASS';self.error=f'{type(error).__name__}: {error}' if error else None;self.memory_snapshot('before_close');value=self.receipt()
  self.pipeline.predict_noise=self.predict;self.pipeline.scheduler.step=self.scheduler
  for module,original in self.original_ff:module.forward=original
  for g in self.graphs.values():g.close()
  for m in self.monitors:m.close()
  self.memory_snapshot('after_close');value['memory']=self.memory;self.final=value
  if self.audit:
   path=self.directory/'backend_receipt.json';path.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
   total=sum(p.stat().st_size for p in self.directory.parent.rglob('*') if p.is_file())
   if total>MAX_AUDIT_BYTES:raise ValueError('per-request compact audit cap exceeded')
  return False
