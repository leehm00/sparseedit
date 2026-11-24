# SPDX-License-Identifier: Apache-2.0
"""Schema2 bounded, synchronous, full-content BF16 KV evidence; audit only."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import sys
import time
import torch

MAX_CHUNK_BYTES=8*1024**2
MAX_SCRATCH_BYTES=64*1024**2
MAX_AUDIT_BYTES=256*1024**2
DISK_RESERVE_BYTES=5*1024**3
SITES=('konly_w8_pre_bind','production_w8_post_bind_pre_scheduler','production_final_pre_scheduler','konly_final_post_loop')
ORDINARY_SITES=SITES[1:3]
FORMAT={'kind':'full_content_typed_bf16_kv_evidence','schema_version':2,'algorithm':'sha256',
 'dtype':'torch.bfloat16','logical_order':'row_major','byte_order':'little','signed_zero_policy':'negative_zero_to_positive_zero',
 'finite_policy':'reject_every_NaN_and_Inf_by_BF16_exponent_bits','sites':list(SITES),
 'max_chunk_bytes':MAX_CHUNK_BYTES,'max_scratch_bytes':MAX_SCRATCH_BYTES,'max_serialized_audit_bytes':MAX_AUDIT_BYTES,
 'scope':'full-content equality under SHA-256 collision assumption; finite checking is producer-attested; raw KV cannot be reconstructed'}

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
FORMAT_SHA=digest(FORMAT)
def require(ok,message):
    if not ok:raise ValueError(message)

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(MAX_CHUNK_BYTES),b''):h.update(block)
    return h.hexdigest()

def hex_digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

@dataclass(frozen=True)
class TensorProof:
    shape:tuple
    numel:int
    canonical_nbytes:int
    content_sha256:str
    elements_hashed:int
    bytes_hashed:int
    chunks:int
    max_chunk_bytes:int
    peak_scratch_bytes:int
    def to_dict(self):
        return dict(kind='full_content_tensor_evidence',schema_version=2,format_sha256=FORMAT_SHA,dtype='torch.bfloat16',shape=list(self.shape),
            logical_order='row_major',byte_order='little',finite=True,signed_zero_policy='negative_zero_to_positive_zero',algorithm='sha256',**{k:v for k,v in asdict(self).items() if k!='shape'})

class DigestBuilder:
    """Only complete, ordered logical coverage can finalize an immutable proof."""
    def __init__(self,shape,*,max_chunk_bytes=MAX_CHUNK_BYTES):
        require(isinstance(shape,tuple) and shape and all(type(n) is int and n>0 for n in shape),'unsupported tensor shape')
        require(type(max_chunk_bytes) is int and 2<=max_chunk_bytes<=MAX_CHUNK_BYTES,'chunk cap invalid')
        self.shape=shape;self.numel=math.prod(shape);self.cap=max_chunk_bytes;self.next=0;self.chunks=0;self.maximum=0;self.scratch=0;self.failed=False;self.finalized=False;self._hash=hashlib.sha256()
    def consume(self,tile,offset):
        require(not self.failed and not self.finalized,'failed/finalized coverage builder')
        try:
            require(type(offset) is int and offset==self.next,'logical coverage offset/reorder/repeat differs')
            require(isinstance(tile,torch.Tensor) and tile.dtype==torch.bfloat16 and tile.layout==torch.strided and tile.device.type in ('cpu','cuda'),'unsupported dtype/layout/device')
            count=tile.numel();size=count*2
            require(0<count and size<=self.cap and self.next+count<=self.numel,'logical coverage chunk size differs')
            # Only this already-bounded tile is packed/copied. No original cache
            # normalization, full-tensor flattening, float conversion or clone.
            cpu=tile.detach().to(device='cpu',copy=True,memory_format=torch.contiguous_format)
            require(cpu.untyped_storage().nbytes()<=self.cap,'CPU tile allocation exceeds chunk cap')
            words=cpu.view(torch.int16)
            exponent=words & 0x7f80
            nonfinite=exponent==0x7f80
            require(not bool(nonfinite.any()),'nonfinite BF16 KV input')
            explicit_peak=cpu.untyped_storage().nbytes()+exponent.untyped_storage().nbytes()+nonfinite.untyped_storage().nbytes()
            del exponent,nonfinite
            zero=words==-32768;words.masked_fill_(zero,0)
            explicit_peak=max(explicit_peak,cpu.untyped_storage().nbytes()+zero.untyped_storage().nbytes());del zero
            if sys.byteorder!='little':words.numpy().byteswap(inplace=True)
            raw=cpu.view(torch.uint8).numpy().tobytes(order='C')
            require(len(raw)==size,'logical byte coverage differs')
            explicit_peak=max(explicit_peak,cpu.untyped_storage().nbytes()+len(raw))
            # Two bounded transfer/packing tiles plus64KiB allow transient
            # implementation staging beyond explicitly measured CPU buffers.
            scratch=explicit_peak+2*size+65536
            require(scratch<=MAX_SCRATCH_BYTES,'hash scratch cap exceeded')
            self._hash.update(raw);self.next+=count;self.chunks+=1;self.maximum=max(self.maximum,size);self.scratch=max(self.scratch,scratch)
        except BaseException:
            self.failed=True;raise
    def finish(self):
        require(not self.failed and not self.finalized,'failed/finalized coverage builder')
        if self.next!=self.numel:self.failed=True;raise ValueError('incomplete logical coverage finalization')
        self.finalized=True
        return TensorProof(self.shape,self.numel,self.numel*2,self._hash.hexdigest(),self.next,self.next*2,self.chunks,self.maximum,self.scratch)


def logical_tiles(tensor,max_chunk_bytes):
    """Rectangular row-major tiles, including arbitrary supported strided views."""
    sizes=tuple(tensor.shape);limit=max_chunk_bytes//2
    axis=next(i for i in range(len(sizes)) if math.prod(sizes[i+1:])<=limit)
    trailing=math.prod(sizes[axis+1:]);width=max(1,limit//trailing);offset=0
    for prefix in itertools.product(*(range(n) for n in sizes[:axis])):
        for start in range(0,sizes[axis],width):
            item=prefix+(slice(start,min(sizes[axis],start+width)),)+(slice(None),)*(len(sizes)-axis-1)
            tile=tensor[item];yield offset,tile;offset+=tile.numel()


def hash_tensor(tensor,*,max_chunk_bytes=MAX_CHUNK_BYTES):
    require(isinstance(tensor,torch.Tensor) and tensor.dtype==torch.bfloat16 and tensor.layout==torch.strided and tensor.device.type in ('cpu','cuda'),'unsupported dtype/layout/device')
    builder=DigestBuilder(tuple(tensor.shape),max_chunk_bytes=max_chunk_bytes)
    for offset,tile in logical_tiles(tensor,max_chunk_bytes):builder.consume(tile,offset)
    return builder.finish()


def validate_tensor_descriptor(value,shape=None):
    fixed={'kind':'full_content_tensor_evidence','schema_version':2,'format_sha256':FORMAT_SHA,'dtype':'torch.bfloat16','logical_order':'row_major','byte_order':'little','finite':True,'signed_zero_policy':'negative_zero_to_positive_zero','algorithm':'sha256'}
    fields=set(fixed)|{'shape','numel','canonical_nbytes','content_sha256','elements_hashed','bytes_hashed','chunks','max_chunk_bytes','peak_scratch_bytes'}
    require(isinstance(value,dict) and set(value)==fields and all(value.get(k)==v and type(value.get(k)) is type(v) for k,v in fixed.items()),'tensor evidence schema/type/canonicalization differs')
    dims=value['shape'];require(isinstance(dims,list) and dims and all(type(n) is int and n>0 for n in dims),'tensor evidence shape differs')
    if shape is not None:require(dims==list(shape),'realized tensor evidence geometry differs')
    n=math.prod(dims)
    for key,want in (('numel',n),('canonical_nbytes',2*n),('elements_hashed',n),('bytes_hashed',2*n)):
        require(type(value[key]) is int and value[key]==want,'tensor evidence coverage differs: '+key)
    require(hex_digest(value['content_sha256']),'malformed full-content digest')
    require(type(value['chunks']) is int and 0<value['chunks']<=n and type(value['max_chunk_bytes']) is int and 2<=value['max_chunk_bytes']<=MAX_CHUNK_BYTES and value['max_chunk_bytes']%2==0 and type(value['peak_scratch_bytes']) is int and 0<value['peak_scratch_bytes']<=MAX_SCRATCH_BYTES,'tensor evidence chunk/scratch counters differ')
    require(value['max_chunk_bytes']<=2*n and value['chunks']*value['max_chunk_bytes']>=2*n,'tensor evidence chunk coverage differs')
    return value


def tensor_equal(left,right):
    a=left.to_dict() if isinstance(left,TensorProof) else left;b=right.to_dict() if isinstance(right,TensorProof) else right
    validate_tensor_descriptor(a);validate_tensor_descriptor(b)
    return all(a[k]==b[k] for k in ('dtype','shape','numel','canonical_nbytes','content_sha256','format_sha256'))

CONTEXT_FIELDS={'schema_version','phase','case_tag','arm','repetition','request_ordinal','warmup','preparation_sha256','effective_source_sha256','protocol_sha256','input_sha256','source_pixel_sha256','instruction_sha256','launch_ticket_sha256','format_sha256','helper_sha256','request_spec_sha256'}
GEOMETRY_FIELDS={'text_tokens','generated_tokens','reference_tokens','heads','head_dim','layer_counts'}

def validate_context(context):
    require(isinstance(context,dict) and set(context)==CONTEXT_FIELDS,'request context fields differ')
    require(type(context['schema_version']) is int and context['schema_version']==2 and type(context['warmup']) is bool,'request context schema/type differs')
    require(all(isinstance(context[k],str) and context[k] for k in ('phase','case_tag','arm')),'request identity differs')
    require(all(type(context[k]) is int and context[k]>=0 for k in ('repetition','request_ordinal')),'request ordinal differs')
    require(all(hex_digest(context[k]) for k in CONTEXT_FIELDS if k.endswith('_sha256')),'request provenance digest malformed')
    require(context['format_sha256']==FORMAT_SHA and context['helper_sha256']==sha_file(__file__),'producer/helper format identity differs')
    require(context['request_spec_sha256']==digest({k:v for k,v in context.items() if k!='request_spec_sha256'}),'request specification digest differs')

def validate_geometry(geometry):
    require(isinstance(geometry,dict) and set(geometry)==GEOMETRY_FIELDS,'geometry fields differ')
    require(all(type(geometry[k]) is int and geometry[k]>0 for k in GEOMETRY_FIELDS-{'layer_counts'}),'geometry counts differ')
    require(type(geometry['layer_counts']) is list and len(geometry['layer_counts'])==2 and all(type(v) is int and v>0 for v in geometry['layer_counts']),'layer geometry differs')
    return [1,sum(geometry[k] for k in ('text_tokens','generated_tokens','reference_tokens')),geometry['heads'],geometry['head_dim']]

def boundary_digest(node):return digest({k:v for k,v in node.items() if k!='root_sha256'})

def validate_boundary(node,site,context,geometry):
    require(isinstance(node,dict) and set(node)=={'site','step','context_sha256','geometry_sha256','format_sha256','helper_sha256','leaves','tensors','root_sha256'},'boundary schema differs')
    require(node['site']==site and type(node['step']) is int and node['step']==(7 if 'w8' in site else 27),'boundary site/step differs')
    require(node['context_sha256']==digest(context) and node['geometry_sha256']==digest(geometry) and node['format_sha256']==FORMAT_SHA and node['helper_sha256']==context['helper_sha256'],'boundary provenance differs')
    require(type(node['leaves']) is int and node['leaves']==4*sum(geometry['layer_counts']),'boundary leaf coverage differs')
    require(hex_digest(node['root_sha256']) and node['root_sha256']==boundary_digest(node),'boundary root digest differs')
    shape=validate_geometry(geometry);branches=node['tensors'];require(type(branches) is dict and set(branches)=={'positive','negative'},'boundary branch set differs')
    for branch,streams in branches.items():
        require(type(streams) is dict and set(streams)=={'dual','single'},'boundary stream set differs')
        for stream,count in zip(('dual','single'),geometry['layer_counts']):
            layers=streams[stream];require(type(layers) is dict and set(layers)=={str(i) for i in range(count)},'boundary layer set differs')
            for index,roles in layers.items():
                require(type(roles) is dict and set(roles)=={'key','value'},'boundary KV role set differs')
                for role,leaf in roles.items():
                    require(type(leaf) is dict and set(leaf)=={'subject','tensor'} and leaf['subject']=={'branch':branch,'stream':stream,'layer':int(index),'role':role},'boundary leaf subject differs')
                    require(type(leaf['subject']['layer']) is int,'boundary layer type differs')
                    validate_tensor_descriptor(leaf['tensor'],shape)

def validate_document(document,context,geometry,*,scored,allow_legacy=False):
    validate_context(context);validate_geometry(geometry)
    require(type(document) is dict and set(document)=={'kind','schema_version','format_sha256','context','geometry','scored','origin','boundaries'},'KV document fields differ')
    require(document['kind']=='full_content_kv_evidence' and type(document['schema_version']) is int and document['schema_version']==2 and document['format_sha256']==FORMAT_SHA,'KV document schema differs')
    require(document['context']==context and document['geometry']==geometry and type(document['scored']) is bool and document['scored']==scored,'KV document request/geometry differs')
    origin=document['origin']
    if origin!={'kind':'live_boundary_observation'}:
        require(allow_legacy and type(origin) is dict and set(origin)=={'kind','raw_sha256','raw_nbytes'} and origin['kind']=='legacy_saved_tensor_projection' and hex_digest(origin['raw_sha256']) and type(origin['raw_nbytes']) is int and origin['raw_nbytes']>0,'KV evidence origin differs')
    sites=SITES if scored else ORDINARY_SITES
    require(type(document['boundaries']) is dict and set(document['boundaries'])==set(sites),'KV boundary set differs')
    for site in sites:validate_boundary(document['boundaries'][site],site,context,geometry)
    return document

@dataclass(frozen=True)
class BoundaryProof:
    serialized:str
    def to_dict(self):return json.loads(self.serialized)

class KVEvidence:
    def __init__(self,context,geometry,*,scored,origin=None):
        validate_context(context);validate_geometry(geometry);require(type(scored) is bool,'scored type differs')
        self.context=canonical(context);self.geometry=canonical(geometry);self.scored=scored;self.origin=canonical(origin or {'kind':'live_boundary_observation'});self.sites=SITES if scored else ORDINARY_SITES;self.proofs=[];self.failed=False;self.finalized=False;self.hash_wall_seconds=0.;self.serialization_wall_seconds=0.
    def observe(self,site,caches):
        require(not self.failed and not self.finalized,'failed/finalized KV collector')
        started=time.perf_counter()
        try:
            require(len(self.proofs)<len(self.sites) and site==self.sites[len(self.proofs)],'KV boundary order/coverage differs')
            context=json.loads(self.context);geometry=json.loads(self.geometry);shape=validate_geometry(geometry)
            require(type(caches) is dict and set(caches)=={'positive','negative'},'live cache branches differ')
            devices=set()
            for cache in caches.values():
                require(cache.mode=='rotated','live cache mode differs')
                for attr,key in (('text_token_count','text_tokens'),('generated_token_count','generated_tokens'),('reference_token_count','reference_tokens'),('num_kv_heads','heads'),('head_dim','head_dim')):
                    require(getattr(cache,attr)==geometry[key],'live cache geometry differs: '+key)
                require(set(cache.layers)=={(stream,i) for stream,n in zip(('dual','single'),geometry['layer_counts']) for i in range(n)},'live cache layer coverage differs')
                for storage in cache.layers.values():
                    for tensor in (storage.key,storage.value):
                        require(isinstance(tensor,torch.Tensor) and list(tensor.shape)==shape,'live KV shape differs')
                        if tensor.device.type=='cuda':devices.add(tensor.device)
            # An all-stream device join completes every producer before the
            # synchronous bounded scan. CPU fixtures make no CUDA query.
            for device in devices:torch.cuda.synchronize(device)
            tensors={}
            for branch,cache in caches.items():
                tensors[branch]={}
                for stream,n in zip(('dual','single'),geometry['layer_counts']):
                    tensors[branch][stream]={}
                    for i in range(n):
                        storage=cache.layers[stream,i]
                        tensors[branch][stream][str(i)]={role:{'subject':{'branch':branch,'stream':stream,'layer':i,'role':role},'tensor':hash_tensor(getattr(storage,role)).to_dict()} for role in ('key','value')}
            node={'site':site,'step':7 if 'w8' in site else 27,'context_sha256':digest(context),'geometry_sha256':digest(geometry),'format_sha256':FORMAT_SHA,'helper_sha256':context['helper_sha256'],'leaves':4*sum(geometry['layer_counts']),'tensors':tensors}
            node['root_sha256']=boundary_digest(node);validate_boundary(node,site,context,geometry);self.proofs.append(BoundaryProof(canonical(node)))
            self.hash_wall_seconds+=time.perf_counter()-started
        except BaseException:self.failed=True;raise
    def finalize(self,path):
        require(not self.failed and not self.finalized and len(self.proofs)==len(self.sites),'incomplete/failed KV boundary coverage')
        document={'kind':'full_content_kv_evidence','schema_version':2,'format_sha256':FORMAT_SHA,'context':json.loads(self.context),'geometry':json.loads(self.geometry),'scored':self.scored,'origin':json.loads(self.origin),'boundaries':{site:proof.to_dict() for site,proof in zip(self.sites,self.proofs)}}
        validate_document(document,document['context'],document['geometry'],scored=self.scored,allow_legacy=True)
        started=time.perf_counter();save_json_capped(document,path);self.serialization_wall_seconds=time.perf_counter()-started;self.finalized=True
        return document

def compare_boundaries(left,left_site,right,right_site):
    for doc in (left,right):validate_document(doc,doc['context'],doc['geometry'],scored=doc['scored'])
    allowed=left_site==right_site or frozenset((left_site,right_site)) in (frozenset(SITES[:2]),frozenset(SITES[2:]))
    require(allowed,'unsupported boundary comparison pairing')
    common=('phase','case_tag','preparation_sha256','effective_source_sha256','protocol_sha256','input_sha256','source_pixel_sha256','instruction_sha256','format_sha256','helper_sha256')
    require(all(left['context'][k]==right['context'][k] for k in common) and left['geometry']==right['geometry'],'comparison source/case/geometry provenance differs')
    if left_site!=right_site:require(left['context']==right['context'],'cross-site comparison requires same request')
    a=left['boundaries'][left_site]['tensors'];b=right['boundaries'][right_site]['tensors']
    for branch in a:
        for stream in a[branch]:
            for layer in a[branch][stream]:
                for role in ('key','value'):require(tensor_equal(a[branch][stream][layer][role]['tensor'],b[branch][stream][layer][role]['tensor']),'full KV content differs: '+str((branch,stream,layer,role)))

class CappedWriter:
    def __init__(self,stream,cap):self.stream=stream;self.cap=cap
    def write(self,data):
        require(self.stream.tell()+len(data)<=self.cap,'serialized audit cap exceeded')
        return self.stream.write(data)
    def flush(self):return self.stream.flush()
    def tell(self):return self.stream.tell()

def remaining_cap(path,cap=MAX_AUDIT_BYTES):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    existing=sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file() and p!=path)
    require(not path.exists(),'refuse overwrite existing audit artifact')
    require(existing<cap,'serialized audit request cap exceeded')
    return cap-existing

def save_torch_capped(value,path,*,cap=MAX_AUDIT_BYTES):
    allowance=remaining_cap(path,cap)
    with Path(path).open('xb') as stream:torch.save(value,CappedWriter(stream,allowance))

def save_json_capped(value,path,*,cap=MAX_AUDIT_BYTES):
    allowance=remaining_cap(path,cap)
    with Path(path).open('xb') as stream:
        writer=CappedWriter(stream,allowance)
        for part in json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False).iterencode(value):writer.write(part.encode())

def require_disk_space(path,remaining_audits,*,free_bytes=None):
    require(type(remaining_audits) is int and remaining_audits>=0,'disk audit count differs')
    location=Path(path)
    while not location.exists():location=location.parent
    available=shutil.disk_usage(location).free if free_bytes is None else free_bytes
    require(type(available) is int and available>=0,'disk free byte witness differs')
    needed=remaining_audits*MAX_AUDIT_BYTES+DISK_RESERVE_BYTES
    require(available>=needed,'disk free space below remaining audit caps plus reserve')
    return {'remaining_audits':remaining_audits,'audit_cap_bytes':MAX_AUDIT_BYTES,'reserve_bytes':DISK_RESERVE_BYTES,'required_bytes':needed,'free_bytes':available}

def reference(document,path,site):
    require(Path(path).name=='kv_evidence.json' and site in document['boundaries'],'descriptor artifact/site differs')
    return {'kind':'kv_evidence_reference','schema_version':2,'artifact':'kv_evidence.json','artifact_sha256':sha_file(path),'site':site,'root_sha256':document['boundaries'][site]['root_sha256'],'context_sha256':digest(document['context']),'format_sha256':FORMAT_SHA}

def validate_reference(value,document,path,site):
    require(type(value) is dict and value==reference(document,path,site),'KV descriptor reference differs')
