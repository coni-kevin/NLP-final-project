"""CPU-only package and archived-result verification; never calls a model."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile


ROOT=Path(__file__).resolve().parent


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify_package(root=ROOT):
    manifest=json.loads((root/'FILES_SHA256.json').read_text())
    for name,expected in manifest.items():
        path=root/name
        if not path.is_file() or sha256(path)!=expected:raise ValueError(f'Package file missing/changed: {name}')
    sys.path.insert(0,str(root/'tools/condition_check_experiment'))
    from baseline_core import build_chunks,load_inputs
    from fixed_contexts import FixedContexts
    from prompt_policy import INSTRUCTION
    from evaluate_answers import match
    corpus,queries,hashes=load_inputs(root)
    expected=json.loads((root/'configs/expected_run.json').read_text())
    if hashes!=expected['data_hashes']:raise ValueError('Original data hashes differ')
    loader=FixedContexts(root,build_chunks(corpus),queries[:200],hashes)
    records=root/'reference_runs/condition_check_200/execution_records.jsonl'
    predictions=json.loads((root/'reference_runs/condition_check_200/predictions.json').read_text())
    references=json.loads((root/'data/raw/MultiHopRAG.json').read_text())
    if len(predictions)!=200:raise ValueError('Archived prediction count differs')
    count=0;correct=0
    with records.open(encoding='utf-8') as f:
        for i,line in enumerate(f):
            r=json.loads(line)
            if i>=200 or r['question_id']!=f'q{i:04d}' or r['query']!=queries[i]['query'] or r['status']!='ok':
                raise ValueError('Archived query/status differs')
            if r['context']!=loader.rows[r['query']]['context'] or r['context_chunks']!=loader.retrieve(r['query']):
                raise ValueError('Archived original BM25 context differs')
            if r['messages']!=[{'role':'user','content':INSTRUCTION.format(context=r['context'],query=r['query'])}]:
                raise ValueError('Archived prompt differs from frozen source')
            if predictions[i]!={'question_id':r['question_id'],'model_answer':r['model_answer']}:
                raise ValueError('Prediction and execution record differ')
            correct+=match(r['model_answer'],references[i]['answer']);count+=1
    if (correct,count)!=(166,200):raise ValueError(f'Archived score differs: {correct}/{count}')
    print('VERIFIED: all package hashes, original data/code/context, and archived 166/200 = 83.00%.')
    print('Scope: first 200 development questions. This command did not run a model.')
    return expected


def verify_model(model_path,expected,root=ROOT):
    if sha256(model_path/'config.json')!=expected['model_config_sha256']:
        raise ValueError('MiniCPM model configuration differs from the actual experiment')
    for name,digest in expected['model_code_hashes'].items():
        if sha256(model_path/name)!=digest:raise ValueError(f'Model custom code differs: {name}')
    index=model_path/'model.safetensors.index.json'
    if index.exists():
        required=set(json.loads(index.read_text())['weight_map'].values())
    else:
        required={p.name for p in model_path.glob('*.safetensors')}
        if not required and (model_path/'pytorch_model.bin').exists():required={'pytorch_model.bin'}
    if not required or any(not (model_path/name).is_file() for name in required):
        raise ValueError('Complete original model weight files are missing')
    actual=root/'configs/actual_model_files_sha256.json'
    if actual.exists():
        model_manifest=json.loads(actual.read_text())
        for name,digest in model_manifest.items():
            if not (model_path/name).is_file() or sha256(model_path/name)!=digest:
                raise ValueError(f'Exported original model file hash differs: {name}')
        print('Verified exact model-file hashes captured by the complete server export.')
    else:
        print('Model configuration/code and shard presence verified. Exact original weight hashes require export_complete.py on the original server.')
    return sorted(required)


if __name__=='__main__':verify_package()
