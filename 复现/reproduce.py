"""Fresh GPU reproduction of the 83% configuration, or CPU artifact validation."""
import argparse
from collections import Counter,defaultdict
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from verify import ROOT,verify_model,verify_package


def compare(root,output):
    sys.path.insert(0,str(root/'tools/condition_check_experiment'))
    from evaluate_answers import match
    refs=json.loads((root/'data/raw/MultiHopRAG.json').read_text())[:200]
    old=[]
    with (root/'outputs/minicpm_bm25_baseline/execution_records.jsonl').open() as f:
        old=[json.loads(next(f)) for _ in range(200)]
    new=[json.loads(s) for s in (output/'execution_records.jsonl').read_text().splitlines()]
    if len(new)!=200:raise ValueError('Fresh run is incomplete')
    pairs=Counter();types=defaultdict(Counter)
    for i,(a,b,ref) in enumerate(zip(old,new,refs)):
        if a['query']!=b['query'] or b['question_id']!=f'q{i:04d}' or b['query']!=ref['query']:
            raise ValueError('Fresh-run query mismatch')
        if a['context']!=b['context'] or a['context_chunks']!=b['context_chunks']:
            raise ValueError('Context changed during reproduction')
        ca=match(a['model_answer'],ref['answer']);cb=match(b['model_answer'],ref['answer'])
        pairs['baseline_correct']+=ca;pairs['fresh_correct']+=cb
        pairs['improvement']+=not ca and cb;pairs['regression']+=ca and not cb
        types[ref['question_type']]['total']+=1;types[ref['question_type']]['correct']+=cb
    result={'scope':'fresh first-200 development run','actual_context_identical_questions':200,
        'recorded_target_correct':166,'recorded_target_accuracy_percent':83.0,'counts':dict(pairs),
        'fresh_accuracy_percent':pairs['fresh_correct']/2,'by_type':dict(types),
        'target_matched':pairs['fresh_correct']==166}
    (output/'reproduction_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--model-dir',type=Path,help='Use existing original MiniCPM4.1-8B directory via a local symlink')
    p.add_argument('--full',action='store_true',help='Full 2556-question confirmation, a different evaluation scope from the recorded 83%')
    args=p.parse_args();root=ROOT
    expected=verify_package(root)
    if args.check_only:return
    target=root/'models/MiniCPM4.1-8B'
    if args.model_dir:
        original=args.model_dir.resolve()
        if not original.is_dir():raise FileNotFoundError(original)
        if not target.exists():target.symlink_to(original,target_is_directory=True)
        elif target.resolve()!=original:raise ValueError('Existing packaged model path points to a different model')
    if not target.is_dir():raise FileNotFoundError('Model weights are not in the small download package. Pass --model-dir /actual/model/path or restore the complete server export.')
    verify_model(target,expected,root)
    for name,version in expected['recorded_packages'].items():
        actual=importlib.metadata.version(name)
        if actual!=version:raise RuntimeError(f'Recorded {name}=={version}, installed {actual}. Restore recorded core versions before reproducing.')
    if args.full:
        subprocess.run([sys.executable,'-u',str(root/'run_condition_check_full.py'),'--project-root',str(root)],check=True)
        return
    output=root/'outputs/reproduced_condition_check_200'
    if (output/'execution_records.jsonl').exists():
        raise RuntimeError('Fresh-run output already exists. Move it aside before starting a new independent reproduction; archived results are untouched.')
    scripts=root/'tools/condition_check_experiment'
    for script in ['run_condition_check.py','evaluate_answers.py']:
        subprocess.run([sys.executable,'-u',str(scripts/script),'--project-root',str(root),
            '--limit','200','--output',str(output)],check=True)
        if script=='run_condition_check.py':
            state=json.loads((output/'summary.json').read_text())
            if state.get('processed')!=200 or state.get('failed')!=0:raise SystemExit('Fresh run incomplete/failed; inspect its journal.')
    compare(root,output)
    print('Fresh reproduction complete. Archived reference outputs were not used as new predictions.')


if __name__=='__main__':main()
