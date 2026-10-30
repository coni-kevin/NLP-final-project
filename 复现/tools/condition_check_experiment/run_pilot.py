"""Run the prompt-only first-200 pilot, score, verify controls, package results."""
from collections import Counter,defaultdict
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from baseline_core import atomic_json
from evaluate_answers import match


def read_first(path,n):
    rows=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            if len(rows)==n:break
            rows.append(json.loads(line))
    if len(rows)!=n:raise ValueError('Missing question records')
    return rows


def compare(root,out):
    base=root/'outputs/minicpm_bm25_baseline'
    cfg0=json.loads((base/'run_config.json').read_text());cfg1=json.loads((out/'run_config.json').read_text())
    allowed={'pipeline_version','script_sha256','context_source','prompt_policy_sha256'}
    changed={k for k in set(cfg0)|set(cfg1) if cfg0.get(k)!=cfg1.get(k)}
    if changed-allowed:raise ValueError(f'Unexpected non-prompt configuration changes: {changed-allowed}')
    old=read_first(base/'execution_records.jsonl',200);new=read_first(out/'execution_records.jsonl',200)
    refs=json.loads((root/'data/raw/MultiHopRAG.json').read_text(encoding='utf-8-sig'))[:200]
    paired=Counter();types=defaultdict(Counter);cases=[]
    for i,(a,b,r) in enumerate(zip(old,new,refs)):
        if a['question_id']!=b['question_id'] or b['question_id']!=f'q{i:04d}' or a['query']!=b['query'] or b['query']!=r['query']:
            raise ValueError('Question mismatch')
        if a['context']!=b['context'] or a['context_chunks']!=b['context_chunks']:
            raise ValueError('Actual context changed; this is no longer a prompt-only comparison')
        if a['status']!='ok' or b['status']!='ok':raise ValueError('A QA record failed')
        ca=match(a['model_answer'],r['answer']);cb=match(b['model_answer'],r['answer'])
        kind='both_correct' if ca and cb else 'regression' if ca else 'improvement' if cb else 'both_wrong'
        paired[kind]+=1
        t=types[r['question_type']];t['total']+=1;t['baseline_correct']+=ca;t['condition_check_correct']+=cb
        if ca!=cb:cases.append({'question_id':b['question_id'],'query':b['query'],'question_type':r['question_type'],
            'change':kind,'gold_answer':r['answer'],'baseline_answer':a['model_answer'],'new_answer':b['model_answer'],
            'baseline_raw':a['raw_output'],'new_raw':b['raw_output']})
    ca=paired['both_correct']+paired['regression'];cb=paired['both_correct']+paired['improvement']
    result={'scope':'Same first 200 development questions; local scoring, not an independent held-out test',
        'actual_context_identical_questions':200,'baseline_correct':ca,'condition_check_correct':cb,'total':200,
        'baseline_accuracy_percent':ca/2,'condition_check_accuracy_percent':cb/2,
        'delta_percentage_points':(cb-ca)/2,'paired_counts':dict(paired),'by_type':dict(types),
        'costs':{label:{k:round(sum(r[k] for r in records),3) for k in ['input_tokens','output_tokens','elapsed_seconds']}
                 for label,records in [('baseline',old),('condition_check',new)]}}
    atomic_json(out/'paired_comparison.json',result);atomic_json(out/'changed_cases.json',cases)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path('/root/autodl-tmp/multihop_rag'))
    args=p.parse_args();root=args.project_root.resolve();scripts=Path(__file__).resolve().parent
    out=root/'outputs/minicpm_condition_check_pilot'
    def run(name):subprocess.run([sys.executable,'-u',str(scripts/name),'--project-root',str(root),'--limit','200'],check=True)
    print('Only the instruction changes. Reusing original eight chunks on the same 200 queries.',flush=True)
    run('run_condition_check.py')
    state=json.loads((out/'summary.json').read_text())
    if state.get('processed')!=200 or state.get('failed')!=0 or not state.get('complete'):
        raise SystemExit('Generation incomplete/failed. Inspect logs; no final score/archive produced.')
    run('evaluate_answers.py');compare(root,out)
    target=root/'condition_check_pilot_results.zip';tmp=target.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            if f.is_file() and not f.name.startswith('.'):z.write(f,f.relative_to(root/'outputs'))
    tmp.replace(target)
    print('\nALL DONE. Result archive:',target,flush=True)
    print('No training, full-benchmark run, or shutdown was performed.',flush=True)


if __name__=='__main__':main()
