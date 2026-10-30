"""Strict ID mapping + unchanged fixed official evaluator; CPU only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from verify import ROOT,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,default=ROOT/'reference_runs/condition_check_200/predictions.json')
    p.add_argument('--limit',type=int,choices=[200,2556],default=200)
    args=p.parse_args()
    if sha256(ROOT/'evaluation/qa_evaluate.py')!='a518bc37d94d2e84cc9012b41a758567e308dceea92aba14cda44a1364626ae7':raise ValueError('Fixed official evaluator was changed')
    if sha256(ROOT/'data/raw/MultiHopRAG.json')!='03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff':raise ValueError('Fixed question file differs')
    qa=json.loads((ROOT/'data/raw/MultiHopRAG.json').read_text())
    predictions=json.loads(args.predictions.read_text(encoding='utf-8-sig'))
    byid={}
    for row in predictions:
        if set(row)!={'question_id','model_answer'} or not isinstance(row['model_answer'],str) or row['question_id'] in byid:
            raise ValueError('Wrong prediction fields/types or duplicate IDs')
        byid[row['question_id']]=row['model_answer']
    expected={f'q{i:04d}' for i in range(args.limit)}
    if set(byid)!=expected:raise ValueError('Missing or unknown prediction IDs for selected scope')
    out=ROOT/'evaluation'/('pilot_200' if args.limit==200 else 'full_2556');out.mkdir(exist_ok=True)
    mapped=[{'query':row['query'],'question_type':row['question_type'],'model_answer':byid[f'q{i:04d}']}
            for i,row in enumerate(qa[:args.limit])]
    target=out/'qa_submission.json';target.write_text(json.dumps(mapped,ensure_ascii=False,indent=2))
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/qa_evaluate.py'),'--file',str(target),
        '--queries',str(ROOT/'data/raw/MultiHopRAG.json')],capture_output=True,text=True)
    (out/'eval_output.txt').write_text(result.stdout);(out/'eval_progress.txt').write_text(result.stderr)
    if result.returncode or 'Skipping query' in result.stdout:raise RuntimeError(result.stderr or result.stdout)
    print(f'Evaluation scope: {args.limit} questions. '+('Development subset, NOT a full-benchmark score.' if args.limit==200 else 'Full benchmark.'))
    print(result.stdout)


if __name__=='__main__':main()
