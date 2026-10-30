#!/usr/bin/env python3
"""Separate offline scoring and official-evaluator input adapter.

The local score reproduces the assignment's token-overlap rule. It is not a run
of the instructor's fixed official script. --official-script runs a supplied
script unchanged against a separate, query-linked evaluation input.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def match(prediction, reference):
    extracted = re.search(r'The answer to the question is "(.*?)"', prediction)
    if extracted:
        prediction = extracted.group(1)
    predicted_tokens = set(prediction.lower().split())
    return bool(predicted_tokens & set(reference.lower().split()))


def evaluate(rows, predictions, limit=0):
    selected = rows[:limit] if limit else rows
    expected = {f'q{i:04d}' for i in range(len(selected))}
    by_id = {}
    all_ids = {f'q{i:04d}' for i in range(len(rows))}
    for p in predictions:
        qid, answer = p['question_id'], p['model_answer']
        if qid not in all_ids or qid in by_id or not isinstance(answer, str):
            raise ValueError('Unknown/duplicate ID or non-string answer')
        by_id[qid] = answer
    if set(by_id) - expected:
        raise ValueError('Prediction IDs exceed selected scoring subset')
    counts = defaultdict(lambda: {'correct': 0, 'total': 0})
    adapted = []
    for i, row in enumerate(selected):
        answer = by_id.get(f'q{i:04d}', '')
        correct = int(match(answer, row['answer']))
        for k in ('overall', row['question_type']):
            counts[k]['correct'] += correct
            counts[k]['total'] += 1
        # References are used ONLY in this post-prediction evaluator input.
        adapted.append({'query': row['query'], 'question_type': row['question_type'],
                        'gold_answer': row['answer'], 'model_answer': answer})
    for value in counts.values():
        value['accuracy'] = value['correct'] / value['total'] if value['total'] else 0
        value['accuracy_percent'] = value['accuracy'] * 100
    return {'metric_source': 'local reproduction of assignment token-overlap rule; not official-script output',
            'full_benchmark': len(selected) == 2556,
            'missing_predictions': len(expected - set(by_id)), 'results': dict(counts)}, adapted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=Path('/root/autodl-tmp/multihop_rag'))
    p.add_argument('--output', type=Path)
    p.add_argument('--limit', type=int, default=200, help='0: all questions (missing predictions count as incorrect); N: pilot subset')
    p.add_argument('--official-script', type=Path, help='Instructor-supplied, fixed qa_evaluate.py; do not substitute upstream main')
    args = p.parse_args()
    if args.limit < 0:
        p.error('limit must be nonnegative')
    root = args.project_root.resolve()
    out = (args.output or root / 'outputs/minicpm_condition_check_pilot').resolve()
    rows = json.loads((root / 'data/raw/MultiHopRAG.json').read_text(encoding='utf-8-sig'))
    if len(rows) != 2556:
        raise ValueError('Expected all 2556 official questions')
    predictions = json.loads((out / 'predictions.json').read_text(encoding='utf-8'))
    scores, adapted = evaluate(rows, predictions, args.limit)
    (out / 'scores_local.json').write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding='utf-8')
    official_input = out / 'evaluation_input_with_references.json'
    official_input.write_text(json.dumps(adapted, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    if args.official_script:
        script = args.official_script.resolve()
        result = subprocess.run([sys.executable, str(script), '--file', str(official_input)],
                                capture_output=True, text=True)
        (out / 'official_score_output.txt').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
        (out / 'official_evaluator_metadata.json').write_text(json.dumps({
            'script_path': str(script), 'sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
            'returncode': result.returncode, 'full_benchmark': scores['full_benchmark']}, indent=2), encoding='utf-8')
        print(result.stdout)
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
