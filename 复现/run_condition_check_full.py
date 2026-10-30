"""Run the unchanged condition-check configuration on all 2556 questions.

Upload this file to /root/autodl-tmp/multihop_rag and run it in the rag environment.
Requires the existing tools/condition_check_experiment directory and original
outputs/minicpm_bm25_baseline logs. Uses a NEW output folder, preserving pilots.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path('/root/autodl-tmp/multihop_rag'))
    args=parser.parse_args()
    root=args.project_root.resolve()
    scripts=root/'tools/condition_check_experiment'
    output=root/'outputs/minicpm_condition_check_full'
    for name in ['run_condition_check.py','evaluate_answers.py']:
        if not (scripts/name).is_file():raise FileNotFoundError(f'Missing existing experiment script: {scripts/name}')
    print('Full confirmation: same condition-check prompt, original eight BM25 chunks, all 2556 queries.',flush=True)
    def run(name):
        subprocess.run([sys.executable,'-u',str(scripts/name),'--project-root',str(root),
            '--limit','0','--output',str(output)],check=True)
    run('run_condition_check.py')
    state=json.loads((output/'summary.json').read_text())
    if state.get('processed')!=2556 or state.get('failed')!=0 or not state.get('full_benchmark'):
        raise SystemExit('Full run incomplete or failed. Inspect logs; no final score/archive produced.')
    check=json.loads((output/'fixed_context_check.json').read_text())
    if check.get('questions')!=2556 or not check.get('all_original_contexts_fit'):
        raise SystemExit('Full fixed-context check is missing or incomplete.')
    run('evaluate_answers.py')
    target=root/'condition_check_full_results.zip'
    temporary=target.with_suffix('.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(output.iterdir()):
            if file.is_file() and not file.name.startswith('.'):
                archive.write(file,file.relative_to(root/'outputs'))
    temporary.replace(target)
    print('\nALL DONE. Result archive:',target,flush=True)
    print('The printed score is local scoring; use the instructor fixed official script for final verification.',flush=True)
    print('No model training or shutdown was performed.',flush=True)


if __name__=='__main__':main()
