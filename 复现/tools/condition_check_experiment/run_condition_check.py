#!/usr/bin/env python3
"""Frozen MiniCPM4.1-8B + word-chunk BM25 baseline. Generation sees query only.

No gold answer or reference evidence is passed to the retriever or generator.
Use evaluate_baseline.py separately after predictions have been written.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import heapq
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from fixed_contexts import FixedContexts

STOP = set("a an the and or of to in on at by for from with as is are was were be been being it its this that these those do does did has have had who what which when where how why according reported reports article articles both".split())
from prompt_policy import INSTRUCTION


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terms(text):
    return [t for t in re.findall(r'[a-z0-9]+', text.lower()) if t not in STOP]


def load_inputs(root):
    corpus_path = root / 'data/raw/corpus.json'
    qa_path = root / 'data/raw/MultiHopRAG.json'
    corpus = json.loads(corpus_path.read_text(encoding='utf-8-sig'))
    # Immediately project the question file to ID + query. All other fields are discarded.
    questions = [{'question_id': f'q{i:04d}', 'query': row['query']}
                 for i, row in enumerate(json.loads(qa_path.read_text(encoding='utf-8-sig')))]
    if len(corpus) != 609 or len(questions) != 2556:
        raise ValueError(f'Expected full official inputs: 609 docs / 2556 queries; got {len(corpus)} / {len(questions)}')
    if any(not isinstance(q['query'], str) or not q['query'].strip() for q in questions):
        raise ValueError('Invalid question text')
    return corpus, questions, {'corpus': sha256_file(corpus_path), 'queries': sha256_file(qa_path)}


def build_chunks(corpus, size=250, overlap=50):
    chunks = []
    for i, doc in enumerate(corpus):
        body = doc.get('body')
        if not isinstance(body, str) or not body.strip():
            raise ValueError(f'Invalid document body: d{i:04d}')
        spans = list(re.finditer(r'\S+', body))
        for start in range(0, len(spans), size - overlap):
            end = min(start + size, len(spans))
            lo, hi = spans[start].start(), spans[end - 1].end()
            chunks.append({'chunk_id': f'd{i:04d}_w{start:05d}', 'document_id': f'd{i:04d}',
                           **{k: str(doc.get(k, '')) for k in ('title', 'source', 'published_at', 'url')},
                           'word_start': start, 'word_end': end,
                           'char_start': lo, 'char_end': hi, 'text': body[lo:hi]})
            if end == len(spans):
                break
    return chunks


class BM25:
    """BM25 with k1=1.5, b=.75 and positive Robertson IDF log(1+(N-df+.5)/(df+.5))."""
    def __init__(self, chunks):
        self.chunks = chunks
        self.postings = defaultdict(list)
        self.lengths = []
        for i, c in enumerate(chunks):
            ts = terms(c['title'] + ' ' + c['source'] + ' ' + c['published_at'] + ' ' + c['text'])
            self.lengths.append(len(ts))
            for term, tf in Counter(ts).items():
                self.postings[term].append((i, tf))
        self.avg = sum(self.lengths) / max(1, len(self.lengths))

    def retrieve(self, query, top_k=8, max_per_doc=2):
        scores = defaultdict(float)
        count = len(self.chunks)
        for term in set(terms(query)):
            entries = self.postings.get(term, [])
            idf = math.log1p((count - len(entries) + .5) / (len(entries) + .5))
            for i, tf in entries:
                denom = tf + 1.5 * (.25 + .75 * self.lengths[i] / max(self.avg, 1))
                scores[i] += idf * tf * 2.5 / denom
        selected, uses = [], Counter()
        # Stable ties use corpus/chunk order. Diversity cap is applied to this full ranking.
        for i in heapq.nlargest(len(scores), scores, key=lambda i: (scores[i], -i)):
            doc_id = self.chunks[i]['document_id']
            if uses[doc_id] >= max_per_doc:
                continue
            selected.append({**self.chunks[i], 'bm25_score': scores[i]})
            uses[doc_id] += 1
            if len(selected) >= top_k:
                break
        return selected


def render_chunk(chunk):
    return (f"[{chunk['chunk_id']}]\nTitle: {chunk['title']}\nSource: {chunk['source']}\n"
            f"Published: {chunk['published_at']}\nURL: {chunk['url']}\nText: {chunk['text']}")


def pack_prompt(tokenizer, query, retrieved, max_input_tokens):
    used = []
    def render(items):
        context = '\n\n'.join(render_chunk(c) for c in items) or '(No relevant excerpts retrieved.)'
        user = INSTRUCTION.format(context=context, query=query)
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': user}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return context, user, prompt
    for chunk in retrieved:
        trial = render(used + [chunk])
        if len(tokenizer.encode(trial[2], add_special_tokens=True)) <= max_input_tokens:
            used.append(chunk)
    context, user, prompt = render(used)
    if len(tokenizer.encode(prompt, add_special_tokens=True)) > max_input_tokens:
        raise ValueError('Question plus instructions exceeds input-token budget')
    return used, context, user, prompt


def clean_answer(raw):
    text = raw.strip()
    if '<think>' in text:
        if '</think>' not in text:
            return ''
        text = text.rsplit('</think>', 1)[1].strip()
    text = re.sub(r'^(?:final answer|answer)\s*:\s*', '', text, flags=re.I).strip()
    text = text.splitlines()[0].strip() if text else ''
    text = text.strip('"').strip()
    if text.rstrip('.').casefold() == 'insufficient information':
        return 'Insufficient Information'
    if text.rstrip('.').casefold() in ('yes', 'no'):
        return text.rstrip('.').capitalize()
    return text


def read_journal(path):
    if not path.exists():
        return []
    records, good_end = [], 0
    raw = path.read_bytes()
    for line in raw.splitlines(keepends=True):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if good_end + len(line) != len(raw):
                raise RuntimeError('Corrupt journal before final line')
            with path.open('r+b') as f:
                f.truncate(good_end)
            break
        records.append(row)
        good_end += len(line)
    if good_end and path.read_bytes()[-1:] != b'\n':
        with path.open('ab') as f:
            f.write(b'\n')
    return records


def export(out, questions, records, chunk_count):
    by_id = {r['question_id']: r for r in records}
    predictions = [{'question_id': q['question_id'], 'model_answer': by_id[q['question_id']]['model_answer']}
                   for q in questions if q['question_id'] in by_id]
    atomic_json(out / 'predictions.json', predictions)
    complete = len(predictions) == len(questions)
    success = sum(r['status'] == 'ok' for r in records)
    atomic_json(out / 'summary.json', {'evaluated_question_count': len(questions),
        'processed': len(predictions), 'successful': success, 'failed': len(records) - success,
        'complete': complete, 'full_benchmark': complete and len(questions) == 2556,
        'chunk_count': chunk_count, 'model_calls': sum(r['model_calls'] for r in records),
        'input_tokens': sum(r['input_tokens'] for r in records),
        'output_tokens': sum(r['output_tokens'] for r in records),
        'question_runtime_seconds': round(sum(r['elapsed_seconds'] for r in records), 2),
        'peak_allocated_gib': max((r['peak_allocated_gib'] for r in records), default=0),
        'note': 'A partial predictions.json is not a full-benchmark submission. No quality score is calculated here.'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=Path('/root/autodl-tmp/multihop_rag'))
    p.add_argument('--limit', type=int, default=200, help='First N questions; 0 means all 2556')
    p.add_argument('--top-k', type=int, default=8)
    p.add_argument('--max-per-doc', type=int, default=2)
    p.add_argument('--chunk-words', type=int, default=250)
    p.add_argument('--overlap-words', type=int, default=50)
    p.add_argument('--max-input-tokens', type=int, default=6000)
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--retries', type=int, default=1, help='Extra attempts after an error')
    p.add_argument('--output', type=Path)
    p.add_argument('--dry-run', action='store_true', help='CPU retrieval check; no model load or prediction writes')
    args = p.parse_args()
    if args.limit < 0 or not 0 <= args.overlap_words < args.chunk_words or min(args.top_k, args.max_per_doc, args.max_input_tokens, args.max_new_tokens) < 1 or args.retries < 0:
        p.error('Invalid size, limit, or retry settings')
    started = time.monotonic()
    root = args.project_root.resolve()
    corpus, questions, hashes = load_inputs(root)
    questions = questions[:args.limit] if args.limit else questions
    chunks = build_chunks(corpus, args.chunk_words, args.overlap_words)
    if (args.top_k, args.max_per_doc, args.chunk_words, args.overlap_words, args.max_input_tokens, args.max_new_tokens) != (8, 2, 250, 50, 6000, 128):
        p.error('Keep original configuration; only the instruction changes')
    retriever = FixedContexts(root, chunks, questions, hashes)
    print(f'Corpus: {len(corpus)} documents; chunks: {len(chunks)}; questions this run: {len(questions)}', flush=True)
    if args.dry_run:
        for q in questions[:3]:
            hits = retriever.retrieve(q['query'], args.top_k, args.max_per_doc)
            print(q['question_id'], q['query'])
            for h in hits[:3]:
                print(' ', h['chunk_id'], round(h['bm25_score'], 3), h['title'])
        print('CPU retrieval check passed; no model loaded and no predictions written.')
        return
    model_path = root / 'models/MiniCPM4.1-8B'
    if not (model_path / 'config.json').exists():
        raise FileNotFoundError(f'Missing local model: {model_path}')
    out = (args.output or root / 'outputs/minicpm_condition_check_pilot').resolve()
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {'pipeline_version': 'condition-check-1', 'context_source': retriever.signature,
        'prompt_policy_sha256': sha256_file(Path(__file__).with_name('prompt_policy.py')), 'model': 'openbmb/MiniCPM4.1-8B',
        'model_path': str(model_path), 'model_config_sha256': sha256_file(model_path / 'config.json'),
        'model_code_hashes': {f.name: sha256_file(f) for f in sorted(model_path.glob('*.py'))},
        'data_hashes': hashes, 'script_sha256': sha256_file(Path(__file__)),
        **{k: getattr(args, k) for k in ('top_k', 'max_per_doc', 'chunk_words', 'overlap_words',
            'max_input_tokens', 'max_new_tokens', 'retries')},
        'bm25': {'k1': 1.5, 'b': .75, 'idf': 'log(1+(N-df+0.5)/(df+0.5))'},
        'precision': 'bfloat16', 'quantization': None, 'thinking': False,
        'decoding': 'greedy', 'seed': 42,
        'packages': {k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'accelerate')}}
    manifest = out / 'run_config.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise RuntimeError('Settings/code/data changed. Use a new --output folder.')
    atomic_json(manifest, config)
    journal = out / 'execution_records.jsonl'
    records = read_journal(journal)
    ids = [r['question_id'] for r in records]
    full_ids = {f'q{i:04d}' for i in range(2556)}
    if len(set(ids)) != len(ids) or any(i not in full_ids for i in ids):
        raise ValueError('Duplicate or unknown question IDs in journal')
    # Allow extending a completed 10-question pilot to all questions in the same folder.
    if set(ids) - {q['question_id'] for q in questions}:
        raise ValueError('Requested limit is smaller than the existing journal; use original/larger limit')
    export(out, questions, records, len(chunks))
    done = set(ids)
    pending = [q for q in questions if q['question_id'] not in done]
    if not pending:
        print('All requested questions already processed:', out)
        return
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for generation')
    set_seed(42)
    print('Loading local MiniCPM4.1-8B in BF16...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    retriever.preflight(tokenizer, INSTRUCTION, out, args.max_input_tokens)
    print('All unchanged baseline contexts fit the new prompt; loading MiniCPM...', flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
        local_files_only=True, dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval()
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos[0] if isinstance(eos, list) else eos
    gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False,
        eos_token_id=eos, pad_token_id=pad)
    atomic_json(out / 'hardware.json', {'gpu': torch.cuda.get_device_name(0),
        'total_vram_gib': torch.cuda.get_device_properties(0).total_memory / 1024**3,
        'torch_cuda': torch.version.cuda, 'python': sys.version, 'model_class': type(model).__name__,
        'model_total_parameters': model.num_parameters(),
        'load_and_index_seconds_this_process': round(time.monotonic() - started, 2)})
    for q in pending:
        tick = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        hits = retriever.retrieve(q['query'], args.top_k, args.max_per_doc)
        used, context, user, prompt = pack_prompt(tokenizer, q['query'], hits, args.max_input_tokens)
        if used != hits or context != retriever.rows[q['query']]['context']:
            raise RuntimeError('Original evidence would change; stop this controlled experiment')
        record = {**q, 'time_utc': datetime.now(timezone.utc).isoformat(),
            'retrieval_candidates': hits, 'context_chunks': used, 'context': context,
            'messages': [{'role': 'user', 'content': user}], 'rendered_prompt': prompt,
            'model_answer': '', 'raw_output': '', 'status': 'failed', 'model_calls': 0,
            'input_tokens': 0, 'output_tokens': 0, 'errors': []}
        for attempt in range(args.retries + 1):
            try:
                inputs = tokenizer(prompt, return_tensors='pt').to('cuda:0')
                n = inputs['input_ids'].shape[1]
                record['model_calls'] += 1
                record['input_tokens'] += n
                with torch.inference_mode():
                    output = model.generate(**inputs, generation_config=gen_config)
                generated = output[0, n:]
                record['output_tokens'] += len(generated)
                record['raw_output'] = tokenizer.decode(generated, skip_special_tokens=True)
                record['model_answer'] = clean_answer(record['raw_output'])
                if not record['model_answer']:
                    raise ValueError('Empty final answer')
                record['status'] = 'ok'
                break
            except torch.OutOfMemoryError as exc:
                record['errors'].append({'attempt': attempt + 1, 'error': type(exc).__name__, 'message': str(exc)[:1500]})
                torch.cuda.empty_cache()
                break
            except Exception as exc:
                record['errors'].append({'attempt': attempt + 1, 'error': type(exc).__name__, 'message': str(exc)[:1500]})
        record['elapsed_seconds'] = round(time.monotonic() - tick, 3)
        record['peak_allocated_gib'] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        with journal.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        records.append(record)
        export(out, questions, records, len(chunks))
        print(f"[{len(records)}/{len(questions)}] {q['question_id']} [{record['status']}] "
              f"{record['model_answer']} ({record['elapsed_seconds']:.1f}s)", flush=True)
        if record['status'] != 'ok':
            print('  Error:', record['errors'][-1], flush=True)
            # Preserve the failed ID and stop to diagnose; do not burn 2556 failed calls.
            print('Stopped after failed question. Inspect journal; failed rows are not retried on resume.', flush=True)
            return
    print('Finished:', out, flush=True)
    print('Next: run evaluate_answers.py separately. No training was performed.', flush=True)


if __name__ == '__main__':
    main()
