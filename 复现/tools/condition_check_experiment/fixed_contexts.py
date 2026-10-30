"""Reuse exact original BM25 context. Never use its old answers for generation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from baseline_core import atomic_json,render_chunk,sha256_file


class FixedContexts:
    def __init__(self,root,chunks,questions,hashes):
        directory=root/'outputs/minicpm_bm25_baseline'
        cfg_path=directory/'run_config.json'
        cfg=json.loads(cfg_path.read_text())
        expected={'top_k':8,'max_per_doc':2,'chunk_words':250,'overlap_words':50,
            'max_input_tokens':6000,'max_new_tokens':128,'precision':'bfloat16',
            'thinking':False,'decoding':'greedy','seed':42}
        if cfg['data_hashes']!=hashes or any(cfg.get(k)!=v for k,v in expected.items()):
            raise ValueError('Baseline settings/data differ from this controlled experiment')
        if cfg['script_sha256']!=sha256_file(Path(__file__).with_name('baseline_core.py')):
            raise ValueError('Baseline source differs from the frozen original')
        index={c['chunk_id']:c for c in chunks}
        self.rows={};snapshots=[]
        with (directory/'execution_records.jsonl').open(encoding='utf-8') as f:
            for i,line in enumerate(f):
                if i>=len(questions):break
                record=json.loads(line);q=questions[i]
                if record['question_id']!=q['question_id'] or record['query']!=q['query'] or record['status']!='ok':
                    raise ValueError('Baseline query order, count, or status differs')
                selected=record['context_chunks']
                if len(selected)!=8 or len({c['chunk_id'] for c in selected})!=8 or max(Counter(c['document_id'] for c in selected).values())>2:
                    raise ValueError('Expected exactly eight original chunks with max two per document')
                for c in selected:
                    if c['chunk_id'] not in index or any(c.get(k)!=v for k,v in index[c['chunk_id']].items()):
                        raise ValueError('Saved baseline excerpt is not the original corpus chunk')
                context='\n\n'.join(render_chunk(c) for c in selected)
                if context!=record['context']:raise ValueError('Saved baseline context formatting differs')
                # Project away the old model outputs and all other journal fields.
                row={**q,'context_chunks':selected,'context':context}
                self.rows[q['query']]=row;snapshots.append(row)
        if len(snapshots)!=len(questions):raise ValueError('Baseline does not contain every requested question')
        self.signature={'method':'exact saved original BM25 context','baseline_config_sha256':sha256_file(cfg_path),
            'selected_contexts_sha256':hashlib.sha256(json.dumps(snapshots,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
            'loader_sha256':sha256_file(Path(__file__)),'question_count':len(questions)}

    def retrieve(self,query,top_k=8,max_per_doc=2):
        if (top_k,max_per_doc)!=(8,2):raise ValueError('Keep original top-k=8 and document cap=2')
        return self.rows[query]['context_chunks']

    def preflight(self,tokenizer,instruction,output,max_input_tokens):
        checks=[]
        for row in self.rows.values():
            user=instruction.format(context=row['context'],query=row['query'])
            prompt=tokenizer.apply_chat_template([{'role':'user','content':user}],tokenize=False,
                add_generation_prompt=True,enable_thinking=False)
            count=len(tokenizer.encode(prompt,add_special_tokens=True))
            if count>max_input_tokens:
                raise ValueError(f"{row['question_id']}: the unchanged context plus the new prompt exceeds {max_input_tokens} tokens; stop rather than drop evidence")
            checks.append({'question_id':row['question_id'],'input_tokens':count,
                'chunk_ids':[c['chunk_id'] for c in row['context_chunks']]})
        atomic_json(output/'fixed_context_check.json',{'all_original_contexts_fit':True,
            'questions':len(checks),'max_input_tokens_observed':max(c['input_tokens'] for c in checks),
            'context_source':self.signature,'checks':checks})
