import ast
import json
from pathlib import Path
import tempfile
import unittest
from baseline_core import render_chunk,sha256_file
from fixed_contexts import FixedContexts
from prompt_policy import INSTRUCTION,ORIGINAL_INSTRUCTION,CHECK_INSTRUCTION
from evaluate_answers import evaluate


class PromptExperimentChecks(unittest.TestCase):
    def test_only_check_instruction_added(self):
        self.assertEqual(INSTRUCTION.replace('\n'+CHECK_INSTRUCTION+'\n','\n'),ORIGINAL_INSTRUCTION)
        self.assertEqual(INSTRUCTION.count('{context}'),1)
        self.assertEqual(INSTRUCTION.count('{query}'),1)

    def test_context_packing_and_answer_extraction_are_frozen(self):
        root=Path(__file__).parent
        def nodes(name):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse((root/name).read_text()).body
                if isinstance(n,ast.FunctionDef) and n.name in {'pack_prompt','clean_answer','render_chunk','export','read_journal'}}
        self.assertEqual(nodes('baseline_core.py'),nodes('run_condition_check.py'))

    def fixture(self,root):
        folder=root/'outputs/minicpm_bm25_baseline';folder.mkdir(parents=True)
        chunks=[{'chunk_id':str(i),'document_id':str(i//2),'title':'T','source':'S','published_at':'D',
                 'url':'U','text':'Evidence '+str(i)} for i in range(8)]
        config={'data_hashes':{'data':'hash'},'top_k':8,'max_per_doc':2,'chunk_words':250,'overlap_words':50,
            'max_input_tokens':6000,'max_new_tokens':128,'precision':'bfloat16','thinking':False,'decoding':'greedy','seed':42,
            'script_sha256':sha256_file(Path(__file__).with_name('baseline_core.py'))}
        (folder/'run_config.json').write_text(json.dumps(config))
        row={'question_id':'q0000','query':'Q','status':'ok','model_answer':'OLD ANSWER MUST NOT ENTER PROMPT',
            'context_chunks':chunks,'context':'\n\n'.join(render_chunk(c) for c in chunks)}
        (folder/'execution_records.jsonl').write_text(json.dumps(row)+'\n')
        return folder,chunks,[{'question_id':'q0000','query':'Q'}]

    def test_exact_saved_context_and_no_old_answer_exposure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder,chunks,queries=self.fixture(root)
            loader=FixedContexts(root,chunks,queries,{'data':'hash'})
            self.assertEqual(loader.retrieve('Q'),chunks)
            self.assertNotIn('OLD ANSWER',json.dumps(loader.rows))
            row=json.loads((folder/'execution_records.jsonl').read_text())
            row['context_chunks'][0]['text']='CORRUPT'
            (folder/'execution_records.jsonl').write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):FixedContexts(root,chunks,queries,{'data':'hash'})

    def test_overlength_new_prompt_stops_without_dropping_evidence(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):return messages[0]['content']
            def encode(self,prompt,**kwargs):return [0]*6001
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,chunks,queries=self.fixture(root)
            loader=FixedContexts(root,chunks,queries,{'data':'hash'})
            with self.assertRaises(ValueError):loader.preflight(Tokenizer(),INSTRUCTION,root,6000)
            self.assertEqual(len(loader.retrieve('Q')),8)

    def test_missing_prediction_is_still_wrong(self):
        refs=[{'query':'A','answer':'Yes','question_type':'comparison_query'},
              {'query':'B','answer':'No','question_type':'comparison_query'}]
        scores,_=evaluate(refs,[{'question_id':'q0000','model_answer':'Yes'}])
        self.assertEqual(scores['results']['overall']['accuracy_percent'],50)


if __name__=='__main__':unittest.main()
