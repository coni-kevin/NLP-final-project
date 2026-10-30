"""Only the instruction changes; the answer format stays the original one."""
from baseline_core import INSTRUCTION as ORIGINAL_INSTRUCTION

CHECK_INSTRUCTION = '''Before answering, silently check each condition required by the question against the excerpts: the entity, named source, event time, quantity, and any negation.
For a yes/no question asserting that several conditions all hold, answer Yes only when every condition is supported. Answer No when at least one required condition is contradicted by the evidence. If a necessary condition is neither supported nor contradicted, use Insufficient Information.
Do not substitute another source, entity, event, or time period for the one asked about. A difference in publication dates alone does not establish a change in the reported facts.
For a requested name, number, date, or initial, verify the requested relationships before extracting it. Do not fill a missing relationship from background knowledge.
Perform these checks silently and output only the final short answer, following the original answer format.
'''

INSTRUCTION = ORIGINAL_INSTRUCTION.replace('\nRetrieved excerpts:\n', '\n' + CHECK_INSTRUCTION + '\nRetrieved excerpts:\n')
