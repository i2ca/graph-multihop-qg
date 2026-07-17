import pandas as pd
from extract import Extract
from llm_openai import LlmOpenaiApi
from llm_local import LlmLocalApi
import json

output_file = "./test_questions/gpt5-100.json"
max_questions = 100

extract = Extract()

rank_type_sequences = [
    ['ORGANIZATION', 'ORGANIZATION', 'PROCESS'],
    ['ORGANIZATION', 'PROCESS', 'PERSON_ROLE'], 
    ['PROCESS', 'PERSON_ROLE', 'PROCESS'],
    ['PROCESS', 'ORGANIZATION', 'PROCESS'], 
    ['ORGANIZATION', 'PROCESS', 'PROCESS'],
    ['PROCESS', 'PROCESS', 'PERSON_ROLE'],
    ['ORGANIZATION', 'ORGANIZATION', 'PERSON_ROLE'],
    ['ORGANIZATION', 'ORGANIZATION', 'ORGANIZATION'],
    ['ORGANIZATION', 'PROCESS', 'ORGANIZATION'],
    ['PROCESS', 'PERSON_ROLE', 'PERSON_ROLE'],
    ['PROCESS', 'ORGANIZATION', 'ORGANIZATION'],
    ['PROCESS', 'ORGANIZATION', 'PERSON_ROLE'], 
    ['ORGANIZATION', 'PERSON_ROLE', 'PROCESS'],
    ['PERSON_ROLE', 'PROCESS', 'PROCESS'],
    ['PROCESS', 'PROCESS', 'ORGANIZATION'],
    ['PERSON_ROLE', 'PROCESS', 'PERSON_ROLE'],
    ['PROCESS', 'PROCESS', 'PROCESS'],
    ['PROCESS', 'PERSON_ROLE', 'ORGANIZATION'],
    ['PERSON_ROLE', 'PROCESS', 'ORGANIZATION'],
    ['ORGANIZATION', 'PERSON_ROLE', 'PERSON_ROLE'],
    ['ORGANIZATION', 'ORGANIZATION', 'GEO'],
    ['PERSON_ROLE', 'ORGANIZATION', 'PERSON_ROLE'],
    ['PERSON_ROLE', 'ORGANIZATION', 'PROCESS'],
    ['GEO', 'ORGANIZATION', 'PROCESS'],
    ['PROCESS', 'ORGANIZATION', 'GEO'],
    ['PERSON_ROLE', 'PERSON_ROLE', 'PROCESS'],
    ['GEO', 'ORGANIZATION', 'ORGANIZATION'],
    ['ORGANIZATION', 'PERSON_ROLE', 'ORGANIZATION'],
    ['PERSON_ROLE', 'PERSON_ROLE', 'ORGANIZATION'],
    ['ORGANIZATION', 'GEO', 'ORGANIZATION'],
    ['PERSON_ROLE', 'ORGANIZATION', 'GEO']
]


n_questions = 0
qa_list = []


for type_sequence in rank_type_sequences:
    entity_sequence_list = extract.list_by_type_sequence(type_sequence)
    for entity_sequence in entity_sequence_list:
        prompt = extract.get_qa_prompt(entity_sequence)

        qa_obj = {
            "type_sequence": type_sequence,
            "entity_sequence": entity_sequence,
            "prompt": prompt
        }

        qa_list.append(qa_obj)

        n_questions += 1

        if n_questions >= max_questions:
            break
    if n_questions >= max_questions:
        break


llmApi = LlmLocalApi()

for qa_item in qa_list:
    completion = llmApi.query(qa_item["prompt"])
    qa_item["completion"] = completion
    qa_item["multi-hop-auto"] = True
    qa_item["multi-hop-manual"] = True

    print(completion)
    print("\n\n")


with open(output_file, 'w+', encoding='utf-8') as file1:
    json.dump(qa_list, file1)

print(qa_list)
