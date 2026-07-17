import pandas as pd
from extract import Extract
from llm_openai import LlmOpenaiApi
from llm_local import LlmLocalApi
import json

output_file = "./test_questions/gpt5-100-test-a.json"
max_questions = 10

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


llmApi = LlmOpenaiApi(model="gpt-5")

for type_sequence in rank_type_sequences:
    entity_sequence_list = extract.list_by_type_sequence(type_sequence)
    for entity_sequence in entity_sequence_list:
        prompt1 = extract.get_single_hop_prompt(entity_sequence[0], entity_sequence[1])
        completion1 = llmApi.query(prompt1)

        prompt2 = extract.get_single_hop_prompt(entity_sequence[1], entity_sequence[2])
        completion2 = llmApi.query(prompt2)

        prompt = extract.get_composition_prompt(completion1, completion2)
        completion = llmApi.query(prompt)

        qa_obj = {
            "question1": completion1,
            "question2": completion2,
            "mh_question": completion
        }

        qa_list.append(qa_obj)

        print(completion)
        print("\n\n")

        n_questions += 1

        if n_questions >= max_questions:
            break
    if n_questions >= max_questions:
        break

with open(output_file, 'w+', encoding='utf-8') as file1:
    json.dump(qa_list, file1)

print(qa_list)
