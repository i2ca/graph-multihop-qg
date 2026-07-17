from datasets import load_dataset
from mhq_evaluator import MultiHopQuestionEvaluator
import json

output_file = "./test_evaluator/wikihop-evaluation.json"

n_questions = 100
random_seed = 45
    
dataset = load_dataset(path="MoE-UNC/wikihop", name="default", split="train")
shuffled_dataset = dataset.shuffle(random_seed)
random_sample = shuffled_dataset.select(range(n_questions))

question_list = []

for item in random_sample:

    mh_qa = {
        "context": item["supports"],
        "question": item["query"],
        "answer": item["answer"],
        "answerable": None,
        "multi-hop": None
    }

    question_list.append(mh_qa)

with open(output_file, 'w+', encoding='utf-8') as file1:
    json.dump(question_list, file1, indent="    ")