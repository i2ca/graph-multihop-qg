from datasets import load_dataset
from mhq_evaluator import MultiHopQuestionEvaluator
import json

output_file = "./test_evaluator/musique-evaluation.json"

n_questions = 100
random_seed = 44
    
dataset = load_dataset(path="dgslibisey/MuSiQue", name="default", split="train")
shuffled_dataset = dataset.shuffle(random_seed)
random_sample = shuffled_dataset.select(range(n_questions))

question_list = []

for item in random_sample:

    mh_qa = {
        "context": item["paragraphs"],
        "question": item["question"],
        "answer": item["answer"],
        "answerable": None,
        "multi-hop": None
    }

    question_list.append(mh_qa)

    for sh_question in item["question_decomposition"]:
        generated = False
        if ">>" in sh_question['question']:
            generated = True
        if "#1" in sh_question['question']:
            generated = True
     
        if not generated:
            print(sh_question)
            sh_qa = {
                "context": item["paragraphs"],
                "question": sh_question["question"],
                "answer": sh_question["answer"],
                "answerable": None,
                "multi-hop": None
            }
            question_list.append(sh_qa)

with open(output_file, 'w+', encoding='utf-8') as file1:
    json.dump(question_list, file1, indent="    ")