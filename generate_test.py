import pandas as pd
from extract import Extract
import json
from llm_api import LlmApi
from llm_openai import LlmOpenaiApi


llmApi = LlmOpenaiApi(model="gpt-4o")

extract = Extract()

example_json = {
    "entities": [
        {
            "name": "PRPPG",
            "type": "ORGANIZATION",
            "description": "PRPPG, or the Pró-Reitoria de Pesquisa e Pós-Graduação, is a division within the Universidade Federal do Espírito Santo. It is responsible for overseeing research and postgraduate programs at the university. As the Dean of Research and Graduate Studies, PRPPG plays a crucial role in managing and registering research projects, ensuring that all research initiatives are properly documented and aligned with the university's academic goals. This organization serves as the central body where all research projects must be registered, highlighting its importance in the academic and research landscape of the university."
        },
        {
            "name": "PIBIC",
            "type": "PROCESS",
            "description": "The Programa Institucional de Bolsas de Iniciação Científica (PIBIC) is a scholarship program at the Universidade Federal do Espírito Santo (Ufes) designed to support undergraduate students engaged in scientific research. As a subprogram of the broader Piic initiative, PIBIC provides financial scholarships to students based on a selection process that considers specific criteria and classifications. The program aims to foster academic and research activities by ensuring that students are guided by advisors with high scientific competence and active research involvement. Advisors (orientadores) must be affiliated with Ufes and meet certain academic and employment criteria to participate. Each advisor can receive up to two scholarships for their students. Candidates for PIBIC must be associated with a research project and meet a minimum score for their proposals to be considered. The program's objectives and norms are similar to those of Pivic, but PIBIC uniquely includes scholarship payments to participating students, thereby enhancing their scientific initiation experience."
            },
        {
            "name": "CONSELHO DE ENSINO, PESQUISA E EXTENSÃO",
            "type": "ORGANIZATION",
            "description": "The CONSELHO DE ENSINO, PESQUISA E EXTENSÃO is the council at the Universidade Federal do Espírito Santo responsible for overseeing and approving regulations related to teaching, research, and extension activities."
        }
    ],
    "relationships": [
        {
            "source": "PRPPG",
            "target": "PIBIC",
            "description": "The PRPPG registers research projects that candidates in the Pibic program must be associated with"
            },
        {
            "source": "PIBIC",
            "target": "CNPQ",
            "description": "CNPq oversees the PIBIC program and its scholarship distribution"
        }
    ]
}

example_json_string = json.dumps(example_json)


example = \
f"""<input>
{example_json_string}
</input>
<instructions>
Considerando o input dado, crie uma questão de múltipla escolha.
O enunciado deve exigir que o leitor combine, em cadeia, todos relacionamentos diferentes do JSON para descobrir a resposta correta.
Forneça 5 alternativas rotuladas de A) a E) – exatamente 1 correta e 4 distratores plausíveis
</instructions>
<output>
O CNPq é o órgão federal que supervisiona a concessão de suas bolsas de iniciação científica de um programa de bolsas. No contexto da Ufes, qual é o órgão responsável por registrar os projetos de pesquisa aos quais os candidatos à esse programa de bolsas devem estar vinculados?

A) Departamento de Ensino de Graduação (DEG)
B) Pró-Reitoria de Extensão (ProEx)
C) Pró-Reitoria de Planejamento e Desenvolvimento Institucional (Proplan)
D) Pró-Reitoria de Pesquisa e Pós-Graduação (PRPPG)
E) Secretaria de Relações Internacionais (SRI)
</output>"""

list_types = ["PROCESS", "ORGANIZATION", "PERSON_ROLE"]
a = extract.list_by_type_sequence(list_types)

with open("./output/questions-1.txt", "w") as file:
    for item in a:

        entity_1 = extract.return_by_name(item[0])
        entity_2 = extract.return_by_name(item[1])
        entity_3 = extract.return_by_name(item[2])

        entity_1_dict = {
            "name":entity_1["name"],
            "type":entity_1["type"],
            "description":entity_1["description"]
        }

        relationship_1 = extract.return_relationship_by_name(item[0], item[1])
        relationship_2 = extract.return_relationship_by_name(item[1], item[2])

        final_json = {
            "entities": [
                {
                    "name":entity_1["name"],
                    "type":entity_1["type"],
                    "description":entity_1["description"]
                },
                {
                    "name":entity_2["name"],
                    "type":entity_2["type"],
                    "description":entity_2["description"]
                },
                {
                    "name":entity_3["name"],
                    "type":entity_3["type"],
                    "description":entity_3["description"]
                }
            ],
            "relationships": [
                {
                    "source":relationship_1["source"],
                    "target":relationship_1["target"],
                    "description":relationship_1["description"]
                },
                {
                    "source":relationship_2["source"],
                    "target":relationship_2["target"],
                    "description":relationship_2["description"]
                },
            ]
        }

        json_string = json.dumps(final_json)

        b = \
f"""
{example}
<input>
{json_string}
</input>
<instructions>
Considerando o input dado, crie uma questão de múltipla escolha.
O enunciado deve exigir que o leitor combine, em cadeia, todos relacionamentos diferentes do JSON para descobrir a resposta correta.
Forneça 5 alternativas rotuladas de A) a E) – exatamente 1 correta e 4 distratores plausíveis
</instructions>
<output>
"""
        response = llmApi.query(b)

        print(b)
        print()

        file.write(response)
        file.write("\n\n\n")

        break