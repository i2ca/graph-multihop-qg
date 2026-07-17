import pandas as pd
from collections import defaultdict, Counter

class Extract():

    def __init__(self):
        relationships_path = './graphrag/create_final_relationships.parquet'
        entities_path = './graphrag/create_final_entities.parquet'

        self.df_relationships = pd.read_parquet(relationships_path)
        self.df_entities = pd.read_parquet(entities_path)

    # List all entities with a given type
    def list_by_type(self, type):
        filter = (self.df_entities['type'] == type)
        df_filtered = self.df_entities[filter]
        return df_filtered['name'].tolist()

    # List all entities related to a given entity
    def list_related(self, entity):
        filter = (self.df_relationships['source'] == entity)
        df_filtered = self.df_relationships[filter]
        return df_filtered['target'].tolist()

    # List all entities that have a relationship with the given entity and that are of the given type
    def list_related_with_type(self, entity, type):
        df_related_entities = self.list_related(entity)
        filter = self.df_entities['name'].isin(df_related_entities)
        
        output = self.df_entities[filter]
        filter_2 = (output['type'] == type)
        output = output[filter_2]

        return output['name'].tolist()

    # Input: List of strings
    # Return type: List of lists of strings
    # Return all possible paths in a graph with the given type sequence
    def list_by_type_sequence(self, list_types):
        list_trios = list()
        for i, type in enumerate(list_types):

            if i == 0:
                next_entities = self.list_by_type(type)
                for entity in next_entities:
                    list_entity_sequence = list()
                    list_entity_sequence.append(entity)
                    list_trios.append(list_entity_sequence)
            else:
                list_new_trios = list()
                for element in list_trios:
                    entity = element[-1]
                    new_related = self.list_related_with_type(entity, type)

                    for haah in new_related:
                        new_element = element.copy()
                        new_element.append(haah)
                        list_new_trios.append(new_element)
                list_trios = list_new_trios
        return list_trios

    def return_by_name(self, name):
        df_filter = self.df_entities['name'] == name      
        output = self.df_entities[df_filter]
        if len(output) > 1:
            raise Exception(f"Two entities have the same name. Name={name}.") 
        return output.iloc[0].to_dict()

    def return_relationship_by_name(self, entity_1, entity_2):
        cond1 = (self.df_relationships['source'] == entity_1) & (self.df_relationships['target'] == entity_2)
        cond2 = (self.df_relationships['source'] == entity_2) & (self.df_relationships['target'] == entity_1)
    
        df_filter = cond1 | cond2

        output = self.df_relationships[df_filter]
        if len(output) > 1:
            raise Exception(f"Two relationships have the same Entities. Entities={entity_1}-{entity_2}.") 
        return output.iloc[0].to_dict()
    
    def property_extract_prompt(example_input, example_output, input):
        prompt = f"""
        <instructions>
        Identify and list all the properties being queried or compared in the following question.
        </instructions>
        <input>
        {example_input}
        </input>
        <output>
        {example_output}
        </output>
        <input>
        {input}
        </input>
        <output>
        """
    
        return prompt

    def entity_property_extract_prompt(example_extraction, context, entities, properties):
        prompt = f"""
        <instructions>
        Given a block of text and a list of entities with associated properties, extract all restrictive claims that impose hard or soft constraints on the possible values of those properties.
        </instructions>
        {example_extraction}
        <entities>
        {entities}
        </entities>
        <properties>
        {properties}
        </properties>
        <context>
        {context}
        </context>
        <output>
        """

        return prompt
    
    def rank_type_sequences(self, N: int, directed: bool = False):
        if N <= 0:
            raise ValueError("N must be >= 1")

        # Map entity name -> type
        name_to_type = dict(zip(self.df_entities['name'], self.df_entities['type']))

        # Build adjacency
        adj = defaultdict(set)
        src_col = self.df_relationships['source']
        tgt_col = self.df_relationships['target']

        # Only keep edges where both endpoints are known entities
        for src, tgt in zip(src_col, tgt_col):
            if src in name_to_type and tgt in name_to_type:
                adj[src].add(tgt)
                if not directed:
                    adj[tgt].add(src)

        seq_counter = Counter()
        seen_paths = set()  # for undirected de-duplication

        def dfs(path, visited):
            if len(path) == N:
                # Canonicalize path for undirected graphs to avoid counting reverse duplicates
                if directed:
                    key = tuple(path)
                else:
                    t = tuple(path)
                    rt = tuple(reversed(path))
                    key = t if t <= rt else rt  # pick lexicographically smaller of forward/reverse
                if key in seen_paths:
                    return
                seen_paths.add(key)

                type_seq = tuple(name_to_type[node] for node in key)
                seq_counter[type_seq] += 1
                return

            last = path[-1]
            for nbr in adj.get(last, ()):
                if nbr not in visited:
                    visited.add(nbr)
                    path.append(nbr)
                    dfs(path, visited)
                    path.pop()
                    visited.remove(nbr)

        # Start DFS from every entity (even isolates—those will only contribute when N == 1)
        for start in name_to_type.keys():
            dfs([start], {start})

        # Rank sequences: most paths first; tie-break by the sequence lexicographically
        ranked = sorted(seq_counter.items(), key=lambda x: (-x[1], x[0]))
        return ranked

    def get_entity_description_string(self, entity_sequence):

        name_to_description = self.df_entities.set_index('name')['description'].to_dict()

        description_string = ""

        for item in entity_sequence:
            description_string += item
            description_string += "\n"
            description_string += name_to_description[item]
            description_string += "\n\n"

        return description_string
    
    def get_relationship_description_string(self, entity_sequence, allow_missing: bool = False):

        description_string = ""

        for a, b in zip(entity_sequence, entity_sequence[1:]):
            cond1 = (self.df_relationships['source'] == a) & (self.df_relationships['target'] == b)
            cond2 = (self.df_relationships['source'] == b) & (self.df_relationships['target'] == a)
            df_pair = self.df_relationships[cond1 | cond2]

            if df_pair.empty:
                if not allow_missing:
                    raise KeyError(f"No relationship found between '{a}' and '{b}'.")
                rel_desc = ""
            else:
                if len(df_pair) > 1:
                    # Mirror your class's duplicate-relationship behavior.
                    raise Exception(f"Two relationships have the same Entities. Entities={a}-{b}.")
                rel_desc = df_pair.iloc[0]['description']
                if pd.isna(rel_desc):
                    rel_desc = ""

            description_string += f"RELATIONSHIP: {a} - {b}\n{rel_desc}\n\n"

        return description_string
    
    def get_qa_prompt(self, entity_sequence):

        entities_description = self.get_entity_description_string(entity_sequence)
        relationships_description = self.get_relationship_description_string(entity_sequence)

        prompt = f"""
<instructions>
Dado um texto de entrada com uma sequência de entidades com suas descrições e relacionamentos entre essas entidades e suas descrições.
Crie uma pergunta de múltiplos saltos (multi-hop), que exija conhecimento de todas as entidades e relacionamentos para ser respondida. 
Ela deve ser uma questão de múltipla escolha com 5 opções. As perguntas devem ser em português.
</instructions>
<input>
{entities_description}
{relationships_description}
</input>
<output>
"""

        return prompt

    def get_single_hop_prompt(self, entity1, entity2):

        entity_sequence = [entity1, entity2]

        entities_description = self.get_entity_description_string(entity_sequence)
        relationships_description = self.get_relationship_description_string(entity_sequence)

        prompt = f"""
<instructions>
Given two entities and a relationship between them. Create a question about the first entity where the answer of the question is the second entity.
</instructions>
<input>
{entities_description}
{relationships_description}
</input>
<output>
"""

        return prompt
    
    def get_composition_prompt(self, question1, question2):

        prompt = f"""
<instructions>
Given two entities and a relationship between them. Create a question about the first entity where the answer of the question is the second entity.
</instructions>
<input>
Question 1: What is the city where Christ the Redeemer is located?
Question 2: Who is the mayor of Rio de Janeiro?
</input>
<output>
Who is the mayor of the city where Christ the Redeemer is located?
</output>
<instructions>
Given two entities and a relationship between them. Create a question about the first entity where the answer of the question is the second entity.
</instructions>
<input>
Question 1: {question1}
Question 2: {question2}
</input>
<output>
"""

        return prompt