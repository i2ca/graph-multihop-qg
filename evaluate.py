import pandas as pd

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



