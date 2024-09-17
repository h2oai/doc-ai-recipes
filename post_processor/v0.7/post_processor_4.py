
from typing import List
from argus.data_model import Document
from argus.processors.post_processors.base_post_processor import BasePostProcessor, BaseEntity
from argus.processors.post_processors.utils.utility import doc_to_df
from argus.processors.post_processors.utils import post_process as pp


class GenericEntity(BaseEntity):
    text: str
    label: str
    labelConfidence: float
    ocrConfidence: float
    xmin: int
    xmax: int
    ymin: int
    ymax: int


class PostProcessor(BasePostProcessor):

    def client_resolution(self):
        return None

    def argus_resolution(self):
        return self.ARGUS_DPI

    def get_entities(self, doc: Document, doc_id: str) -> List[GenericEntity]:

        if not self.has_labelling_model:
            return []

        df_doc = doc_to_df(doc, doc_id, self.token_label_names)

        docs = pp.post_process_predictions(model_preds=df_doc,
                                           top_n_preds=self.class_names,
                                           token_merge_type='MIXED_MERGE',
                                           token_merge_xdist_regular=1.0,
                                           label_merge_x_regular='ALL',
                                           token_merge_xydist_regular=1.0,
                                           label_merge_xy_regular='address',
                                           token_merge_xdist_wide=1.5,
                                           label_merge_x_wide='phone|fax',
                                           output_labels='INCLUDE_O',
                                           parse_line_items=False,
                                           line_item_completeness=0.6,
                                           try_templates=False,
                                           templates_dict_dir='',
                                           templates_input_dir='./',
                                           use_camelot_tables=False,
                                           images_dir_camelot='',
                                           verbose=True)

        df_list = []
        for doc in docs:
            predictions = docs[doc]
            predictions = predictions.round(decimals=4)
            for idx, row in predictions.iterrows():
                # Remove unlabeled tokens
                if row["label"] == "":
                    continue
                df_list.append(row.to_dict())
        return df_list

    '''
    Converting the dictionary to a dataframe
    import pandas as pd
    import json

    f = open('result.json')
    dict_data = json.load(f)
    df = pd.DataFrame(dict_data['entities'])
    df.to_csv('result.csv')
    '''