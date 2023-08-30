"""
This script works with DccAI v0.7.*
It provides potential redaction of specific image regions based on given labels and/or text patterns.
"""
import os
import re
import uuid
import base64
import cv2
import numpy as np
import pandas as pd
from typing import List

from argus.data_model import Document
from argus.processors.post_processors.base_post_processor import BasePostProcessor, BaseEntity
from argus.processors.post_processors.utils.utility import doc_to_df
from argus.processors.post_processors.utils import post_process as pp


"""
User input: labels/text patterns to be redacted.
Valid values:
labels_to_redact = None
labels_to_redact = ['billing_name', 'billing_address']  # Must be model class labels
text_patterns_to_redact = None
text_patterns_to_redact = [r'[A-Za-z]{3}\d{6}', r'\d{3}-\d{2}-\d{4}', r'\b\d{11}\b', r"\b\d{3}-\d{3}-\d{4}\b"]
"""

labels_to_redact = None
text_patterns_to_redact = [r"^\d{3} \d{3} \d{3}$", r'\b\d{9}\b']  # regex pattern: 111 222 333, 111222333


class CustomEntity(BaseEntity):
    image: str


class PostProcessor(BasePostProcessor):
    """
    A post-processor class for processing intermediate results and
    translating them into a final JSON structure for user output.
    """

    def client_resolution(self) -> None:
        return None

    def argus_resolution(self) -> int:
        return self.ARGUS_DPI

    @staticmethod
    def redact_region(image: np.array, xmin: int, ymin: int, xmax: int, ymax: int) -> np.array:
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 0), -1)
        return image

    def get_filtered_predictions(self, doc_predictions) -> pd.DataFrame:
        doc_predictions['text'] = doc_predictions['text'].str.strip()

        # Filter by labels
        label_filtered = doc_predictions[
            doc_predictions['label'].isin(labels_to_redact)] if labels_to_redact else pd.DataFrame()

        # Filter by text patterns
        pattern_filtered = pd.concat([
            doc_predictions[doc_predictions['text'].str.contains(pattern, regex=True)]
            for pattern in text_patterns_to_redact
        ]).drop_duplicates() if text_patterns_to_redact else pd.DataFrame()

        return pd.concat([label_filtered, pattern_filtered]).drop_duplicates()

    def get_entities(self, doc: Document, doc_id: str) -> List[CustomEntity]:

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
                                           verbose=True)

        redacted_images_per_page = {}
        redacted_text_per_page = {}

        for doc in docs:
            docs[doc]["id"] = docs[doc]["label"].apply(lambda row: str(uuid.uuid4()))

            redacted_images_per_page = {}
            processed_page_ids = set()
            for _, row in docs[doc].iterrows():
                page_id = row['page_id']
                if page_id not in processed_page_ids:
                    filename = f"{doc}+{page_id}{self.img_extension}"

                    if ".pdf" not in filename:
                        filename_without_extension = os.path.splitext(doc)[0]
                        filename = f"{filename_without_extension}+{page_id}{self.img_extension}"

                    img_path = os.path.join(self._resource_dir, filename)
                    image = cv2.imread(img_path)
                    if image is not None:
                        redacted_images_per_page[filename] = image.copy()
                        processed_page_ids.add(page_id)

            filtered_predictions = self.get_filtered_predictions(docs[doc])

            for _, row in filtered_predictions.iterrows():
                filename = f"{doc}+{row['page_id']}{self.img_extension}"
                processed_filename = filename

                if ".pdf" not in filename:
                    filename_without_extension = os.path.splitext(doc)[0]
                    processed_filename = f"{filename_without_extension}+{row['page_id']}{self.img_extension}"

                self.redact_region(
                    redacted_images_per_page[processed_filename],
                    int(row["xmin"]), int(row["ymin"]),
                    int(row["xmax"]), int(row["ymax"])
                )

                page_id = row['page_id']
                text_to_redact = row['text']

                # Concatenate redacted text for each page
                if page_id in redacted_text_per_page:
                    redacted_text_per_page[page_id] += " | " + text_to_redact
                else:
                    redacted_text_per_page[page_id] = text_to_redact

        # Create the data bundles
        redacted_data_bundles = []
        for filename, redacted_image in redacted_images_per_page.items():
            _, img_encoded = cv2.imencode(".png", redacted_image)
            page_index = filename.split('+')[1].split(self.img_extension)[0]

            # Get the concatenated redacted text for this page
            concatenated_redacted_text = redacted_text_per_page.get(page_index, "")

            data_bundle: CustomEntity = {
                "pageIndex": page_index,
                "redactedData": concatenated_redacted_text,
                "image": base64.b64encode(img_encoded).decode("ascii")

            }

            redacted_data_bundles.append(data_bundle)

        return redacted_data_bundles
