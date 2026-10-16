from contextgem import Document, keep_sat_model_loaded


texts = [
    "The Supplier shall deliver the goods. Delivery is due within 30 days.",
    "The Customer shall pay the invoice. Payment is due within 14 days.",
]

# A SaT model splits the texts into sentences. Outside the block, each document
# would load the model again; in the block, the documents use one model.
# The same applies to extraction with sentence-level references in the block.
documents = []
with keep_sat_model_loaded():
    for text in texts:
        documents.append(Document(raw_text=text, pre_segment_sentences=True))
# The model is released here

for document in documents:
    print([sentence.raw_text for sentence in document.sentences])
