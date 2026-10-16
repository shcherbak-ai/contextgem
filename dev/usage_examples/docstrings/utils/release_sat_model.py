from contextgem import Document, release_sat_model


texts = [
    "The Supplier shall deliver the goods. Delivery is due within 30 days.",
    "The Customer shall pay the invoice. Payment is due within 14 days.",
]

# A SaT model splits the texts into sentences. It is loaded for the first
# document, and kept loaded for the documents that follow.
documents = []
for text in texts:
    documents.append(Document(raw_text=text, pre_segment_sentences=True))

for document in documents:
    print([sentence.raw_text for sentence in document.sentences])

# Free the memory of the model, e.g. when the process segments no more
# documents. A document that needs the model then loads it again.
release_sat_model()
