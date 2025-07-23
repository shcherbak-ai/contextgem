from contextgem import LabelConcept


# Multi-class classification: single label selection
document_type_concept = LabelConcept(
    name="Document Type",
    description="Classify the type of legal document",
    labels=["NDA", "Consultancy Agreement", "Privacy Policy", "Other"],
    classification_type="multi_class",
    singular_occurrence=True,
)

# Multi-label classification: multiple label selection
content_topics_concept = LabelConcept(
    name="Content Topics",
    description="Identify all relevant topics covered in the document",
    labels=["Finance", "Legal", "Technology", "HR", "Operations", "Marketing"],
    classification_type="multi_label",
    add_justifications=True,
    justification_depth="brief",  # add justifications for the selected labels
)
