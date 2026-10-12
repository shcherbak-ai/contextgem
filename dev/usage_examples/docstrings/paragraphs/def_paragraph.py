from contextgem import Paragraph


# Create a paragraph with raw text content
contract_paragraph = Paragraph(
    raw_text=(
        "This agreement is effective as of January 1, 2025. "
        "All parties must comply with the terms outlined herein. "
        "Failure to adhere to these terms may result in termination of the agreement."
    )
)

# Create paragraphs with additional context, e.g. the structure or formatting
# details provided by your document conversion tool. The additional context is
# passed to the model together with the text of the paragraph.
heading_paragraph = Paragraph(
    raw_text="Term and Termination",
    additional_context="Heading, level 2",
)
table_cell_paragraph = Paragraph(
    raw_text="Net 30 days",
    additional_context="Table: Payment terms, row 2, column 3",
)
