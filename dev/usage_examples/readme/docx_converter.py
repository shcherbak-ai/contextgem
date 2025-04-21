# Using ContextGem's DocxConverter

from contextgem import DocxConverter

converter = DocxConverter()

# Convert a DOCX file to a ContextGem Document
# from path
document = converter.convert("path/to/document.docx")
# or from file object
with open("path/to/document.docx", "rb") as docx_file_object:
    document = converter.convert(docx_file_object)

# You can also use it as a standalone text extractor
docx_text = converter.convert_to_text_format(
    "path/to/document.docx",
    output_format="markdown",  # or "raw"
)
