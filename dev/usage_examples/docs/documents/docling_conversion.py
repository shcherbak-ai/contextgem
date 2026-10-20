# ContextGem: Creating a Document from a File Converted with Docling

from docling.document_converter import DocumentConverter

from contextgem import Document


# Convert the file with Docling, e.g. a PDF, DOCX, PPTX, XLSX or HTML file
result = DocumentConverter().convert("path/to/service_agreement.pdf")

# Create a document from the Docling document
document = Document.from_docling(result.document)

# The additional context of each paragraph gives its place in the structure
# of the file, and is sent to the models together with the paragraph
for paragraph in document.paragraphs[7:10]:
    print(f"[{paragraph.additional_context}] {paragraph.raw_text}")
# [Table caption; Section: 2. Fees and Payment] Table 1: Monthly fees
# [Table header; Table caption: Table 1: Monthly fees; Section: 2. Fees and Payment]
#   Service; Monthly fee; Billing
# [Table row; Table caption: Table 1: Monthly fees; Section: 2. Fees and Payment]
#   Service: Hosting; Monthly fee: 1,200 EUR; Billing: In advance

# The custom data of each paragraph holds the page and the bounding box of its
# text in the file
row = document.paragraphs[9]
print(row.custom_data["docling"]["prov"][0]["page_no"])  # 1
