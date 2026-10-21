# ContextGem: Creating a Document from a File Converted with Docling

from docling.document_converter import DocumentConverter

from contextgem import Document


# Convert the file with Docling, e.g. a PDF, DOCX, PPTX, XLSX or HTML file
result = DocumentConverter().convert("path/to/service_agreement.pdf")

# Create a document from the Docling document
document = Document.from_docling(result.document)

# The additional context of each paragraph gives its place in the structure of
# the file, and is sent to the models together with the paragraph, e.g. when
# aspects are extracted. Its page numbers are the pages that its text is on.
for paragraph in document.paragraphs[5:10]:
    print(paragraph.page_numbers, paragraph.additional_context)
    print("   ", paragraph.raw_text)
# [1] Heading, level 1
#     2. Fees and Payment
# [1] Section: 2. Fees and Payment
#     2.1 The Customer shall pay the monthly fees in Table 1. The fees exclude VAT.
# [1] Table caption; Section: 2. Fees and Payment
#     Table 1: Monthly fees
# [1] Table header; Table caption: Table 1: Monthly fees; Section: 2. Fees and Payment
#     Service; Monthly fee; Billing
# [1] Table row; Table caption: Table 1: Monthly fees; Section: 2. Fees and Payment
#     Service: Hosting; Monthly fee: 1,200 EUR; Billing: In advance
