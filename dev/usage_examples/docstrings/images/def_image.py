from pathlib import Path

from contextgem import Document, Image, image_to_base64


# Path is adapted for doc tests
current_file = Path(__file__).resolve()
root_path = current_file.parents[4]

# Using the utility function to convert an image file to base64
image_path = root_path / "tests" / "images" / "invoices" / "invoice.jpg"
base64_data = image_to_base64(image_path)

# Create an image instance with the base64-encoded data
jpg_image = Image(mime_type="image/jpg", base64_data=base64_data)

# Using pre-encoded base64 data directly
png_image = Image(
    mime_type="image/png",
    base64_data="base64-string",  # image as a base64 string
)

# Using a different supported image format
webp_image = Image(
    mime_type="image/webp",
    base64_data=image_to_base64(
        root_path / "tests" / "images" / "invoices" / "invoice.webp"
    ),
)

# Attaching an image to a document
# Documents can contain both text and multiple images, or just images

# Create a document with text content
text_document = Document(
    raw_text="This is a document with an attached image that shows an invoice.",
    images=[jpg_image],
)

# Create a document with only image content (no text)
image_only_document = Document(images=[jpg_image])
