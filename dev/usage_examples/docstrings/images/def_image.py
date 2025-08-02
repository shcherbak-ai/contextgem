from pathlib import Path

from contextgem import Document, Image, create_image, image_to_base64


# Path is adapted for doc tests
current_file = Path(__file__).resolve()
root_path = current_file.parents[4]

# Using the create_image utility function (recommended approach)
image_path = root_path / "tests" / "images" / "invoices" / "invoice.jpg"
jpg_image = create_image(
    image_path
)  # Automatically detects MIME type and converts to base64

# Using pre-encoded base64 data directly
png_image = Image(
    mime_type="image/png",
    base64_data="base64-string",  # image as a base64 string
)

# Using a different supported image format with create_image
webp_image = create_image(root_path / "tests" / "images" / "invoices" / "invoice.webp")

# Alternative: Manual approach using image_to_base64 (when you need specific control)
manual_image = Image(mime_type="image/jpeg", base64_data=image_to_base64(image_path))

# Attaching an image to a document
# Documents can contain both text and multiple images, or just images

# Create a document with text content
text_document = Document(
    raw_text="This is a document with an attached image that shows an invoice.",
    images=[jpg_image],
)

# Create a document with only image content (no text)
image_only_document = Document(images=[jpg_image])
