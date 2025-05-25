"""
This module provides usage examples for Gemini models with the ContextGem framework.
"""

import os
from contextgem.public.llms import DocumentLLM
from contextgem.public.images import Image

# Ensure your Gemini API key is set as an environment variable.
# LiteLLM typically uses GOOGLE_API_KEY or GEMINI_API_KEY for Google AI Studio models.
# Example: os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY"
# Refer to LiteLLM documentation for the most up-to-date environment variable names.

# --- Gemini Text Model Example ---
try:
    # Instantiate DocumentLLM for a Gemini text model
    gemini_text_llm = DocumentLLM(
        model="gemini/gemini-pro",
        api_key=os.environ.get("GEMINI_API_KEY"), # Or GOOGLE_API_KEY
        # Add other parameters like temperature, max_tokens if needed
    )

    # Simple chat call
    prompt_text = "What is the capital of France?"
    response_text = gemini_text_llm.chat(prompt=prompt_text)
    print(f"Gemini Text Model ({gemini_text_llm.model}) Response to '{prompt_text}':")
    print(response_text)

except Exception as e:
    print(f"Error with Gemini text model example: {e}")
    print("Please ensure your API key is correctly set and the model is accessible.")

print("\n---\n")

# --- Gemini Vision Model Example ---
try:
    # Instantiate DocumentLLM for a Gemini vision model
    gemini_vision_llm = DocumentLLM(
        model="gemini/gemini-pro-vision",
        api_key=os.environ.get("GEMINI_API_KEY"), # Or GOOGLE_API_KEY
        # Add other parameters if needed
    )

    # Simple chat call with a placeholder image
    # In a real scenario, you would load an image using:
    # from contextgem.public.images import Image
    # image = Image.from_path("path/to/your/image.jpg")
    # For this example, we'll simulate a tiny transparent PNG as base64
    # (This is just a placeholder and not a real image of a landmark)
    placeholder_image_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )
    placeholder_image = Image(
        base64_data=placeholder_image_base64,
        mime_type="image/png",
        name="placeholder.png"
    )

    prompt_vision = "Describe this image."
    response_vision = gemini_vision_llm.chat(prompt=prompt_vision, images=[placeholder_image])
    print(f"Gemini Vision Model ({gemini_vision_llm.model}) Response to '{prompt_vision}':")
    print(response_vision)

except Exception as e:
    print(f"Error with Gemini vision model example: {e}")
    print("Please ensure your API key is correctly set, the model is accessible, and supports vision.")

# Note:
# - For actual image loading, use `Image.from_path("path/to/your/image.png")` or
#   `Image.from_url("url_to_image")`.
# - Ensure the model name (e.g., "gemini/gemini-pro-vision") correctly supports vision
#   capabilities as per LiteLLM and Google AI Studio documentation.
# - API keys are crucial. If not set as environment variables, you can pass them directly
#   to the `api_key` parameter during DocumentLLM instantiation, though using environment
#   variables is generally recommended for security.
# - You might need to install additional dependencies for specific image handling if not
#   already covered by your ContextGem installation (e.g., libraries for image processing
#   if you are creating Image objects from complex sources).
#   ContextGem's Image class primarily deals with base64 data and MIME types.
# - The `chat` method is a direct way to interact with the LLM. For structured data
#   extraction (Aspects, Concepts), you would use methods like `extract_all_async`,
#   `extract_aspects_from_document_async`, etc.
# - Refer to the official LiteLLM and Google AI Studio documentation for the latest
#   model names, capabilities, and API key management practices.
