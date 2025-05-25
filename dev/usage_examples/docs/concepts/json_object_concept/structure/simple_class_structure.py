from pydantic import BaseModel

from contextgem import JsonObjectConcept


# Use a Pydantic model to define the structure of the JSON object
class ProductSpec(BaseModel):
    name: str
    version: str
    features: list[str]


product_spec_concept = JsonObjectConcept(
    name="Product Specification",
    description="Technical specifications for a product",
    structure=ProductSpec,
)
