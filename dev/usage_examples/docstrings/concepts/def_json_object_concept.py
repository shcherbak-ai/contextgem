from typing import Literal

from contextgem import JsonObjectConcept


# Define a JSON object concept for capturing address information
address_info_concept = JsonObjectConcept(
    name="Address information",
    description=(
        "Structured address data from text including street, "
        "city, state, postal code, and country."
    ),
    structure={
        "street": str | None,
        "city": str | None,
        "state": str | None,
        "postal_code": str | None,
        "country": str | None,
        "address_type": Literal["residential", "business"] | None,
    },
)
