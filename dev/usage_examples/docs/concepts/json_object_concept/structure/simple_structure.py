from contextgem import JsonObjectConcept

product_info_concept = JsonObjectConcept(
    name="Product Information",
    description="Product details",
    structure={
        "name": str,
        "price": float,
        "is_available": bool,
        "ratings": list[float],
    },
)
