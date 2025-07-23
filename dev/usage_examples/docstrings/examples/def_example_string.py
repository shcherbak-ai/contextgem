from contextgem import StringConcept, StringExample


# Create string examples
string_examples = [
    StringExample(content="X (Client)"),
    StringExample(content="Y (Supplier)"),
]

# Attach string examples to a StringConcept
string_concept = StringConcept(
    name="Contract party name and role",
    description="The name and role of the contract party",
    examples=string_examples,  # Attach the example to the concept (optional)
)
