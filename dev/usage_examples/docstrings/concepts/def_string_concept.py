from contextgem import StringConcept, StringExample


# Define a string concept for identifying contract party names
# and their roles in the contract
party_names_and_roles_concept = StringConcept(
    name="Party names and roles",
    description=(
        "Names of all parties entering into the agreement and their contractual roles"
    ),
    examples=[
        StringExample(
            content="X (Client)",  # guidance regarding format
        )
    ],
)
