from contextgem import Aspect


# Define an aspect focused on termination clauses
termination_aspect = Aspect(
    name="Termination provisions",
    description="Contract termination conditions, notice requirements, and severance terms.",
    reference_depth="sentences",
    add_justifications=True,
    justification_depth="comprehensive",
)
