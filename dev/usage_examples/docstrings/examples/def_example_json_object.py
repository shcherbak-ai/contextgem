from contextgem import JsonObjectConcept, JsonObjectExample


# Create a JSON object example
json_example = JsonObjectExample(
    content={
        "name": "John Doe",
        "education": "Bachelor's degree in Computer Science",
        "skills": ["Python", "Machine Learning", "Data Analysis"],
        "hobbies": ["Reading", "Traveling", "Gaming"],
    }
)


# Define a structure for JSON object concept
class PersonInfo:
    name: str
    education: str
    skills: list[str]
    hobbies: list[str]


# Also works as a dict with type hints, e.g.
# PersonInfo = {
#     "name": str,
#     "education": str,
#     "skills": list[str],
#     "hobbies": list[str],
# }

# Attach JSON example to a JsonObjectConcept
json_concept = JsonObjectConcept(
    name="Candidate info",
    description="Structured information about a job candidate",
    structure=PersonInfo,  # Define the expected structure
    examples=[json_example],  # Attach the example to the concept (optional)
)
