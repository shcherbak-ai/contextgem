# ContextGem: JsonObjectConcept Extraction with Examples

import os
from pprint import pprint

from contextgem import Document, DocumentLLM, JsonObjectConcept, JsonObjectExample


# Document object with ambiguous medical report text
medical_report = """
PATIENT ASSESSMENT
Date: March 15, 2023
Patient: John Doe (ID: 12345)

Vital Signs:
BP: 125/82 mmHg
HR: 72 bpm
Temp: 98.6°F
SpO2: 98%

Chief Complaint:
Patient presents with persistent cough for 2 weeks, mild fever in evenings (up to 100.4°F), and fatigue. 
No shortness of breath. Patient reports recent travel to Southeast Asia 3 weeks ago.

Assessment:
Physical examination shows slight wheezing in upper right lung. No signs of pneumonia on chest X-ray.
WBC slightly elevated at 11,500. Patient appears in stable condition but fatigued.

Impression:
1. Acute bronchitis, likely viral
2. Rule out early TB given travel history
3. Fatigue, likely secondary to infection

Plan:
- Rest for 5 days
- Symptomatic treatment with over-the-counter cough suppressant
- Follow-up in 1 week
- TB test ordered

Dr. Sarah Johnson, MD
"""
doc = Document(raw_text=medical_report)

# Create a JsonObjectConcept for extracting medical assessment data
# Without examples, the LLM might struggle with ambiguous fields or formatting variations
medical_assessment_concept = JsonObjectConcept(
    name="Medical Assessment",
    description="Key information from a patient medical assessment",
    structure={
        "patient": {
            "id": str,
            "vital_signs": {
                "blood_pressure": str,
                "heart_rate": int,
                "temperature": float,
                "oxygen_saturation": int,
            },
        },
        "clinical": {
            "symptoms": list[str],
            "diagnosis": list[str],
            "travel_history": bool,
        },
        "treatment": {"recommendations": list[str], "follow_up_days": int},
    },
    # Examples provide helpful guidance on how to:
    # 1. Map data from unstructured text to structured fields
    # 2. Handle formatting variations (BP as "120/80" vs separate systolic/diastolic)
    # 3. Extract implicit information (converting "SpO2: 98%" to just 98)
    examples=[
        JsonObjectExample(
            content={
                "patient": {
                    "id": "87654",
                    "vital_signs": {
                        "blood_pressure": "130/85",
                        "heart_rate": 68,
                        "temperature": 98.2,
                        "oxygen_saturation": 99,
                    },
                },
                "clinical": {
                    "symptoms": ["headache", "dizziness", "nausea"],
                    "diagnosis": ["Migraine", "Dehydration"],
                    "travel_history": False,
                },
                "treatment": {
                    "recommendations": [
                        "Hydration",
                        "Pain medication",
                        "Dark room rest",
                    ],
                    "follow_up_days": 14,
                },
            }
        ),
        JsonObjectExample(
            content={
                "patient": {
                    "id": "23456",
                    "vital_signs": {
                        "blood_pressure": "145/92",
                        "heart_rate": 88,
                        "temperature": 100.8,
                        "oxygen_saturation": 96,
                    },
                },
                "clinical": {
                    "symptoms": ["sore throat", "cough", "fever"],
                    "diagnosis": ["Strep throat", "Pharyngitis"],
                    "travel_history": True,
                },
                "treatment": {
                    "recommendations": ["Antibiotics", "Throat lozenges", "Rest"],
                    "follow_up_days": 7,
                },
            }
        ),
    ],
)

# Attach the concept to the document
doc.add_concepts([medical_assessment_concept])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract the concept from the document
medical_assessment_concept = llm.extract_concepts_from_document(doc)[0]

# Print the extracted medical assessment
print("Extracted medical assessment:")
assessment = medical_assessment_concept.extracted_items[0].value
pprint(assessment)
