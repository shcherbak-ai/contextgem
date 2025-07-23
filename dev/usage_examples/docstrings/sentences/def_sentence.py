from contextgem import Sentence


# Create a sentence with raw text content
sentence = Sentence(raw_text="This is a simple sentence.")

# Sentences are immutable - their content cannot be changed after creation
try:
    sentence.raw_text = "Attempting to modify the sentence."
except ValueError as e:
    print(f"Error when trying to modify sentence: {e}")
