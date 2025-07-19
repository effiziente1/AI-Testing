from transformers import pipeline

# Load NER pipeline
ner_pipeline = pipeline(
    'ner',
    aggregation_strategy="simple",  # Use this instead of grouped_entities=True
    model="dslim/bert-base-NER"
)

# Your sample text
text = "I am Abigail and I like to write articles related to QA on substack and linkedin. I live in Mexico."

# Run NER
result = ner_pipeline(text)

# Filter out low-confidence predictions (score < 0.9)
high_confidence_entities = [
    entity for entity in result if entity['score'] >= 0.9]

print("All entities:")
for entity in result:
    print(
        f"  {entity['word']} -> {entity['entity_group']} (confidence: {entity['score']:.3f})")

print("\nHigh-confidence entities only:")
for entity in high_confidence_entities:
    print(
        f"  {entity['word']} -> {entity['entity_group']} (confidence: {entity['score']:.3f})")
