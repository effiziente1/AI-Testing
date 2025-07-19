from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import pipeline
import warnings

# Sample text to summarize
text = """
Testing a WebSocket app requires a different approach than web apps, some of the challenges are:

Connectivity: You can check that clients can connect without any problem and the connection is closed when the user exit the app or the app can reconnect after some internet connection issues
Message exchange: Check that the messages contains all the information and the information is correct for example on rideshare apps you need to test that all the status are correct.
Performance: You need to check if the WebSocket allows the expected number of the users, you can use the performance tools like K6, JMeter, Gatling or Artillery.
Security: You need to check unauthorized access, if are encrypted (check that is using wss://, cross site scripting, denial of service (DoS).
"""

print("=== Summarization Examples ===\n")

# Option 1: BART model (most reliable, no warnings)
print("1. Using BART model:")
summarizer_bart = pipeline("summarization", model="facebook/bart-large-cnn")
summary_bart = summarizer_bart(
    text, max_length=50, min_length=20, do_sample=False)
print(f"Summary: {summary_bart[0]['summary_text']}\n")

# Option 2: Direct model usage (no pipeline warnings)
print("3. Using T5 directly (no warnings):")

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Prepare input
inputs = tokenizer("summarize: " + text, return_tensors="pt",
                   max_length=50, truncation=True)

# Generate summary
summary_ids = model.generate(
    inputs["input_ids"], max_new_tokens=30, min_length=10, do_sample=False)
summary_direct = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(f"Summary: {summary_direct}")
