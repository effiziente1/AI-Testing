import os
from dotenv import load_dotenv
from getpass import getpass
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval import assert_test

load_dotenv()

# Set up OpenAI API Key (Required)
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")


# Determines whether an LLM output is factually correct based on some ground truth.
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT,
                       LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.8
)

# actual_output = "We offer a 30-day full refund at no extra cost."
actual_output = "You should pay for another shoes"
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace this with the actual output from your LLM application
    actual_output=actual_output,
    expected_output="You are eligible for a 30 day full refund at no extra cost."
)

score = correctness_metric.measure(test_case)
# Print the score and associated reason
print(f"Correctness score: {score}")
print(f"Reason: {correctness_metric.reason}")
