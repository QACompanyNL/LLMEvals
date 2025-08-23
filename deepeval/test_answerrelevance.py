import os
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Define the prompt
prompt = "Explain the concept of quantum entanglement in simple terms."

# Generate text using GPT-3.5
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=150
)

generated_text = response.choices[0].message.content

golden_actual_text = """
Quantum entanglement is a phenomenon in quantum physics where two or more particles become linked, 
such that the state of one particle instantly influences the state of the other, no matter how far apart they are. 
This connection persists even if the particles are separated by large distances.
"""

# Define the test case
test_case = LLMTestCase(
    input=prompt,
    actual_output=generated_text,
    expected_output=golden_actual_text
)

relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model="gpt-4o")
results = evaluate([test_case], metrics=[AnswerRelevancyMetric()])

print(results)