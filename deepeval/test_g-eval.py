from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval.dataset import EvaluationDataset, Golden

# Stap 1: Maak goldens
golden_actual_text = """
Quantum entanglement is a phenomenon in quantum physics where two or more particles become linked, 
such that the state of one particle instantly influences the state of the other, no matter how far apart they are. 
This connection persists even if the particles are separated by large distances.
"""

goldens = [
    Golden(
        input="Explain the concept of quantum entanglement in simple terms.",
        expected_output=golden_actual_text
    )
]

# Stap 2: Bouw de dataset
dataset = EvaluationDataset(goldens=goldens)

# Stap 3: Genereer testcases en voeg toe aan dataset
test_cases = [
    LLMTestCase(
        input=goldens[0].input,
        actual_output="Quantum entanglement happens when two particles stay connected, so a change in one affects the other instantly, even miles apart.",
        expected_output=goldens
    ),
    LLMTestCase(
        input=goldens[0].input,
        actual_output="Quantum entanglement is when two particles always have the same properties, even at a distance, due to hidden variables in physics.",
        expected_output=goldens[0].expected_output
    ),
    LLMTestCase(
        input=goldens[0].input,
        actual_output="Quantum entanglement is the process by which atoms travel faster than light and communicate instantly across space.",
        expected_output=goldens[0].expected_output
    )
]

dataset.test_cases.extend(test_cases)

# Definieer metric
correctness_metric = GEval(
    criteria="correctness",
    name="Correctness",
    model="gpt-4o",
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ]
)

# Run evaluatie
evaluation_output = dataset.evaluate([correctness_metric])
print(evaluation_output)
