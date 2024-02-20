import json
import pennylane as qml
import pennylane.numpy as np
"""
not solved yet
"""
symbols = ["H", "H", "H"]


def h3_ground_energy(bond_length):
    """
    Uses VQE to calculate the ground energy of the H3+ molecule with the given bond length.

    Args:
        - bond_length(float): The bond length of the H3+ molecule modelled as an
        equilateral triangle.
    Returns:
        - Union[float, np.tensor, np.array]: A float-like output containing the ground
        state of the H3+ molecule with the given bond length.
    """


# These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    outs = h3_ground_energy(ins)
    return str(outs)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(solution_output, expected_output, atol=1e-4), "Not the correct ground energy"


# These are the public test cases
test_cases = [
    ('1.5', '-1.232574'),
    ('0.8', '-0.3770325')
]

# This will run the public test cases locally
for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")