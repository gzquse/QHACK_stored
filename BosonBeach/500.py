import json
import pennylane as qml
import pennylane.numpy as np
"""
not solved yet
"""

def U():
    """
    Quantum circuit that facilitates the distribution of goods.
    It should not return anything, you simply need to add the gates.
    """


    # Put your code here #



# These functions are responsible for testing the solution.


def run(test_case_input: str) -> str:
    return None


def check(have: str, want: str) -> None:
    have, want = have, want
    params = np.random.rand(10, 2)

    dev = qml.device("default.qubit", wires=13, shots=1000)

    def generate_phi(params, wires):
        for i in range(len(wires)):
            qml.RX(params[i][0], wires=wires[i])

        for i in range(len(wires) - 1):
            qml.CNOT(wires=[i, i + 1])

        for i in range(len(wires)):
            qml.RX(params[i][1], wires=wires[i])

    @qml.qnode(dev)
    def circuit():
        generate_phi(params, wires=range(10))
        U()
        return qml.sample(wires=range(10))

    shots = circuit()
    for shot in shots:
        assert sum(shot) % 3 == 0, "Wrong answer"

    for op in circuit.tape.operations:
        assert not isinstance(op, qml.QubitUnitary), "You can't use QubitUnitary"
        assert not isinstance(op, qml.measurements.mid_measure.MidMeasureMP), "You cannot use measurements"


# These are the public test cases
test_cases = [
    ('No input', 'No output')
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