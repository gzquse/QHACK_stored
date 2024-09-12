import json
import pennylane as qml
import pennylane.numpy as np

# Write any helper functions you need here


dev = qml.device('default.qubit', wires=[0, 1, 2])


@qml.qnode(dev)
def cloning_machine(coefficients, wire):
    """
    Returns the reduced density matrix on a wire for the cloning machine circuit.

    Args:
        - coefficients (np.array(float)): an array [c0,c1] containing the coefficients parametrizing
        the input state fed into the middle and bottom wires of the cloning machine.
        wire (int): The wire on which we calculate the reduced density matrix.

    Returns:
        - np.tensor(complex): The reduced density matrix on wire = wire, as returned by qml.density_matrix.

    """

    # Put your code here
    coefficients = coefficients / np.sqrt(2)
    c0 = coefficients[0]
    c1 = coefficients[1]
    theta1 = 2 * np.arcsin(c0)
    theta2 = np.arcsin(c1 / (np.cos(theta1 / 2)))

    # circuit can create 3 states 00 01 and 11,
    # doing those operations we find evaluations for the states weights
    # and after angles of the rotation
    qml.RY(theta1, 1)
    qml.RY(theta2, 2)
    qml.CNOT([1, 2])
    qml.RY(theta2, 2)

    # clone state
    qml.CNOT([0, 1])
    qml.CNOT([0, 2])

    qml.CNOT([1, 0])
    qml.CNOT([2, 0])

    return qml.density_matrix(wire)

    # Return the reduced density matrix


def fidelity(coefficients):
    """
    Calculates the fidelities between the reduced density matrices in wires 0 and 1 and the input state |0>.

    Args:
        - coefficients (np.array(float)): an array [c0,c1] containing the coefficients parametrizing
        the input state fed into the middle and bottom wires of the cloning machine.
    Returns:
        - (np.array(float)): An array whose elements are:
            - 0th element:  The fidelity between the output reduced state on wire 0 and the state |0>.
            - 1st element:  The fidelity between the output reduced state on wire 1 and the state |0>.
    """

    # Put your code here
    # find density matrix at wire 0 and 1
    Wire0 = cloning_machine(coefficients, wire=0)
    Wire1 = cloning_machine(coefficients, wire=1)
    # density matrix for state 0
    state0 = np.array([[1.0, 0.0], [0.0, 0.0]])
    # find fidelity beetween wire 0 and 1 and state 1
    res = []
    res.append(qml.math.fidelity(Wire0, state0))
    res.append(qml.math.fidelity(Wire1, state0))
    return np.array(res)


# These functions are responsible for testing the solution.


def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    outs = fidelity(ins).tolist()

    return str(outs)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    u = cloning_machine([1 / np.sqrt(3), 1 / np.sqrt(3)], 1)
    for op in cloning_machine.tape.operations:
        assert (isinstance(op, qml.RX) or isinstance(op, qml.RY) or isinstance(op,
                                                                               qml.CNOT)), "You are using forbidden gates!"
    assert np.allclose(solution_output, expected_output, atol=1e-4), "Not the correct fidelities"


# These are the public test cases
test_cases = [
    ('[0.5773502691896258, 0.5773502691896257]', '[0.83333333, 0.83333333]'),
    ('[0.2, 0.8848857801796105]', '[0.60848858, 0.98]')
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