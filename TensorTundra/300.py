import json
import pennylane as qml
import pennylane.numpy as np
"""
not solved yet
"""

class AbsMagnetization(qml.measurements.StateMeasurement):
    """A measurement class that estimates <|M|>."""

    def process_state(self, state, wire_order):
        """Calculates <|M|>.

        Args:
            state (Sequence[complex]): quantum state with a flat shape. It may also have an
                optional batch dimension.

            wire_order (Wires): wires determining the subspace that the state acts on; a matrix of
                dimension 2**n that acts on a subspace of n wires

        Returns:
            abs_mag (float): <|M|>


        See the docs for more information:
        https://docs.pennylane.ai/en/stable/code/api/pennylane.measurements.StateMeasurement.html
        """

        state = qml.state().process_state(state, wire_order)

        # Put your code here #

        return  # return <|M|>


def tfim_ground_state(num_qubits, h):
    """Calculates the ground state of the 1D TFIM Hamiltonian.

    Args:
        num_qubits (int): The number of qubits / spins.
        h (float): The transverse field strength.

    Returns:
        (numpy.tensor): The ground state.
    """

    # Put your code here #

    return  # return the ground state of the 1D TFIM Hamiltonian


dev = qml.device("default.qubit")


@qml.qnode(dev)
def magnetization(num_qubits, h):
    """Calculates the absolute value of the magnetization of the 1D TFIM
    Hamiltonian.

    Args:
        num_qubits (int): The number of qubits / spins.
        h (float): The transverse field strength.

    Returns:
        (float): <|M|>.
    """

    # Put your code here #

    return AbsMagnetization(wires=list(range(num_qubits)))


def critical_point_estimate(mags, h_values):
    """Provides a finite-size estimate of the critical point of the 1D TFIM
    Hamiltonian. The estimate is done by taking the average value of h for which
    adjacent values of <|M|> differ the most.

    Args:
        mags (numpy.tensor):
            <|M|> values for various values of h (the transverse field strength).
        h_values (numpy.tensor): The transverse field strength values.

    Returns:
        (float): The critical point estimate, h_c.
    """
    differences = [np.abs(mags[i] - mags[i + 1]) for i in range(len(mags) - 1)]
    ind = np.argmax(np.array(differences))

    h_c = np.mean([h_values[ind], h_values[ind + 1]])
    return h_c


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    num_qubits = json.loads(test_case_input)
    h_values = np.arange(0.2, 1.1, 0.005)
    mags = []

    for h in h_values:
        mags.append(magnetization(num_qubits, h) / num_qubits)

    output = critical_point_estimate(np.array(mags), h_values)

    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)

    assert np.isclose(solution_output, expected_output, rtol=5e-3)


# These are the public test cases
test_cases = [
    ('5', '0.6735'),
    ('2', '0.3535')
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