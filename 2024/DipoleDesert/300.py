import json
import pennylane as qml
import pennylane.numpy as np


# Write any helper functions you need here


def GHZ_circuit(noise_param, n_qubits):
    """
    Quantum circuit that prepares an imperfect GHZ state using gates native to a neutral atom device.

    Args:
        - noise_param (float): Parameter that quantifies the noise in the CZ gate, modelled as a
        depolarizing channel on the target qubit. noise_param is the parameter of the depolarizing channel
        following the PennyLane convention.
        - n_qubits (int): The number of qubits in the prepared GHZ state.
    Returns:
        - (np.tensor): A density matrix, as returned by `qml.state`, representing the imperfect GHZ state.

    """

    # Put your code here
    phi, theta = np.pi / 2, np.pi
    # RY with theta = pi/2 acting on state |0> does same job as hadamard
    qml.RY(phi, wires=0)
    for i in range(n_qubits - 1):
        # RY on pi/2 and RX on pi gives -iH
        # for CNOT, X can be replaced with HZH
        qml.RY(phi, wires=i + 1)
        qml.RX(theta, i + 1)
        qml.CZ([i, i + 1])
        qml.DepolarizingChannel(noise_param, wires=i + 1)

    qml.RY(phi, wires=n_qubits - 1)
    qml.RX(theta, n_qubits - 1)

    return qml.state()


def GHZ_fidelity(noise_param, n_qubits):
    """
    Calculates the fidelity between the imperfect GHZ state returned by GHZ_circuit and the ideal GHZ state.

    Args:
        - noise_param (float): Parameter that quantifies the noise in the CZ gate, modelled as a
        depolarizing channel on the target qubit. noise_param is the parameter of the depolarizing channel
        following the PennyLane convention.
        - n_qubits (int): The number of qubits in the GHZ state.
    Returns:
        - (float): The fidelity between the noisy and ideal GHZ states.
    """

    dev = qml.device('default.mixed', wires=n_qubits)

    GHZ_QNode = qml.QNode(GHZ_circuit, dev)

    # Use GHZ_QNode to find the fidelity between
    # the noisy GHZ state and an ideal GHZ state
    GHZ_ideal = GHZ_QNode(0, n_qubits)
    GHZ_noisy = GHZ_QNode(noise_param, n_qubits)
    res = qml.math.fidelity(GHZ_ideal, GHZ_noisy)
    return res


# These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    output = GHZ_fidelity(*ins)

    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)

    dev = qml.device('default.mixed', wires=4)
    qnode = qml.QNode(GHZ_circuit, dev)
    u = qnode(0.05, 3)

    for op in qnode.tape.operations:
        assert (isinstance(op, qml.RX) or isinstance(op, qml.RY) or isinstance(op, qml.CZ) or isinstance(op,
                                                                                                         qml.DepolarizingChannel)), "You are using forbidden gates!"

    assert np.isclose(solution_output, expected_output, rtol=1e-4)


# These are the public test cases
test_cases = [
    ('[0.05, 3]', '0.9027779255467782'),
    ('[0.01, 5]', '0.9606614879634601')
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