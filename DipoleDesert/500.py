import functools
import json
import pennylane as qml
import pennylane.numpy as np

stack_last = functools.partial(qml.math.stack, axis=-1)

Operation = qml.operation.Operation


class fSim(Operation):
    num_params = 2
    num_wires = 2
    par_domain = "R"

    ndim_params = (0, 0)

    # grad_method = "A"
    # parameter_frequencies = [(1,), (1,)]

    def __init__(self, theta, phi, wires, id=None):
        super().__init__(theta, phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(theta, phi):
        c = qml.math.cos(theta)
        s = qml.math.sin(theta)

        one = qml.math.ones_like(phi)
        c = c * one
        s = s * one

        mat = [[1, 0, 0, 0],
               [0, c, -1j * s, 0], [0, -1j * s, c, 0], [0, 0, 0, qml.math.exp(-1j * phi)]]

        return qml.math.stack([stack_last(row) for row in mat], axis=-2)


class SQRTX(Operation):
    num_params = 0
    num_wires = 1

    def __init__(self, wires, id=None):
        super().__init__(wires=wires, id=id)

    @staticmethod
    def compute_decomposition(wires):
        return [qml.RX(np.pi / 2, wires=wires)]


class SQRTY(Operation):
    num_params = 0
    num_wires = 1

    def __init__(self, wires, id=None):
        super().__init__(wires=wires, id=id)

    @staticmethod
    def compute_decomposition(wires):
        return [qml.RY(np.pi / 2, wires=wires)]


class Wormhole(Operation):
    num_params = 1
    num_wires = 4

    def __init__(self, g, wires, id=None):
        super().__init__(g, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(g, wires):
        return [qml.IsingZZ(-g, wires=[wires[0], wires[3]]), qml.IsingZZ(-g, wires=[wires[1], wires[2]])]


# Write here any helper functions you need



def negative_i_hadamard(wire):
    """function that creates -iH, to get it we do RY with angle pi/2 and RX with angle pi"""
    SQRTY(wire)
    SQRTX(wire)
    SQRTX(wire)

def CZ(wires):
    fSim(0, np.pi, wires)

def bell(wires: list):
    """function that makes bell state with 2 wires, remember X = HZH"""
    SQRTY(wires[0])
    negative_i_hadamard(wires[1])
    CZ(wires)
    negative_i_hadamard(wires[1])


def swap(wires: list):
    """swap as a sequence of 3 CNOT, remember X = HZH"""
    negative_i_hadamard(wires[1])
    CZ(wires)
    negative_i_hadamard(wires[1])

    negative_i_hadamard(wires[0])
    CZ(wires[::-1])
    negative_i_hadamard(wires[0])

    negative_i_hadamard(wires[1])
    CZ(wires)
    negative_i_hadamard(wires[1])


def U(wires):
    for i in range(2):
        negative_i_hadamard(wires[i+1])
        fSim(0, np.pi, [wires[i], wires[i+1]])
        SQRTX(wires[i+1])
        fSim(0, np.pi, [wires[i], wires[i+1]])
        negative_i_hadamard(wires[i+1])

    for i in range(3):
        SQRTX(wires[i])


dev = qml.device('default.qubit', wires=range(7))


@qml.qnode(dev)
def wormhole_teleportation(g):
    """ This function implements the wormhole-inspired teleporation protocol
    using gates native to some superconducting devices.

    Args:
        g (float): Parameter for the Wormhole gate as shown in the circuit.

    Returns:
        (float): The expectation value on the sixth wire.
    """

    # Put your code here
    # make initial state as combination of 3 bells
    bell([2, 3])
    bell([1, 4])
    bell([0, 5])

    # U inverse is just two more RX rotation on pi/2 (also could be realised with inverse RX)
    # n for number of times = 3
    n = 3
    for _ in range(n):
        U([0, 1, 2])
        for i in range(3):
            for __ in range(2):
                SQRTX(i)

    swap([0, 6])

    for i in range(n):
        U([0, 1, 2])

    Wormhole(g, [1, 2, 3, 4])

    for i in range(n):
        U([3, 4, 5])

    return qml.expval(qml.PauliZ(5))


# These functions are responsible for testing the solution.


def run(test_case_input: str) -> str:
    ins = np.array(json.loads(test_case_input))
    outs = wormhole_teleportation(ins)

    return str(outs)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)

    u = wormhole_teleportation(np.pi / 4)
    tape = wormhole_teleportation.qtape
    names = [op.name for op in tape.operations]

    assert names.count('Wormhole') == 1, "Can't use Wormhole gate more than once"
    for op in tape.operations:
        assert (isinstance(op, SQRTX) or isinstance(op, SQRTY) or isinstance(op, fSim) or isinstance(op,
                                                                                                     Wormhole)), "You can only use SQRTX, SQRTY, fSim, and Wormhole gates"

    assert np.isclose(solution_output, expected_output, rtol=1e-4), "Not the correct expectation value"


# These are the public test cases
test_cases = [
    ('1.25663706', '-0.9045085'),
    ('1.5707963', '-1.000')
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