import json
import pennylane as qml
import pennylane.numpy as np


def circuit(oracle):
    """The circuit to find if the Bitland Kingdom was in danger.
    You can query the oracle by calling oracle(wires=[0, 1, 2])."""

    # Put your code here #
    """
    idea is: create bell state, state 00 and 11 are opposite villages if they same sign we either get global phase,
     that dont influence probs or nothing will change when we return to original state.
     If both will have opposite sign, bell state will catch sign and 
     state 00 will change to 10 after doing inverse operations. 
     Changed state means that those villages had same sign neighbour and making us rotate last bit to state 1
    """

    #create bell state |+> on first 2 wires
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    # create |+> state on last wire
    # states 000 001 110 111 exist
    qml.Hadamard(2)
    oracle(wires=[0, 1, 2])
    # retrun back doing inverse
    qml.Hadamard(2)
    qml.CNOT([0, 1])
    qml.Hadamard(0)
    # if 0th bit in state 1 we will rotate last bit
    qml.CNOT([0, 2])


# These functions are responsible for testing the solution.

def oracle_maker():
    # States order: |0> , |1>, -|0>, -|1>

    indx = [[0, 0], [1, 0], [1, 1], [0, 1]]

    # Village 00 -> |0>
    # Village 10 -> |1>
    # Village 11 -> -|0>
    # Village 01 -> -|1>

    np.random.shuffle(indx)

    indices_00 = [index for index, value in enumerate(indx) if value == [0, 0]]
    indices_11 = [index for index, value in enumerate(indx) if value == [1, 1]]

    if set([indices_00[0], indices_11[0]]) == set([0, 1]) or set([indices_00[0], indices_11[0]]) == set([2, 3]):
        target = 0
    else:
        target = 1

    def oracle(wires):

        class op(qml.operation.Operation):
            num_wires = 3
            grad_method = None

            def __init__(self, wires, id=None):
                super().__init__(wires=wires, id=id)

            @property
            def num_params(self):
                return 0

            @staticmethod
            def compute_decomposition(wires):
                wires_input = wires[:2]
                wire_output = wires[2]

                ops = []
                ops.append(qml.ctrl(qml.PauliX(wires=wire_output), control=wires_input, control_values=indx[1]))

                ops.append(
                    qml.ctrl(qml.GlobalPhase(np.pi, wires=wire_output), control=wires_input, control_values=indx[2]))

                ops.append(qml.ctrl(qml.PauliX(wires=wire_output), control=wires_input, control_values=indx[3]))
                ops.append(
                    qml.ctrl(qml.GlobalPhase(np.pi, wires=wire_output), control=wires_input, control_values=indx[3]))

                return ops

        return op(wires=wires)

    return oracle, target


def run(case: str) -> str:
    return "No output"


def check(have: str, want: str) -> None:
    for _ in range(100):
        oracle, target = oracle_maker()

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def test_circuit():
            circuit(oracle)
            return qml.probs(wires=2)

        test_circuit()
        assert [op.name for op in test_circuit.tape.operations].count("op") == 1, "You can use the oracle once."

        assert np.isclose(test_circuit()[1], target), "Wrong answer!"


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