import numpy as np
from perceptron import Perceptron

# Data for the logic gates and the robot
AND_GATE_DATA = {
    "inputs": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    "outputs": np.array([[0], [0], [0], [1]])
}

OR_GATE_DATA = {
    "inputs": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    "outputs": np.array([[0], [1], [1], [1]])
}

XOR_GATE_DATA = {
    "inputs": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    "outputs": np.array([[0], [1], [1], [0]])
}

ROBOT_DATA = {
    "inputs": np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], 
                        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]),
    "outputs": np.array([[1, 1], [0, 1], [1, 0], [0, 1],
                         [1, 0], [1, 0], [1, 0], [1, 0]])
}

def train_and_test(test_name, data, num_inputs, num_outputs):
    print(f"### Training and Testing: {test_name} ###")
    p = Perceptron(num_inputs, num_outputs)

    # Train the network for 10,000 epochs
    for epoch in range(10000):
        epoch_error = 0
        for i in range(len(data["inputs"])):
            inputs = data["inputs"][i]
            expected_output = data["outputs"][i]
            calculated_output = p.train(inputs, expected_output)
            epoch_error += np.sum(np.abs(expected_output - calculated_output))
        
        if (epoch + 1) % 1000 == 0:
             print(f"Epoch {(epoch + 1)}, approximation error: {epoch_error}")

    # Test the network after training
    print(f"\nResults after training for {test_name}:")
    for i in range(len(data["inputs"])):
        inputs = data["inputs"][i]
        result = p.execute(inputs)
        print(f"Input: {inputs} Output: {np.round(result, 3)} Expected: {data['outputs'][i]}")
    print("########################################\n")


if __name__ == "__main__":
    train_and_test("AND Logic Gate", AND_GATE_DATA, 2, 1)
    train_and_test("OR Logic Gate", OR_GATE_DATA, 2, 1)
    train_and_test("XOR Logic Gate", XOR_GATE_DATA, 2, 1)
    train_and_test("Robot", ROBOT_DATA, 3, 2)