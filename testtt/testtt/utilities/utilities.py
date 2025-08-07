import numpy as np

def test_function():
    # Example test function that uses numpy
    arr = np.array([1, 2, 3])
    assert np.sum(arr) == 6, "Sum of array should be 6"
    print("Test passed!")



if __name__ == "__main__":
    test_function()