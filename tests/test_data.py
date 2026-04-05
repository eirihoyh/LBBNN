import numpy as np
from LBBNN import create_bsr_data, create_data_unif

def test_create_data_unif_shapes_and_labels():
    y, X = create_data_unif(20, classification=True, seed=1)
    assert X.shape == (20, 5)
    assert y.shape == (20,)
    assert set(np.unique(y)).issubset({0, 1})

def test_create_bsr_data_shapes():
    y, X = create_bsr_data(30, func=4, seed=2)
    assert X.shape == (30, 2)
    assert y.shape == (30,)
