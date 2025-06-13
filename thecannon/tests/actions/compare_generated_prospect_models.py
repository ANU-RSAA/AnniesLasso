from thecannon.model import CannonModel
import numpy as np

orig = "thecannon/tests/data/prospect_model_ref.model"
new = "prospect_model_test.model"

old_model = CannonModel.read(orig)
new_model = CannonModel.read(new)

assert old_model == new_model, "Model mismatch!"

orig_labels = "thecannon/tests/data/prospect_model_labels_ref.npy"
new_labels = "prospect_model_labels.npy"

orig_labels_arr = np.load(orig_labels)
new_labels_arr = np.load(new_labels)

try:
    assert np.allclose(orig_labels_arr, new_labels_arr, atol=0, rtol=1e-5)
except AssertionError:
    labels_delta = new_labels_arr - orig_labels_arr
    raise AssertionError(
        f"Test labels mismatch: delta stats mean={np.mean(np.abs(labels_delta))}, "
        f"median={np.median(labels_delta)}, std={np.std(labels_delta)}, "
        f"max={np.max(labels_delta)} @ {(orig_labels_arr.flatten())[np.argmax(labels_delta)]:.9e}, {(new_labels_arr.flatten())[np.argmax(labels_delta)]:.9e}"
    )


orig_cov = "thecannon/tests/data/prospect_model_cov_ref.npy"
new_cov = "prospect_model_cov.npy"

orig_cov_arr = np.load(orig_labels)
new_cov_arr = np.load(new_labels)
try:
    assert np.allclose(orig_cov_arr, new_cov_arr, atol=0, rtol=1e-5)
except AssertionError:
    cov_delta = new_cov_arr - orig_cov_arr
    raise AssertionError(
        f"Test cov mismatch: delta stats: (abs)mean={np.mean(np.abs(cov_delta))}, "
        f"median={np.median(cov_delta)}, std={np.std(cov_delta)}, "
        f"max={np.abs(cov_delta)} @ {(orig_cov.flatten)[np.argmax(cov_delta)]:.9e}, {(new_cov.flatten())[np.argmax(cov_delta)]:.9e}"
    )
