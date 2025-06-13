from thecannon.model import CannonModel
import numpy as np

orig = "thecannon/tests/data/prospect_model_restricted_ref.model"
new = "prospect_model_restricted_test.model"

old_model = CannonModel.read(orig)
new_model = CannonModel.read(new)

assert old_model == new_model, "Model mismatch!"

orig_labels = "thecannon/tests/data/prospect_model_labels_restricted_ref.npy"
new_labels = "prospect_model_labels_restricted.npy"

assert np.allclose(np.load(orig_labels), np.load(new_labels)), "Test labels mismatch!"


orig_cov = "thecannon/tests/data/prospect_model_cov_restricted_ref.npy"
new_cov = "prospect_model_cov_restricted.npy"

assert np.allclose(np.load(orig_cov), np.load(new_cov)), "Covariance mismatch!"

