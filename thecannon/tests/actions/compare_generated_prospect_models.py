from thecannon.model import CannonModel

orig = "thecannon/tests/data/prospect_model_ref.model"
new = "prospect_model_test.model"

old_model = CannonModel.read(orig)
new_model = CannonModel.read(new)

assert old_model == new_model, "Model mismatch!"
