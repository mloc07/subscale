"""Check f5_tts infer API signature."""

import inspect
from f5_tts.infer import utils_infer

print("load_vocoder:", inspect.signature(utils_infer.load_vocoder))
print("load_model:", inspect.signature(utils_infer.load_model))
print("infer_process:", inspect.signature(utils_infer.infer_process))
