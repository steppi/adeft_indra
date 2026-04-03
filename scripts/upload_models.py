from pathlib import Path

from adeft import get_available_models
from adeft.disambiguate import AdeftDisambiguator
from adeft.locations import ADEFT_HOME
from adeft.modeling.classify import load_model_info

from adeft_indra.results import ResultsManager
from adeft_indra.s3 import model_to_s3
from adeft_indra.training import get_existing_grounding_info


results_manager = ResultsManager("model_retrain_2025-12-17.db")

old_models_path = Path(ADEFT_HOME) / "0.13.0"

for model_name, model_info in results_manager.items():
    if model_info is None:
        continue
    model = load_model_info(model_info)
    grounding_dict, names, _ = get_existing_grounding_info(
        model.shortforms[0], path=old_models_path
    )
    disamb = AdeftDisambiguator(model, grounding_dict, names)
    model_to_s3(disamb)
    
    
