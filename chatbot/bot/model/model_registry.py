from enum import Enum
from bot.model.settings.groq import GroqModelSettings
from bot.model.settings.llama8 import llama8ModelSettings, llama8ModelSettingsHighTemp, llama8ModelSettingsLowTemp
from bot.model.settings.open_ai_20b import OpenAI20BModelSettings, OpenAI20BModelSettingsHighTemp, OpenAI20BModelSettingsLowTemp
from bot.model.settings.groq import GroqModelSettingsHighTemp, GroqModelSettingsLowTemp
from bot.model.settings.open_ai_120b import OpenAI120BModelSettings, OpenAI120BModelSettingsHighTemp, OpenAI120BModelSettingsLowTemp


class Model(Enum):
    GROQ_MODEL_llama8 ="llama-3.1-8b-instant"
    GROQ_MODEL = "llama-3.3-70b-versatile"
    OPENAI_20B = "openai/gpt-oss-20b"
    OPENAI_120B = "openai/gpt-oss-120b"
   


SUPPORTED_MODELS = {
    Model.GROQ_MODEL.value: GroqModelSettings,
   # Model.GROQ_MODEL.value : GroqModelSettingsHighTemp,
   # Model.GROQ_MODEL.value : GroqModelSettingsLowTemp,
    Model.OPENAI_20B.value: OpenAI20BModelSettings,
   # Model.OPENAI_20B.value: OpenAI20BModelSettingsHighTemp,
   # Model.OPENAI_20B.value: OpenAI20BModelSettingsLowTemp,
    Model.OPENAI_120B.value: OpenAI120BModelSettings,
   # Model.OPENAI_120B.value: OpenAI120BModelSettingsHighTemp,
  #  Model.OPENAI_120B.value: OpenAI120BModelSettingsLowTemp,
    Model.GROQ_MODEL_llama8.value: llama8ModelSettings,
   # Model.GROQ_MODEL_llama8.value: llama8ModelSettingsHighTemp,
   # Model.GROQ_MODEL_llama8.value: llama8ModelSettingsLowTemp,
   
}


def get_models():
    return list(SUPPORTED_MODELS.keys())


def get_model_settings(model_name: str):
    model_settings = SUPPORTED_MODELS.get(model_name)

    # validate input
    if model_settings is None:
        raise KeyError(model_name + " is a not supported model")

    return model_settings
