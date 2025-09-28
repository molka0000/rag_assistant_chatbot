from bot.model.base_model import ModelSettings


class OpenAI120BModelSettings(ModelSettings):
    
    
    file_name = "openai/gpt-oss-120b"  # L'ID du modèle fourni par Groq API
    url = "https://console.groq.com/docs/model/openai/openai/gpt-oss-120b"  # Pas nécessaire pour Groq, on utilise l'API
    config = {
        "chat_format": "chatml-function-calling",  # similaire à LLaMA
        # Tu peux mettre ici d'autres configs si nécessaire, sinon laisser vide
    }
    config_answer = {
        "temperature": 0.7,
        "stop": [],  # Groq supporte aussi stop sequences
    }

# Si tu veux plusieurs variantes
class OpenAI120BModelSettingsHighTemp(OpenAI120BModelSettings):
    config = {
        "chat_format": "chatml-function-calling",  # similaire à LLaMA
        # Tu peux mettre ici d'autres configs si nécessaire, sinon laisser vide
    }
    config_answer = {
        "temperature": 1.0,
        "stop": [],
    }

class OpenAI120BModelSettingsLowTemp(OpenAI120BModelSettings):
    config = {
        "chat_format": "chatml-function-calling",  # similaire à LLaMA
        # Tu peux mettre ici d'autres configs si nécessaire, sinon laisser vide
    }
    config_answer = {
        "temperature": 0.3,
        "stop": [],
    }


