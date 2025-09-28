from bot.model.base_model import ModelSettings

class OpenAI20BModelSettings(ModelSettings):
    
    
    file_name = "openai/gpt-oss-20b"  # L'ID du modèle fourni par Groq API
    url = "https://console.groq.com/docs/model/openai/gpt-oss-20b"  # Pas nécessaire pour Groq, on utilise l'API
    config = {
        "chat_format": "chatml-function-calling",  # similaire à LLaMA
        # Tu peux mettre ici d'autres configs si nécessaire, sinon laisser vide
    }
    config_answer = {
        "temperature": 0.7,
        "stop": [],  # Groq supporte aussi stop sequences
    }

# Si tu veux plusieurs variantes
class OpenAI20BModelSettingsHighTemp(OpenAI20BModelSettings):
    config = {
        "chat_format": "chatml-function-calling",  # similaire à LLaMA
        # Tu peux mettre ici d'autres configs si nécessaire, sinon laisser vide
    }
    config_answer = {
        "temperature": 1.0,
        "stop": [],
    }

class OpenAI20BModelSettingsLowTemp(OpenAI20BModelSettings):
    config = {
        "chat_format": "chatml-function-calling",  # similaire à LLaMA
        # Tu peux mettre ici d'autres configs si nécessaire, sinon laisser vide
    }
    config_answer = {
        "temperature": 0.3,
        "stop": [],
    }


