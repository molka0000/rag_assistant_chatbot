from bot.model.base_model import ModelSettings

class GroqModelSettings(ModelSettings):
    """
    Paramètres de configuration pour le modèle Groq.
    """
    file_name = "llama-3.3-70b-versatile"  # L'ID du modèle fourni par Groq API
    url = "https://console.groq.com/docs/model/llama-3.3-70b-versatile"  # Pas nécessaire pour Groq, on utilise l'API
    config = {
        "chat_format": "chatml-function-calling",  # similaire à LLaMA
        # Tu peux mettre ici d'autres configs si nécessaire, sinon laisser vide
    }
    config_answer = {
        "temperature": 0.3,
        "stop": [],  # Groq supporte aussi stop sequences
    }


# Si tu veux plusieurs variantes
class GroqModelSettingsHighTemp(GroqModelSettings):
    config = {
        "chat_format": "chatml-function-calling",  # similaire à LLaMA
        # Tu peux mettre ici d'autres configs si nécessaire, sinon laisser vide
    }
    config_answer = {
        "temperature": 1.0,
        "stop": [],
    }

class GroqModelSettingsLowTemp(GroqModelSettings):
    config = {
        "chat_format": "chatml-function-calling",  # similaire à LLaMA
        # Tu peux mettre ici d'autres configs si nécessaire, sinon laisser vide
    }
    config_answer = {
        "temperature": 0.3,
        "stop": [],
    }


