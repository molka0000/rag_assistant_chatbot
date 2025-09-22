from bot.model.base_model import ModelSettings

class llama8ModelSettings(ModelSettings):
    """
    Paramètres de configuration pour le modèle llama8.
    """
    file_name = "llama-3.1-8b-instant"  # L'ID du modèle fourni par Groq API
    url = "https://console.groq.com/docs/model/llama-3.1-8b-instant"  # Pas nécessaire pour Groq, on utilise l'API
    config = {
    
        "chat_format": "chatml-function-calling",  # similaire à LLaMA
    }
    config_answer = {
        "temperature": 0.7,
        "stop": [],  # Groq supporte aussi stop sequences
    }

# Si tu veux plusieurs variantes
class llama8ModelSettingsHighTemp(llama8ModelSettings):
    config_answer = {
        "temperature": 1.0,
        "stop": [],
    }

class llama8ModelSettingsLowTemp(llama8ModelSettings):
    config_answer = {
        "temperature": 0.3,
        "stop": [],
    }
