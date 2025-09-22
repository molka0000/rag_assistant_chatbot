import json
import sys
from pathlib import Path
import requests

# Ajout du chemin pour les imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from bot.client.lama_cpp_client import LamaCppClient
from bot.model.model_registry import get_model_settings


def call_webstat_api(series_key: str, limit: int = 10) -> str:
    """
    Récupère des données depuis l'API Webstat Banque de France
    
    Args:
        series_key: Code de la série (ex: "MIR1.M.FR.B.L23FRLA.D.R.A.2230U6.EUR.O")
        limit: Nombre de résultats maximum
    """
    base_url = "https://webstat.banque-france.fr/api/explore/v2.1/catalog/datasets/observations/exports/json/"
    api_key = "0ce5738ef88272bdce5af4a93e146fd3c4a1b0daa08e2c6016965ecd"
    
    params = {
        "where": f'series_key IN ("{series_key}")',
        "order_by": "-time_period_start",
        "limit": limit
    }
    
    headers = {
        "Authorization": f"Apikey {api_key}",
        "User-Agent": "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(base_url, params=params, headers=headers)
        print(f"URL appelée: {response.url}")
        print(f"Status code: {response.status_code}")
        response.raise_for_status()
        return json.dumps(response.json(), ensure_ascii=False, indent=2)
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Erreur API Webstat: {str(e)}", "status_code": getattr(e.response, 'status_code', 'N/A') if hasattr(e, 'response') else 'N/A'})


# Séries importantes prédéfinies
WEBSTAT_SERIES = {
    "taux_livret_a": "MIR1.M.FR.B.L23FRLA.D.R.A.2230U6.EUR.O",
    "taux_ldds": "MIR1.M.FR.B.L23FRLD.D.R.A.2254U6.EUR.O",
    "taux_lep": "MIR1.M.FR.B.L23FRLP.H.R.A.2250U6.EUR.O",
    "taux_pel": "MIR1.M.FR.B.L22FRPL.A.T.A.2250U6.EUR.N",
    "taux_refinancement_principale_bce" :"FM.M.U2.EUR.4F.KR.MRR_FR.LEV",
    "taux_de_depot_bce" :"FM.M.U2.EUR.4F.KR.DF.LEV",
    "taux_euribor_3mois" :"ECOFI.INR.FR.FID_PA._Z.D"
}

def get_economic_indicator(indicator_name: str, limit: int = 5) -> str:
    """
    Récupère un indicateur économique par nom simplifié
    """
    series_key = WEBSTAT_SERIES.get(indicator_name.lower())
    if not series_key:
        return json.dumps({"error": f"Indicateur '{indicator_name}' non trouvé. Disponibles: {list(WEBSTAT_SERIES.keys())}"})
    
    return call_webstat_api(series_key, limit)
   


# Configuration object used to instruct functionary model about the tools at its disposal.
WEBSTAT_TOOLS_CONFIG = [
    {
        "type": "function",
        "function": {
            "name": "get_economic_indicator",
            "description": "Récupère les dernières données d'un indicateur économique français (taux livret A, taux LDDS, taux LEP, taux PEL)",
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator_name": {
                        "type": "string", 
                        "enum": ["taux_livret_a", "taux_ldds", "taux_lep", "taux_pel","taux_refinancement_principale_bce","taux_de_depot_bce", "taux_euribor_3mois"],
                        "description": "Nom de l'indicateur économique à récupérer"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Nombre de périodes à récupérer (défaut: 5)",
                        "default": 5
                    }
                },
                "required": ["indicator_name"]
            }
        }
    }
]

# Tool map - when functionary chooses a tool, run the corresponding function from this map
TOOLS_MAP = {
    "get_economic_indicator": get_economic_indicator,
}

if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent.parent.parent.parent
    model_folder = root_folder / "models"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    print("=== TEST 1: Test direct de l'API ===")
    print(get_economic_indicator("taux_livret_a", 3))  # Majuscules pour correspondre à la config
    
    print("\n=== TEST 2: Chargement du modèle LLM ===")
    model_settings = get_model_settings("llama-3.3-70b-versatile")
    llm = LamaCppClient(model_folder, model_settings)
    print("Modèle chargé avec succès !")

    print("\n=== TEST 3: LLM décide automatiquement ===")
    tools = llm.retrieve_tools(
        prompt="Quel est le taux du Livret A actuellement ?", 
        tools=WEBSTAT_TOOLS_CONFIG
    )
    print(f"Tools choisis par le LLM: {tools}")

    print("\n=== TEST 4: LLM avec choix forcé ===")
    tools = llm.retrieve_tools(
        prompt="Donne-moi le taux du LEP",
        tools=WEBSTAT_TOOLS_CONFIG,
        tool_choice="get_economic_indicator"
    )
    print(f"Tools avec choix forcé: {tools}")

    print("\n=== TEST 5: Exécution complète avec réponse finale ===")
    if tools and len(tools) > 0:
        function_name = tools[0]["function"]["name"]
        function_args = json.loads(tools[0]["function"]["arguments"])
        func_to_call = TOOLS_MAP.get(function_name, None)
        
        if func_to_call:
            function_response = func_to_call(**function_args)
            print(f"Réponse de l'API: {function_response}")
            
            # Générer la réponse finale avec contexte
            prompt_with_function_response = llm.generate_ctx_prompt(
                question="Donne-moi le taux du LEP actuellement", 
                context=function_response
            )

            print("\n=== RÉPONSE FINALE DU CHATBOT ===")
            stream = llm.start_answer_iterator_streamer(
                prompt=prompt_with_function_response,
                max_new_tokens=256,
            )
            for output in stream:
                print(output["choices"][0]["delta"].get("content", ""), end="", flush=True)
        else:
            print("Fonction non trouvée dans TOOLS_MAP")
    else:
        print("Aucun tool sélectionné par le LLM")