import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from helpers.log import get_logger
from helpers.prettier import prettify_source
from helpers.reader import read_input
from pyfiglet import Figlet
from rich.console import Console
from rich.markdown import Markdown

sys.path.append(str(Path(__file__).parent.parent.parent))
from bot.client.lama_cpp_client import LamaCppClient
from bot.conversation.chat_history import ChatHistory
from bot.conversation.conversation_handler import answer_with_context, refine_question
from bot.conversation.ctx_strategy import get_ctx_synthesis_strategies, get_ctx_synthesis_strategy
from bot.memory.embedder import Embedder
from bot.memory.vector_database.chroma import Chroma
from bot.model.model_registry import Model, get_model_settings, get_models

logger = get_logger(__name__)

# Ajouts en haut du fichier
from experiments.exp_lama_cpp.function_calling import (
    get_economic_indicator, 
    WEBSTAT_TOOLS_CONFIG, 
    TOOLS_MAP
)
import json

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Chatbot")

    model_list = get_models()
    default_model = "llama-3.3-70b-versatile"  # Pour supporter les function calls

    synthesis_strategy_list = get_ctx_synthesis_strategies()
    default_synthesis_strategy = synthesis_strategy_list[0]

    parser.add_argument(
        "--model",
        type=str,
        choices=model_list,
        help=f"Model to be used. Defaults to {default_model}.",
        required=False,
        const=default_model,
        nargs="?",
        default=default_model,
    )

    parser.add_argument(
        "--synthesis-strategy",
        type=str,
        choices=synthesis_strategy_list,
        help=f"Model to be used. Defaults to {default_synthesis_strategy}.",
        required=False,
        const=default_synthesis_strategy,
        nargs="?",
        default=default_synthesis_strategy,
    )

    parser.add_argument(
        "--k",
        type=int,
        help="Number of chunks to return from the similarity search. Defaults to 2.",
        required=False,
        default=3,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="The maximum number of new tokens to generate.. Defaults to 512.",
        required=False,
        default=512,
    )

    return parser.parse_args()


def loop(llm, chat_history, synthesis_strategy, index, parameters) -> None:
    custom_fig = Figlet(font="graffiti")
    console = Console(color_system="windows")
    console.print(custom_fig.renderText("ChatBot"))
    console.print(
        "[bold magenta]Hi! 👋, I'm your friendly chatbot 🦜 here to assist you. "
        "\nHow can I help you today? [/bold "
        "magenta]Type 'exit' to stop."
    )

    while True:
        console.print("[bold green]Please enter your question:[/bold green]")
        question = read_input()

        if question.lower() == "exit":
            break

        logger.info(f"--- Question: {question}, Chat_history: {chat_history} ---")

        start_time = time.time()
        refined_question = refine_question(llm, question, chat_history)

        # NOUVEAU : Vérifier si APIs nécessaires
        api_tools_needed = llm.retrieve_tools(
            refined_question, 
            tools=WEBSTAT_TOOLS_CONFIG
        )
        
        # Debug pour voir si des APIs sont appelées
        if api_tools_needed:
            console.print(f"[yellow]APIs détectées: {len(api_tools_needed)} tool(s)[/yellow]")

        # Si API détectée, pas de RAG du tout
        if api_tools_needed:
            retrieved_contents, sources = [], []  # Pas de recherche RAG
        else:
            retrieved_contents, sources = index.similarity_search_with_threshold(
                query=refined_question, k=parameters.k
            )
        
        # NOUVEAU : Appeler les APIs si nécessaire
        api_context = ""
        if api_tools_needed and len(api_tools_needed) > 0:
            for tool in api_tools_needed:
                #function_name = tool["function"]["name"]
                function_name = tool.function.name
                #function_args = json.loads(tool["function"]["arguments"])
                function_args = json.loads(tool.function.arguments)
                func_to_call = TOOLS_MAP.get(function_name)
                if func_to_call:
                    try:
                        api_response = func_to_call(**function_args)
                        api_context += f"\n--- Données économiques récentes ---\n{api_response}"
                        console.print(f"[green]✓ Données récupérées depuis {function_name}[/green]")
                    except Exception as e:  
                        console.print(f"[red]Erreur API {function_name}: {e}[/red]") 
        
        # Créer le contexte final
        if api_context:
            # Créer un document uniquement avec les données API
            from bot.memory.vector_database.chroma import Document
            
            api_document = Document(
                page_content=f"{api_context}",  # Seulement les données API
                metadata={"source": "API Webstat Banque de France"}
            )
            final_context = [api_document]  # Seulement l'API, pas de RAG
        else:
            final_context = retrieved_contents

        console.print("\n[bold magenta]Sources:[/bold magenta]")
        for source in sources:
            console.print(Markdown(prettify_source(source)))

        console.print("\n[bold magenta]Answer:[/bold magenta]")

        streamer, fmt_prompts = answer_with_context(
            llm=llm,
            ctx_synthesis_strategy=synthesis_strategy,
            question=refined_question,
            chat_history=chat_history,
            retrieved_contents=final_context, 
           # max_new_tokens=256, # Pour limiter la longueur des réponses
            max_new_tokens=parameters.max_new_tokens
        )
         

           
        
        answer = ""
        for token in streamer:
            parsed_token = llm.parse_token(token)
            answer += parsed_token
            print(parsed_token, end="", flush=True)
        
        # Nettoyer la réponse après la génération complète
        answer = answer.split("| end |")[0]  # Couper au marqueur de fin
        answer = answer.split("<|im_start|>")[0]  # Couper aux balises parasites
        answer = answer.strip()  # Supprimer les espaces en trop

        chat_history.append(
            f"question: {refined_question}, answer: {answer}",
        )

        console.print("\n[bold magenta]Formatted Answer:[/bold magenta]")
        if answer:
            console.print(Markdown(answer))
            took = time.time() - start_time
            print(f"\n--- Took {took:.2f} seconds ---")
        else:
            console.print("[bold red]Something went wrong![/bold red]")


def main(parameters):
    model_settings = get_model_settings(parameters.model)

    root_folder = Path(__file__).resolve().parent.parent.parent
    model_folder = root_folder / "models"
    vector_store_path = root_folder / "vector_store" / "docs_index"

    llm = LamaCppClient(model_folder=model_folder, model_settings=model_settings)

    synthesis_strategy = get_ctx_synthesis_strategy(parameters.synthesis_strategy, llm=llm)
    chat_history = ChatHistory(total_length=2)

    embedding = Embedder()
    index = Chroma(persist_directory=str(vector_store_path), embedding=embedding)

    loop(llm, chat_history, synthesis_strategy, index, parameters)


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)