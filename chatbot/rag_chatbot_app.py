import argparse
import sys
import time
from pathlib import Path
import streamlit as st
import json
from bot.client.lama_cpp_client import LamaCppClient
from bot.conversation.chat_history import ChatHistory
from bot.conversation.conversation_handler import answer_with_context, refine_question
from bot.conversation.ctx_strategy import (
    BaseSynthesisStrategy,
    get_ctx_synthesis_strategies,
    get_ctx_synthesis_strategy,
)
from bot.memory.embedder import Embedder
from bot.memory.vector_database.chroma import Chroma
from bot.model.model_registry import get_model_settings, get_models
from helpers.log import get_logger
from helpers.prettier import prettify_source

# Ajout des imports pour l'API Webstat
from experiments.exp_lama_cpp.function_calling import (
    get_economic_indicator, 
    WEBSTAT_TOOLS_CONFIG, 
    TOOLS_MAP
)

logger = get_logger(__name__)

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", initial_sidebar_state="collapsed")


@st.cache_resource()
def load_llm_client(model_folder: Path, model_name: str) -> LamaCppClient:
    model_settings = get_model_settings(model_name)
    llm = LamaCppClient(model_folder=model_folder, model_settings=model_settings)
    return llm


@st.cache_resource()
def init_chat_history(total_length: int = 2) -> ChatHistory:
    chat_history = ChatHistory(total_length=total_length)
    return chat_history


@st.cache_resource()
def load_ctx_synthesis_strategy(ctx_synthesis_strategy_name: str, _llm: LamaCppClient) -> BaseSynthesisStrategy:
    ctx_synthesis_strategy = get_ctx_synthesis_strategy(ctx_synthesis_strategy_name, llm=_llm)
    return ctx_synthesis_strategy


@st.cache_resource()
def load_index(vector_store_path: Path) -> Chroma:
    """
    Loads a Vector Database index based on the specified vector store path.
    """
    embedding = Embedder()
    index = Chroma(persist_directory=str(vector_store_path), embedding=embedding)
    return index


def init_page(root_folder: Path) -> None:
    """
    Initializes the page configuration for the application.
    """
    left_column, central_column, right_column = st.columns([2, 1, 2])

    with left_column:
        st.write(" ")

    with central_column:
        st.image(str(root_folder / "images/bot.png"), use_column_width="always")
        st.markdown("""<h4 style='text-align: center; color: grey;'></h4>""", unsafe_allow_html=True)

    with right_column:
        st.write(" ")

    st.sidebar.title("Options")


@st.cache_resource
def init_welcome_message() -> None:
    """
    Initializes a welcome message for the chat interface.
    """
    with st.chat_message("assistant"):
        st.write("Comment puis-je vous aider aujourd'hui ? 🏦")


def reset_chat_history(chat_history: ChatHistory) -> None:
    """
    Initializes the chat history, allowing users to clear the conversation.
    """
    clear_button = st.sidebar.button("🗑️ Effacer la conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = []
        chat_history.clear()


def display_messages_from_history():
    """
    Displays chat messages from the history on app rerun.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main(parameters) -> None:
    """
    Main function to run the RAG Chatbot application.
    """
    root_folder = Path(__file__).resolve().parent.parent
    model_folder = root_folder / "models"
    vector_store_path = root_folder / "vector_store" / "docs_index"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    model_name = parameters.model
    synthesis_strategy_name = parameters.synthesis_strategy

    init_page(root_folder)
    llm = load_llm_client(model_folder, model_name)
    chat_history = init_chat_history(2)
    ctx_synthesis_strategy = load_ctx_synthesis_strategy(synthesis_strategy_name, _llm=llm)
    index = load_index(vector_store_path)
    reset_chat_history(chat_history)
    init_welcome_message()
    display_messages_from_history()

    # Supervise user input
    if user_input := st.chat_input("Posez votre question !"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process the question and check for API needs
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Analyse de votre question et recherche des données..."):
                #refined_user_input = refine_question(llm, user_input, chat_history=chat_history)
                refined_user_input = user_input  
                
                # NOUVEAU : Vérifier si APIs nécessaires - uniquement pour les taux
                rate_keywords = [
                    'livret a', 'ldds', 'lep', 'pel', 'refinancement bce', 'dépôt bce', 'euribor',
                    'taux livret', 'taux ldds', 'taux lep', 'taux pel', 'taux bce', 'taux euribor',
                    'livret', 'euribor 3 mois'
                ]

                is_rate_question = any(keyword in refined_user_input.lower() for keyword in rate_keywords)

                if is_rate_question:
                    api_tools_needed = llm.retrieve_tools(refined_user_input, tools=WEBSTAT_TOOLS_CONFIG)
                else:
                    api_tools_needed = []
                
                # Debug dans Streamlit
                if api_tools_needed:
                    #st.info(f"🔍 API détectée : {len(api_tools_needed)} outil(s) économique(s)")
                    logger.info(f"API détectée : {len(api_tools_needed)} outil(s)")
                
                # NOUVEAU : Logique API vs RAG - même logique que CLI (j'ai changer pour un autre)
                #if api_tools_needed:
                    # Pas de recherche RAG pour les questions économiques
                   # retrieved_contents, sources = [], []
               # else:
                   # retrieved_contents, sources = index.similarity_search_with_threshold(
                    #    query=refined_user_input, k=parameters.k
                   # )
                # REMPLACER par :
                if is_rate_question:
                   api_tools_needed = llm.retrieve_tools(refined_user_input, tools=WEBSTAT_TOOLS_CONFIG)
                   retrieved_contents, sources = [], []  # FORCER : Pas de RAG
                else:
                  api_tools_needed = []  # FORCER : Pas d'API
                  retrieved_contents, sources = index.similarity_search_with_threshold(
                  query=refined_user_input, k=parameters.k
    )   
                
                message_placeholder.markdown(full_response)
        
        # NOUVEAU : Appel des APIs si nécessaire - même logique que CLI
        api_context = ""
        if api_tools_needed and len(api_tools_needed) > 0:
            with st.spinner("Récupération des données de la Banque de France..."):
                for tool in api_tools_needed:
                    # Adaptation pour la structure Groq comme dans CLI
                    function_name = tool.function.name
                    function_args = json.loads(tool.function.arguments)
                    func_to_call = TOOLS_MAP.get(function_name)
                    if func_to_call:
                        try:
                            api_response = func_to_call(**function_args)
                            api_context += f"\n--- Données économiques récentes ---\n{api_response}"
                            #st.success(f"✅ Données récupérées depuis {function_name}")
                            logger.info(f"Données récupérées depuis {function_name}")
                        except Exception as e:
                            # st.error(f"❌ Erreur API {function_name}: {e}")
                            logger.error(f"❌ Erreur API {function_name}: {e}")
        
        # Créer le contexte final - même logique que CLI
        #if api_context:
        if api_tools_needed and api_context:
            from bot.memory.vector_database.chroma import Document
            api_document = Document(
                page_content=f"{api_context}",
                metadata={"source": "API Webstat Banque de France"}
            )
            final_context = [api_document]  # Seulement l'API, pas de RAG
        else:
            final_context = retrieved_contents

        # Afficher les sources
        console_sources = ""
        for source in sources:
            console_sources += prettify_source(source) + "\n\n"
        
        if console_sources:
            st.session_state.messages.append({"role": "assistant", "content": f"**Sources:**\n\n{console_sources}"})

        # Generate final answer
        start_time = time.time()
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Génération de la réponse finale..."):
                streamer, fmt_prompts = answer_with_context(
                    llm=llm,
                    ctx_synthesis_strategy=ctx_synthesis_strategy,
                    question=refined_user_input,
                    chat_history=chat_history,
                    retrieved_contents=final_context,
                    max_new_tokens=parameters.max_new_tokens
                )
                
                for token in streamer:
                    parsed_token = llm.parse_token(token)
                    full_response += parsed_token
                    message_placeholder.markdown(full_response + "▌")

                # Clean the response - même logique que CLI
                full_response = full_response.split("| end |")[0]
                full_response = full_response.split("<|im_start|>")[0]
                full_response = full_response.strip()
                
                message_placeholder.markdown(full_response)
                chat_history.append(f"question: {refined_user_input}, answer: {full_response}")

        # Add final assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        took = time.time() - start_time
        st.sidebar.info(f"⏱️ Réponse générée en {took:.2f} secondes")
        logger.info(f"\n--- Took {took:.2f} seconds ---")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Chatbot")

    model_list = get_models()
    # Même modèle par défaut que CLI
    default_model = "llama-3.3-70b-versatile"

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
        help="Number of chunks to return from the similarity search. Defaults to 3.",
        required=False,
        default=3,  # Même valeur que CLI
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="The maximum number of new tokens to generate. Defaults to 512.",
        required=False,
        default=512,  # Même valeur que CLI
    )

    return parser.parse_args()


# streamlit run rag_chatbot_app.py
if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)