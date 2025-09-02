# A string template for the system message.
# This template is used to define the behavior and characteristics of the assistant.
SYSTEM_TEMPLATE = """Tu es un assistant IA spécialisé en conseil bancaire français.
Tu utilises uniquement les données officielles de la Banque de France et les codes légaux français.
Tu es professionnel, précis et tu réponds toujours en français.
"""

# A string template for the system message when the assistant can call functions.
# This template is used to define the behavior and characteristics of the assistant
# with the capability to call functions with appropriate input when necessary.
TOOL_SYSTEM_TEMPLATE = """Tu es un assistant IA spécialisé en conseil bancaire français.
Tu utilises uniquement les données officielles de la Banque de France et les codes légaux français.
Tu peux utiliser des fonctions appropriées quand c'est nécessaire.
Tu réponds toujours en français de manière professionnelle.
"""

# A string template with placeholders for question.
QA_PROMPT_TEMPLATE = """Réponds à la question suivante en tant que conseiller bancaire expert :
{question}
"""

# A string template with placeholders for question, and context.
CTX_PROMPT_TEMPLATE = """Informations contextuelles bancaires ci-dessous :
---------------------
{context}
---------------------
En utilisant UNIQUEMENT les informations contextuelles ci-dessus et non tes connaissances antérieures, réponds à la question suivante :

Règles importantes :
1. Réponds UNIQUEMENT en français
2. Cite tes sources précises ("Selon le code monétaire article X..." ou "D'après Webstat série Y...")
3. Si l'information concerne des taux ou chiffres, précise la date des données
4. Si tu ne trouves pas l'information dans le contexte, dis "Je ne trouve pas cette information dans les données fournies"
5. Sois précis et professionnel comme un conseiller bancaire

Question : {question}
"""

# A string template with placeholders for question, existing_answer, and context.
REFINED_CTX_PROMPT_TEMPLATE = """Question originale : {question}
Réponse existante : {existing_answer}
Nous avons l'opportunité d'améliorer cette réponse avec des informations contextuelles supplémentaires ci-dessous.
---------------------
{context}
---------------------
Avec ce nouveau contexte bancaire, améliore la réponse originale pour mieux répondre à la question.
Si le contexte n'est pas utile, retourne la réponse originale.

Règles :
- Réponds en français uniquement
- Cite tes sources précises
- Indique les dates pour les données chiffrées
- Sois professionnel et précis

Réponse améliorée :
"""

# A string template with placeholders for question, and chat_history to refine the question based on the chat history.
REFINED_QUESTION_CONVERSATION_AWARENESS_PROMPT_TEMPLATE = """Historique de la conversation bancaire :
---------------------
{chat_history}
---------------------
Question de suivi : {question}
Étant donné la conversation ci-dessus et cette question de suivi, reformule la question de suivi pour qu'elle soit une question autonome et claire dans le contexte bancaire.
Question autonome :
"""

# A string template with placeholders for question, and chat_history to answer the question based on the chat history.
REFINED_ANSWER_CONVERSATION_AWARENESS_PROMPT_TEMPLATE = """
Tu es un assistant IA spécialisé en conseil bancaire français qui engage une conversation naturelle avec un conseiller bancaire.
Ton objectif est de répondre de manière professionnelle et contextuelle en utilisant l'historique de la conversation.
La conversation doit être naturelle, cohérente et pertinente dans le contexte bancaire français.

Historique de la conversation :
---------------------
{chat_history}
---------------------
Question de suivi : {question}

En utilisant le contexte fourni dans l'historique de la conversation et la question de suivi, réponds à la question ci-dessus.
Si la question de suivi n'est pas liée au contexte de l'historique, réponds simplement à la question en ignorant l'historique.

Règles importantes :
- Réponds en français uniquement
- Sois concis et professionnel
- Cite tes sources quand pertinent
- Reste dans le contexte bancaire français
- N'invente pas d'informations

"""


def generate_qa_prompt(template: str, system: str, question: str) -> str:
    """
    Generates a prompt for a question-answer task.

    Args:
        template (str): A string template with placeholders for system, question.
        system (str): The name or identifier of the system related to the question.
        question (str): The question to be included in the prompt.

    Returns:
        str: The generated prompt.
    """

    prompt = template.format(system=system, question=question)
    return prompt


def generate_ctx_prompt(template: str, system: str, question: str, context: str = "") -> str:
    """
    Generates a prompt for a context-aware question-answer task.

    Args:
        template (str): A string template with placeholders for system, question, and context.
        system (str): The name or identifier of the system related to the question.
        question (str): The question to be included in the prompt.
        context (str, optional): Additional context information. Defaults to "".

    Returns:
        str: The generated prompt.
    """

    prompt = template.format(system=system, context=context, question=question)
    return prompt


def generate_refined_ctx_prompt(
    template: str, system: str, question: str, existing_answer: str, context: str = ""
) -> str:
    """
    Generates a prompt for a refined context-aware question-answer task.

    Args:
        template (str): A string template with placeholders for system, question, existing_answer, and context.
        system (str): The name or identifier of the system related to the question.
        question (str): The question to be included in the prompt.
        existing_answer (str): The existing answer associated with the question.
        context (str, optional): Additional context information. Defaults to "".

    Returns:
        str: The generated prompt.
    """

    prompt = template.format(
        system=system,
        context=context,
        existing_answer=existing_answer,
        question=question,
    )
    return prompt


def generate_conversation_awareness_prompt(template: str, system: str, question: str, chat_history: str) -> str:
    """
    Generates a prompt for a conversation-awareness task.

    Args:
        template (str): A string template with placeholders for system, question, and chat_history.
        system (str): The name or identifier of the system related to the question.
        question (str): The question to be included in the prompt.
        chat_history (str): The chat history associated with the conversation.

    Returns:
        str: The generated prompt.
    """

    prompt = template.format(
        system=system,
        chat_history=chat_history,
        question=question,
    )
    return prompt