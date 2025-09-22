from typing import Any, Iterator
from groq import Groq
import asyncio



from helpers.log import experimental
from bot.client.prompt import (
    CTX_PROMPT_TEMPLATE,
    QA_PROMPT_TEMPLATE,
    REFINED_ANSWER_CONVERSATION_AWARENESS_PROMPT_TEMPLATE,
    REFINED_CTX_PROMPT_TEMPLATE,
    REFINED_QUESTION_CONVERSATION_AWARENESS_PROMPT_TEMPLATE,
    SYSTEM_TEMPLATE,
    TOOL_SYSTEM_TEMPLATE,
    generate_conversation_awareness_prompt,
    generate_ctx_prompt,
    generate_qa_prompt,
    generate_refined_ctx_prompt,
)
from bot.model.base_model import ModelSettings
from dotenv import load_dotenv
load_dotenv()  

class LamaCppClient:
    """
    Class for implementing language model client using Groq.
    """

    def __init__(self, model_folder: str, model_settings: ModelSettings):
        self.model_settings = model_settings

        # Initialise le client Groq
        self.llm = self._load_llm()

    def _load_llm(self) -> Any:
        """
        Initialise le client Groq (pas besoin de model_path ou download)
        """
        return Groq()

    def generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Génère une réponse avec Groq.
        """
        output = self.llm.chat.completions.create(
            model=self.model_settings.file_name,  # ex: "llama-3.3-70b-versatile"
            messages=[
                {"role": "system", "content": SYSTEM_TEMPLATE},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_new_tokens,
            **self.model_settings.config_answer,
        )
        return output.choices[0].message.content

    async def async_generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        return self.generate_answer(prompt, max_new_tokens)

    def stream_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Stream réponse token par token.
        """
        answer = ""
        stream = self.start_answer_iterator_streamer(prompt, max_new_tokens=max_new_tokens)
        for token in stream:
            delta = token["choices"][0]["delta"].get("content", "")
            answer += delta
            print(delta, end="", flush=True)
        return answer

    def start_answer_iterator_streamer(self, prompt: str, max_new_tokens: int = 512) -> Iterator[dict]:
        """
        Retourne un itérateur de tokens.
        """
        return self.llm.chat.completions.create(
            model=self.model_settings.file_name,
            messages=[
                {"role": "system", "content": SYSTEM_TEMPLATE},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_new_tokens,
            stream=True,
            **self.model_settings.config_answer,
        )

    async def async_start_answer_iterator_streamer(self, prompt: str, max_new_tokens: int = 512) -> Iterator[dict]:
        return self.start_answer_iterator_streamer(prompt, max_new_tokens=max_new_tokens)

    @experimental
    def retrieve_tools(self, prompt: str, max_new_tokens: int = 512, tools: list[dict] = None, tool_choice: str = None) -> list[dict] | None:
        """
        Détecte si des APIs / outils doivent être utilisés.
        """
        tool_choice = {"type": "function", "function": {"name": tool_choice}} if tool_choice else "auto"
        output = self.llm.chat.completions.create(
            model=self.model_settings.file_name,
            messages=[
                {"role": "system", "content": TOOL_SYSTEM_TEMPLATE},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_new_tokens,
            stream=False,
            tools=tools,
            tool_choice=tool_choice,
            **self.model_settings.config_answer,
        )
        #return getattr(output.choices[0].message, "tool_calls", None)
        tool_calls = getattr(output.choices[0].message, "tool_calls", None)
        if not tool_calls:
           return None
        return tool_calls

       #return output.choices[0].message.get("tool_calls", None)

    @staticmethod
    def parse_token(token):
       # return token["choices"][0]["delta"].get("content", "")
        content = getattr(token.choices[0].delta, "content", "")
        return content or ""
    # Prompt helpers
    @staticmethod
    def generate_qa_prompt(question: str) -> str:
        return generate_qa_prompt(QA_PROMPT_TEMPLATE, SYSTEM_TEMPLATE, question)

    @staticmethod
    def generate_ctx_prompt(question: str, context: str) -> str:
        return generate_ctx_prompt(CTX_PROMPT_TEMPLATE, SYSTEM_TEMPLATE, question, context)

    @staticmethod
    def generate_refined_ctx_prompt(question: str, context: str, existing_answer: str) -> str:
        return generate_refined_ctx_prompt(REFINED_CTX_PROMPT_TEMPLATE, SYSTEM_TEMPLATE, question, context, existing_answer)

    @staticmethod
    def generate_refined_question_conversation_awareness_prompt(question: str, chat_history: str) -> str:
        return generate_conversation_awareness_prompt(REFINED_QUESTION_CONVERSATION_AWARENESS_PROMPT_TEMPLATE, SYSTEM_TEMPLATE, question, chat_history)

    @staticmethod
    def generate_refined_answer_conversation_awareness_prompt(question: str, chat_history: str) -> str:
        return generate_conversation_awareness_prompt(REFINED_ANSWER_CONVERSATION_AWARENESS_PROMPT_TEMPLATE, SYSTEM_TEMPLATE, question, chat_history)
