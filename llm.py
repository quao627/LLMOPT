# src/utils/llm.py

# general
import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# openai
from openai import OpenAI

# google
from google import genai
from google.genai import types


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response from the LLM."""
        pass


class OpenAIClient(LLMClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        load_dotenv()
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        self.client = OpenAI(api_key=openai_key)


    def generate(self, prompt: str, model="gpt-4o", temperature=1, force_json=False):
        messages = [{"role": "user", "content": prompt}]
        try: 
            if force_json:
                response = self.client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    messages=messages
                )
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=messages
                )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error: {str(e)}"

class GeminiClient(LLMClient):
    """Google Gemini API client implementation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        load_dotenv()
        gemini_key = os.getenv('GEMINI_API_KEY')
        if not gemini_key:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")
        self.client = genai.Client(api_key=gemini_key)

    def generate(self, prompt: str, model="gemini-2.5-flash", temperature=1, force_json=False) -> str:
        """Generate text using Gemini API."""
        if force_json:
            response = self.client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=[types.Part.from_text(prompt)])],
                config=types.GenerateContentConfig(response_mime_type="application/json")    
            )
        else:
            response = self.client.models.generate_content(
                    model=model,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(prompt)])],    
                )
        return response.text


def create_llm_client(provider: str, **kwargs) -> LLMClient:
    if provider == "openai":
        return OpenAIClient(**kwargs)
    elif provider == "gemini":
        return GeminiClient(**kwargs)
    else:
        raise ValueError(f"Invalid provider: {provider}")


if __name__ == "__main__":
    # client = create_llm_client("gemini")
    # print(client.generate("What is the capital of France?"))

    client = create_llm_client("openai")
    print(client.generate("What is the capital of France?"))

