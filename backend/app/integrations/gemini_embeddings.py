from google import genai
from google.genai import types

class GeminiEmbeddingProvider:
    def __init__(
        self,
        client: genai.Client,
        model_id: str = "gemini-embedding-2",
        dimension: int = 768,
    ) -> None:
        self._client = client
        self.model_id = model_id
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        contents = [
            types.Content(parts=[types.Part.from_text(text=text)])
            for text in texts
        ]
        response = self._client.models.embed_content(
            model=self.model_id,
            contents=contents,
            config=types.EmbedContentConfig(output_dimensionality=self.dimension),
        )
        embeddings = response.embeddings or []
        return [list(embedding.values or []) for embedding in embeddings]