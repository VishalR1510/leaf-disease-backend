"""
Groq AI service for leaf verification and disease detail generation.

Two responsibilities:
  1. Vision model (llama-4-scout) — confirms if an image contains a leaf.
  2. Reasoning model (gpt-oss-120b) — provides detailed disease information
     given a disease name from the Keras classifier.
"""

import json
import re
from typing import Any

from groq import Groq, APIError, APIConnectionError, RateLimitError
from loguru import logger

from app.core.config import Settings


class GroqServiceError(Exception):
    """Raised when the Groq AI service encounters an error."""

    def __init__(self, code: str = "AI_SERVICE_ERROR", message: str = "") -> None:
        self.code = code
        self.message = message or "An error occurred while analysing the image."
        super().__init__(self.message)


# ── Prompt for disease details (reasoning model) ─────────────────────
DISEASE_DETAIL_PROMPT = """You are an expert agricultural plant pathologist.

The plant "{plant_name}" has been diagnosed with "{disease_name}" by an image classifier.

Provide detailed information about this disease:

1. Describe the visible symptoms on the leaf.
2. Provide a short description of the disease.
3. Explain possible causes.
4. Provide preventive measures.
5. Provide treatment or management recommendations.

If the leaf is "Healthy", indicate that no disease was found and give general plant care tips.

Respond ONLY in valid JSON format using this schema:

{{
  "symptoms": "string",
  "disease_description": "string",
  "possible_causes": "string",
  "preventive_measures": "string",
  "treatment_suggestions": "string"
}}

Do not include any explanation outside the JSON."""


class GroqService:
    """Communicates with the Groq API for leaf verification and disease details."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = Groq(api_key=settings.GROQ_API_KEY)
        logger.info(
            "GroqService initialised — vision_model={}, reasoning_model={}",
            settings.GROQ_VISION_MODEL,
            settings.GROQ_REASONING_MODEL,
        )

    # ── 1. Leaf verification (vision model) ───────────────

    async def confirm_leaf(self, base64_image: str) -> bool:
        """
        Ask the Groq vision model whether the image contains a plant leaf.

        Returns True if it's a leaf, False otherwise.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._settings.GROQ_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Look at this image carefully. "
                                    "Does it contain a plant leaf? "
                                    "Reply with ONLY 'yes' or 'no'."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=10,
            )
            answer = response.choices[0].message.content.strip().lower()
            logger.info("Leaf verification answer: '{}'", answer)
            return "yes" in answer

        except (APIConnectionError, RateLimitError) as exc:
            logger.error("Groq vision API error: {}", exc)
            raise GroqServiceError(
                code="AI_SERVICE_UNAVAILABLE",
                message="The AI service is temporarily unavailable. Please try again later.",
            ) from exc
        except APIError as exc:
            logger.error("Groq vision API error (status={}): {}", exc.status_code, exc.message)
            raise GroqServiceError(
                code="AI_SERVICE_ERROR",
                message="The AI service returned an error. Please try again.",
            ) from exc

    # ── 2. Disease details (reasoning model) ──────────────

    async def get_disease_details(
        self, plant_name: str, disease_name: str
    ) -> dict[str, str]:
        """
        Use the reasoning model to generate detailed disease information.

        Returns dict with: symptoms, disease_description, possible_causes,
        preventive_measures, treatment_suggestions.
        """
        try:
            prompt = DISEASE_DETAIL_PROMPT.format(
                plant_name=plant_name,
                disease_name=disease_name,
            )

            logger.info(
                "Requesting disease details — plant='{}', disease='{}', model='{}'",
                plant_name, disease_name, self._settings.GROQ_REASONING_MODEL,
            )

            response = self._client.chat.completions.create(
                model=self._settings.GROQ_REASONING_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert agricultural plant pathologist. Respond only in valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self._settings.LLM_TEMPERATURE,
                top_p=self._settings.LLM_TOP_P,
                max_tokens=self._settings.LLM_MAX_TOKENS,
            )

            content: str = response.choices[0].message.content  # type: ignore[assignment]
            logger.debug("Reasoning model response length: {} chars", len(content))

            raw = self._extract_json(content)
            return raw

        except (APIConnectionError, RateLimitError) as exc:
            logger.error("Groq reasoning API error: {}", exc)
            raise GroqServiceError(
                code="AI_SERVICE_UNAVAILABLE",
                message="The AI service is temporarily unavailable. Please try again later.",
            ) from exc

        except APIError as exc:
            logger.error("Groq reasoning API error (status={}): {}", exc.status_code, exc.message)
            raise GroqServiceError(
                code="AI_SERVICE_ERROR",
                message="The AI service returned an error. Please try again.",
            ) from exc

        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.error("Failed to parse reasoning response: {}", exc)
            raise GroqServiceError(
                code="AI_PARSE_ERROR",
                message="Could not interpret the AI response. Please try again.",
            ) from exc

    # ── Helpers ───────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract JSON from text that may be wrapped in markdown fences."""
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
        return json.loads(cleaned)
