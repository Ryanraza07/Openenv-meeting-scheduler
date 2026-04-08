from __future__ import annotations

import json
import os
from urllib import error, request

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from ..env.state import MeetingState
except ImportError:
    from app.env.state import MeetingState

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class SlotDecision(BaseModel):
    chosen_slot: str = Field(min_length=1)


def _build_prompt(state: MeetingState) -> str:
    state_json = state.model_dump_json(indent=2)
    return (
        "Choose the single best meeting slot from all_slots.\n"
        "Rules:\n"
        "1. If any required participant is unavailable for a slot, that slot is invalid and should not be selected.\n"
        "2. Among valid slots, maximize the sum of participant priorities for people available in that slot.\n"
        "3. If multiple slots tie, choose the earliest slot in the order listed in all_slots.\n"
        "4. Return only the chosen slot using the required response schema.\n\n"
        f"Meeting state:\n{state_json}"
    )


def _build_messages(state: MeetingState) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a deterministic scheduling agent. "
                "Respect required participants and maximize total priority. "
                'Return JSON with exactly one key: {"chosen_slot": "..."}'
            ),
        },
        {"role": "user", "content": _build_prompt(state)},
    ]


def _resolve_client_settings(model: str | None) -> tuple[str, str | None, str]:
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY is missing. Use the injected API key or set OPENAI_API_KEY locally.")

    base_url = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    selected_model = model or os.getenv("OPENAI_MODEL") or os.getenv("MODEL") or DEFAULT_MODEL
    return api_key, base_url, selected_model


def _extract_slot_from_text(state: MeetingState, content: str) -> str:
    text = content.strip()
    if not text:
        raise RuntimeError("The model returned an empty response.")

    candidates = [text]
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            chosen_slot = payload.get("chosen_slot")
            if isinstance(chosen_slot, str) and chosen_slot.strip():
                return chosen_slot.strip()

    if text in state.all_slots:
        return text

    for slot in state.all_slots:
        if slot in text:
            return slot

    raise RuntimeError(f"Could not parse a chosen slot from model output: {text}")


def _create_client(api_key: str, base_url: str | None) -> OpenAI:
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def _choose_with_openai_client(state: MeetingState, api_key: str, base_url: str | None, model: str) -> str:
    client = _create_client(api_key, base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=_build_messages(state),
            temperature=0.0,
            max_tokens=32,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=_build_messages(state),
                temperature=0.0,
                max_tokens=32,
            )
        except Exception as retry_exc:
            raise RuntimeError(f"OpenAI request failed: {retry_exc}") from retry_exc

    choice = response.choices[0] if response.choices else None
    if choice is None or choice.message is None or choice.message.content is None:
        raise RuntimeError("The OpenAI response did not include any message content.")

    return _extract_slot_from_text(state, choice.message.content)


def _chat_completions_url(base_url: str | None) -> str:
    normalized_base_url = (base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")
    return f"{normalized_base_url}/chat/completions"


def _choose_with_http_fallback(state: MeetingState, api_key: str, base_url: str | None, model: str) -> str:
    payload = {
        "model": model,
        "messages": _build_messages(state),
        "temperature": 0.0,
        "max_tokens": 32,
        "response_format": {"type": "json_object"},
    }

    def _post(chat_payload: dict[str, object]) -> str:
        body = json.dumps(chat_payload).encode("utf-8")
        req = request.Request(
            _chat_completions_url(base_url),
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=30) as response:
                return response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Proxy request failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Proxy request failed: {exc.reason}") from exc

    try:
        raw_body = _post(payload)
    except RuntimeError:
        retry_payload = dict(payload)
        retry_payload.pop("response_format", None)
        raw_body = _post(retry_payload)

    try:
        response_payload = json.loads(raw_body)
        content = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not parse proxy response: {raw_body}") from exc

    if not isinstance(content, str):
        raise RuntimeError("Proxy response content was not a string.")

    return _extract_slot_from_text(state, content)


def choose_best_slot(state: MeetingState, model: str | None = None, validate: bool = True) -> str:
    if load_dotenv is not None:
        load_dotenv()

    api_key, base_url, selected_model = _resolve_client_settings(model)

    if OpenAI is not None:
        chosen_slot = _choose_with_openai_client(state, api_key, base_url, selected_model)
    else:
        chosen_slot = _choose_with_http_fallback(state, api_key, base_url, selected_model)

    if validate and chosen_slot not in state.all_slots:
        raise ValueError(
            f"Model returned an invalid slot '{chosen_slot}'. Expected one of: {state.all_slots}"
        )

    return chosen_slot
