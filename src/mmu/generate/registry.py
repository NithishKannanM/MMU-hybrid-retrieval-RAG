"""Backend factory — the one place a generator *alias* becomes a live :class:`Generator`.

**Deferred imports**, mirroring ``local_infra/core/registry.py``: selecting the ollama
backend must never import ``mmu.generate.anthropic`` (which lazy-imports the optional
``anthropic`` package on top of that), and selecting anthropic must never import
``mmu.generate.ollama``. Neither backend module is imported until :func:`build_generator`
actually needs it.

**Aliases only, never a raw model id.** ``build_generator`` takes ``alias: str | None``,
not ``model: str``. This is a security boundary, not a style preference: `AnswerRequest`
(WP7) exposes this same alias parameter to HTTP callers, and if a raw model string were
accepted here, a request could name an arbitrary cloud model and route straight past the
local default the rest of this project is built around — including its cost, its
latency profile, and the fact that a benchmark run should always be comparing runs of
the *same*, intentionally-chosen model. The alias table below is the only thing that can
turn a request into a model id; extending it means editing this file, not passing a new
string on the wire.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mmu.config import Settings
    from mmu.generate.base import Generator

#: alias -> the Settings attribute holding the Ollama model id it resolves to. This is
#: the entire alias table for the local backend; there is deliberately no way to spell a
#: model id here other than by adding a line to this dict.
_OLLAMA_ALIASES: dict[str, str] = {
    "default": "ollama_model",
    "fast": "ollama_fast_model",
}

#: Aliases valid under the anthropic backend. Settings carries only one configured
#: Anthropic model (no "fast" cloud tier), so only "default" resolves; "fast" falls
#: through to the same unknown-alias error as any other unrecognized name rather than
#: silently reusing the default model under a different name.
_ANTHROPIC_ALIASES: dict[str, str] = {
    "default": "anthropic_model",
}


def build_generator(settings: "Settings", alias: str | None = None) -> "Generator":
    """Build the :class:`Generator` named by ``settings.generator_backend``, with the
    model chosen by ``alias`` (``"default"`` when ``alias`` is ``None``).

    Raises :class:`KeyError` for an alias unknown to the selected backend, listing the
    known aliases — the caller (a CLI flag or an API request field) gets an actionable
    message instead of a generator silently defaulting to something else. Raises
    :class:`ValueError` for an unrecognized ``settings.generator_backend`` — that one is
    a misconfiguration, not a per-request input, so it does not share KeyError's
    "here are the valid choices" shape aimed at request-time callers.
    """
    backend = settings.generator_backend.lower()
    if backend == "ollama":
        from mmu.generate.ollama import OllamaGenerator

        model = _resolve(_OLLAMA_ALIASES, settings, alias, backend="ollama")
        return OllamaGenerator(
            host=settings.ollama_host,
            model=model,
            num_ctx=settings.ollama_num_ctx,
            timeout=settings.request_timeout_s,
        )
    if backend == "anthropic":
        from mmu.generate.anthropic import AnthropicGenerator

        model = _resolve(_ANTHROPIC_ALIASES, settings, alias, backend="anthropic")
        return AnthropicGenerator(
            model=model,
            api_key=settings.anthropic_api_key,
            timeout=settings.request_timeout_s,
        )
    raise ValueError(
        f"unknown generator_backend {settings.generator_backend!r} "
        f"(expected 'ollama' or 'anthropic')"
    )


def _resolve(table: dict[str, str], settings: "Settings", alias: str | None, *, backend: str) -> str:
    key = alias or "default"
    try:
        attr = table[key]
    except KeyError:
        known = ", ".join(sorted(table))
        raise KeyError(
            f"unknown generator alias {key!r} for backend {backend!r}; "
            f"known aliases: {known}"
        ) from None
    return getattr(settings, attr)


def available_aliases(settings: "Settings") -> list[dict[str, object]]:
    """Report each alias valid under the *current* ``settings.generator_backend``, for
    ``GET /health``. Reports capability only — the model id and whether it is local —
    never a secret: an Anthropic entry never includes ``settings.anthropic_api_key``,
    and its presence/absence is not reported here either, since that is a readiness
    concern for `/health` to check separately, not an aliasing concern for this module.
    """
    backend = settings.generator_backend.lower()
    table = _ANTHROPIC_ALIASES if backend == "anthropic" else _OLLAMA_ALIASES
    return [
        {
            "alias": alias,
            "backend": backend,
            "local": backend != "anthropic",
            "model": getattr(settings, attr),
        }
        for alias, attr in table.items()
    ]
