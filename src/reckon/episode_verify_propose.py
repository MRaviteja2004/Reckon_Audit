"""LLM fallback proposals for episode-level Stage 3 verification."""

import os
from enum import Enum

import instructor
from pydantic import BaseModel, ConfigDict, Field

from reckon.parser import make_openai_client
from reckon.retry import RetryCallback, call_with_retry
from reckon.schema import ExpectedStep, ScreenEpisode

EPISODE_VERIFY_MODEL = os.environ.get(
    "ALIGN_ASSIST_MODEL",
    "claude-opus-5-5",
)
EPISODE_VERIFY_CONFIDENCE_FLOOR = 0.7
_TEMPERATURE: float | None = None

_EPISODE_SYSTEM = """\
You verify whether a UI test expected step is proven by one screen episode.

Return VERIFIED only when visible_texts or the screen summary directly show the \
expected assertion / outcome. Return FAILED when the episode shows a contradictory \
failure or error outcome. Return INSUFFICIENT_EVIDENCE when proof is absent or \
ambiguous — never guess.

When VERIFIED:
- evidence_quote MUST be an exact substring copied from the episode's \
visible_texts or screen_summary (or an action's visible evidence text shown \
there). Do not paraphrase.
- best_action_index MUST be an observed action index from that episode that \
carries or appears with the quoted proof.
When not VERIFIED, set evidence_quote and best_action_index to null.

Do not invent text or indices.\
"""


class EpisodeVerifyVerdict(str, Enum):
    verified = "VERIFIED"
    failed = "FAILED"
    insufficient_evidence = "INSUFFICIENT_EVIDENCE"


class EpisodeVerifyDecision(BaseModel):
    """Structured LLM answer for one episode-verification candidate.

    ``evidence_quote`` is the grounding handle Python validates: it must be an
    exact substring of episode-visible text, not a paraphrase of the assertion.
    """

    model_config = ConfigDict(extra="forbid")

    verdict: EpisodeVerifyVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_quote: str | None = Field(
        default=None,
        description=(
            "Exact substring from episode visible_texts or screen description "
            "that proves the state"
        ),
    )
    best_action_index: int | None = Field(default=None, ge=1)
    rationale: str = Field(min_length=1)


def _format_episode_block(episode: ScreenEpisode) -> str:
    texts = "; ".join(repr(text) for text in episode.visible_texts) or "(none)"
    actions = "; ".join(episode.actions_summary) or "(none)"
    return (
        f"episode_id={episode.episode_id} "
        f"actions=[{episode.start_index}..{episode.end_index}] "
        f"ms=[{episode.start_ms}..{episode.end_ms}] "
        f"ambient={episode.is_ambient_system}\n"
        f"screen_summary={episode.screen_summary!r}\n"
        f"visible_texts={texts}\n"
        f"actions_summary={actions}"
    )


def build_episode_verify_user_content(
    step: ExpectedStep,
    episodes: list[ScreenEpisode],
) -> str:
    parts = [
        "Expected step:",
        f"index={step.index}",
        f"action={step.action.value}",
        f"target={step.target!r}",
        f"assertion={step.assertion!r}",
        f"raw_text={step.raw_text!r}",
        "",
        "Candidate episodes (choose at most one proving episode):",
    ]
    for episode in episodes:
        parts.append(_format_episode_block(episode))
        parts.append("")
    return "\n".join(parts).strip()


def propose_episode_verification(
    step: ExpectedStep,
    episodes: list[ScreenEpisode],
    *,
    openai_client: instructor.Instructor | None = None,
    model: str = EPISODE_VERIFY_MODEL,
    on_retry: RetryCallback | None = None,
) -> EpisodeVerifyDecision:
    """Ask the assist model to judge one step against candidate episodes."""
    if not episodes:
        raise ValueError("propose_episode_verification requires candidate episodes")
    if openai_client is None:
        openai_client = make_openai_client()
    kwargs: dict = {
        "model": model,
        "response_model": EpisodeVerifyDecision,
        "messages": [
            {"role": "system", "content": _EPISODE_SYSTEM},
            {
                "role": "user",
                "content": build_episode_verify_user_content(step, episodes),
            },
        ],
    }
    if _TEMPERATURE is not None:
        kwargs["temperature"] = _TEMPERATURE
    response = call_with_retry(
        lambda: openai_client.chat.completions.create(**kwargs),
        on_retry=on_retry,
    )
    return EpisodeVerifyDecision.model_validate(response)
