"""
Smoke test: both processors inside a REAL Pipecat pipeline, no models, no GPU.
Run:  python -m pytest test_pipeline_smoke.py -v -s

Checks the integration contract that unit tests can't:
  - a Wi-Fi complaint is answered by a TTSSpeakFrame and the LLM trigger is swallowed
  - the walk's turns are NOT logged to unanswered_queries.log by the retrieval processor
  - an FAQ question passes through to the LLM with a trimmed context
  - the same words twice in a row ("yes", "yes") are both handled
"""
import asyncio
import os
import sys
import warnings

import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "db"))
import seed_students  # noqa: E402

from pipecat.frames.frames import EndFrame, LLMContextFrame, TTSSpeakFrame  # noqa: E402
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineTask  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402

from dialogue import DIGIT_WORDS, WifiDiagnosisDialogue  # noqa: E402
from diagnosis_processor import TurnClaims, WifiDiagnosisProcessor  # noqa: E402
from knowledge_retrieval import SYSTEM_TEMPLATE, KnowledgeRetrievalProcessor  # noqa: E402


class FakeEmbedModel:
    def encode(self, text, normalize_embeddings=True):
        class V(list):
            def tolist(self):
                return list(self)
        return V([0.0, 1.0])


class FakeCollection:
    """Always returns one FAQ passage at distance 0.05 (i.e. a confident match)."""
    def query(self, **kwargs):
        return {"documents": [["Approval usually takes a few working days."]],
                "distances": [[0.05]]}


class Capture(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.frames = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, (TTSSpeakFrame, LLMContextFrame)):
            self.frames.append(frame)
        await self.push_frame(frame, direction)


def words(digits):
    return " ".join(DIGIT_WORDS[d] for d in digits)


async def run_scenario(tmp_path, turns, intent_fn):
    db_path = str(tmp_path / "students.sqlite")
    seed_students.seed(db_path)
    log_dir = str(tmp_path / "logs")

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_TEMPLATE.format(context="")}])
    claims = TurnClaims()
    dialogue = WifiDiagnosisDialogue(intent_fn=intent_fn, db_path=db_path, log_dir=log_dir)
    diag = WifiDiagnosisProcessor(context, dialogue, claims, verbose=False)
    retrieval = KnowledgeRetrievalProcessor(
        context=context, collection=FakeCollection(), embed_model=FakeEmbedModel(),
        distance_threshold=0.17, log_path=str(tmp_path / "unanswered.log"), claims=claims)
    capture = Capture()

    task = PipelineTask(Pipeline([diag, retrieval, capture]))
    runner = PipelineRunner(handle_sigint=False)
    runner_task = asyncio.create_task(runner.run(task))
    await asyncio.sleep(0.3)

    per_turn = []
    for text in turns:
        before = len(capture.frames)
        context.messages.append({"role": "user", "content": text})   # what the aggregator does
        await task.queue_frame(LLMContextFrame(context))
        await asyncio.sleep(0.15)
        per_turn.append(capture.frames[before:])

    await task.queue_frame(EndFrame())
    await runner_task
    return per_turn, context, tmp_path


def spoken(frames):
    return [f.text for f in frames if isinstance(f, TTSSpeakFrame)]


def reached_llm(frames):
    return any(isinstance(f, LLMContextFrame) for f in frames)


def test_walk_is_spoken_by_script_and_never_reaches_llm(tmp_path):
    turns = ["my wifi is not working", words("100016"), "yes"]
    per_turn, _, _ = asyncio.run(run_scenario(tmp_path, turns, lambda t: "wifi" in t))
    for i, frames in enumerate(per_turn):
        assert len(spoken(frames)) == 1, f"turn {i}: expected exactly one scripted reply"
        assert not reached_llm(frames), f"turn {i}: LLM trigger should have been swallowed"
    assert "approved" in spoken(per_turn[2])[0]          # provisioning backlog answer
    assert not (tmp_path / "unanswered.log").exists(), "walk turns must not be logged as unanswered FAQs"


def test_faq_question_passes_through_with_trimmed_context(tmp_path):
    turns = ["how long does approval take"]
    per_turn, context, _ = asyncio.run(run_scenario(tmp_path, turns, lambda t: False))
    assert reached_llm(per_turn[0]) and not spoken(per_turn[0])
    assert [m["role"] for m in context.messages] == ["system", "user"]
    assert "Approval usually takes" in context.messages[0]["content"]


def test_same_word_twice_in_a_row_is_handled_both_times(tmp_path):
    # 100021 = mac_mismatch: confirm "yes", then the device question also answered "yes"
    turns = ["my wifi is not working", words("100021"), "yes", "yes"]
    per_turn, _, _ = asyncio.run(run_scenario(tmp_path, turns, lambda t: "wifi" in t))
    assert "new phone or laptop" in spoken(per_turn[2])[0]
    assert "hardware address" in spoken(per_turn[3])[0]
    assert not any(reached_llm(f) for f in per_turn)


def test_walk_then_faq_in_one_session(tmp_path):
    turns = ["my wifi is not working", words("100001"), "yes", "how long does approval take"]
    per_turn, context, _ = asyncio.run(run_scenario(tmp_path, turns, lambda t: "my wifi" in t))
    assert "haven't applied" in spoken(per_turn[2])[0]
    assert reached_llm(per_turn[3]) and not spoken(per_turn[3])
    assert len(context.messages) == 2
