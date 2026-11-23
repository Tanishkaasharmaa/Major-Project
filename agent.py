import asyncio, os
import chromadb
from sentence_transformers import SentenceTransformer

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from knowledge_retrieval import KnowledgeRetrievalProcessor, SYSTEM_TEMPLATE

# --- Phase 2 ---
from dialogue import WifiDiagnosisDialogue
from diagnosis_processor import WifiDiagnosisProcessor, TurnClaims
from intent_router import EmbeddingIntentRouter, load_negative_examples

CHROMA_PATH = "/content/drive/MyDrive/voice-agent-mvp/chroma_db"
DISTANCE_THRESHOLD = 0.17   # E5 + cosine, from the 84-query eval

# Replace with the best values printed by test_intent_routing.py
INTENT_MIN_SIM = 0.80
INTENT_MARGIN = 0.02

# Mock student DB (seed it first: python db/seed_students.py). Run from the repo root.
os.environ.setdefault("STUDENT_DB_PATH", "db/students.sqlite")


async def main():
    transport = LiveKitTransport(
        url=os.environ["LIVEKIT_URL"],
        token=os.environ["BOT_TOKEN"],
        room_name="phase0-test-room",
        params=LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
    )

    stt = WhisperSTTService(
        device="cuda",
        settings=WhisperSTTService.Settings(model="large-v3-turbo"),
    )

    llm = OpenAILLMService(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        settings=OpenAILLMService.Settings(model="qwen2.5:7b-instruct"),
    )

    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(voice="af_heart", speed=1.0),
    )

    context = LLMContext(messages=[
        {"role": "system", "content": SYSTEM_TEMPLATE.format(context="")}
    ])
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    embed_model = SentenceTransformer("intfloat/e5-large-v2")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection("wifi_faqs")

    # Phase 2: personal Wi-Fi complaints go to the deterministic account walk
    router = EmbeddingIntentRouter(
        embed_model, load_negative_examples(),
        min_sim=INTENT_MIN_SIM, margin=INTENT_MARGIN,
    )
    dialogue = WifiDiagnosisDialogue(intent_fn=router.is_account_complaint)
    claims = TurnClaims()
    diagnosis = WifiDiagnosisProcessor(context=context, dialogue=dialogue, claims=claims)

    # Phase 1: everything else is answered from the FAQ knowledge base
    retrieval = KnowledgeRetrievalProcessor(
        context=context,
        collection=collection,
        embed_model=embed_model,
        distance_threshold=DISTANCE_THRESHOLD,
        claims=claims,
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_agg,
        diagnosis,      # NEW: claims Wi-Fi account complaints, speaks scripted replies
        retrieval,
        llm,
        tts,
        transport.output(),
        assistant_agg,
    ])

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))
    await PipelineRunner().run(task)

if __name__ == "__main__":
    asyncio.run(main())
