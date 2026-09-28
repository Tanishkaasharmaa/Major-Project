import asyncio, os
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

async def main():
    transport = LiveKitTransport(
        url=os.environ["LIVEKIT_URL"],
        token=os.environ["BOT_TOKEN"],
        room_name="phase0-test-room",
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    stt = WhisperSTTService(
        device="cuda",
        settings=WhisperSTTService.Settings(model="large-v3-turbo"),
    )

    llm = OpenAILLMService(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        settings=OpenAILLMService.Settings(
            model="qwen3:4b",
            system_instruction=(
                "You are a voice test bot. Repeat back what the user said "
                "in one short sentence. No lists, no markdown."
            ),
        ),
    )

    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(voice="af_heart", speed=1.0),
    )

    context = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_agg,
        llm,
        tts,
        transport.output(),
        assistant_agg,
    ])

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))
    await PipelineRunner().run(task)

if __name__ == "__main__":
    asyncio.run(main())