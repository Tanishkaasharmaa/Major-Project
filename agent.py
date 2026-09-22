import asyncio
import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask

from pipecat.transports.livekit.transport import (
    LiveKitTransport,
    LiveKitParams,
)

from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.openai.llm import OpenAILLMService


async def main():

    print("Starting Pipecat LiveKit agent...")
    print("Room:", os.environ["LIVEKIT_ROOM"])
    # ---------------------------------------------------------
    # LiveKit transport
    # ---------------------------------------------------------

    transport = LiveKitTransport(
        url=os.environ["LIVEKIT_URL"],
        token=os.environ["LIVEKIT_TOKEN"],
        room_name=os.environ["LIVEKIT_ROOM"],
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    # ---------------------------------------------------------
    # Speech-to-Text
    # ---------------------------------------------------------

    stt = WhisperSTTService(
        model="large-v3-turbo",
        device="cuda",
    )

    # ---------------------------------------------------------
    # LLM
    # Ollama exposes an OpenAI-compatible API
    # ---------------------------------------------------------

    llm = OpenAILLMService(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3:8b",
    )

    # ---------------------------------------------------------
    # Pipeline
    # ---------------------------------------------------------

    pipeline = Pipeline([
        transport.input(),
        stt,
        llm,
        transport.output(),
    ])

    task = PipelineTask(pipeline)

    runner = PipelineRunner()

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())