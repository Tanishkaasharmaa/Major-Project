import asyncio, os
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.services.livekit import LiveKitTransport, LiveKitParams
from pipecat.services.whisper import WhisperSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.audio.vad.silero import SileroVADAnalyzer

async def main():
    transport = LiveKitTransport(
        url=os.environ["LIVEKIT_URL"],
        token=os.environ["BOT_TOKEN"],
        room_name="phase0-test-room",
        params=LiveKitParams(
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    stt = WhisperSTTService(model="large-v3-turbo", device="cuda")

    llm = OpenAILLMService(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3:8b",
    )

    # tts = KokoroTTSService(...)  # check Pipecat's current Kokoro wrapper class name

    pipeline = Pipeline([transport.input(), stt, llm, /* tts, */ transport.output()])
    task = PipelineTask(pipeline)
    runner = PipelineRunner()
    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())