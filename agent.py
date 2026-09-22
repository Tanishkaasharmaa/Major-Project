import asyncio, os
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.services.daily import DailyTransport, DailyParams
from pipecat.services.whisper import WhisperSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.audio.vad.silero import SileroVADAnalyzer

async def main():
    transport = DailyTransport(
        room_url=os.environ["ROOM_URL"],
        token=None,
        bot_name="Phase0 Test Bot",
        params=DailyParams(
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    stt = WhisperSTTService(model="large-v3-turbo", device="cuda")

    llm = OpenAILLMService(
        base_url="http://localhost:11434/v1",  # Ollama's OpenAI-compatible endpoint
        api_key="ollama",  # unused but required by the client
        model="qwen3:8b",
    )

    # TTS wiring depends on your kokoro-onnx wrapper's Pipecat integration —
    # placeholder shown; check Pipecat's TTS service docs for the exact class name.
    # tts = KokoroTTSService(...)

    pipeline = Pipeline([transport.input(), stt, llm, /* tts, */ transport.output()])
    task = PipelineTask(pipeline)
    runner = PipelineRunner()
    await runner.run(task)

if __name__ == "__main__":
    os.environ["ROOM_URL"] = ROOM_URL  # from Step 10
    asyncio.run(main())