import asyncio, os
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.openai.llm import OpenAILLMServicefrom pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.audio.vad.silero import SileroVADAnalyzer

async def main():
    transport = LiveKitTransport(
        url=os.environ["LIVEKIT_URL"],
        token=os.environ["BOT_TOKEN"],
        room_name="phase0-test-room",
        params=LiveKitParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            # audio_out_sample_rate left unset (None) — Pipecat should
            # negotiate this from the TTS service's own sample_rate,
            # so no manual matching needed unless we hit a mismatch.
        ),
    )

    stt = WhisperSTTService(model="large-v3-turbo", device="cuda")

    llm = OpenAILLMService(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3:8b",
    )

    tts = KokoroTTSService(
        # model_path / voices_path left as None → auto-downloads on first use
        settings=KokoroTTSService.Settings(
            voice="af_heart",  # PLACEHOLDER — confirm real voice IDs via help(KokoroTTSService.Settings)
            speed=1.0,
        ),
    )

    pipeline = Pipeline([transport.input(), stt, llm, tts, transport.output()])
    task = PipelineTask(pipeline)
    runner = PipelineRunner()
    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())