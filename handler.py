import runpod, base64, tempfile, os, torch
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="uzair0/Katib-ASR",
    torch_dtype=torch.bfloat16,
    device="cuda",
    chunk_length_s=30,
)

def handler(job):
    inp = job.get("input", {})
    audio_b64 = inp.get("audio_base64", "")
    if not audio_b64:
        return {"error": "no audio_base64 provided"}

    # decode base64 -> write to a temp WAV file
    audio_bytes = base64.b64decode(audio_b64)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        result = asr(
            tmp_path,
            generate_kwargs={"language": "pashto", "task": "transcribe"},
        )
        return {"text": result.get("text", "").strip()}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

runpod.serverless.start({"handler": handler})
