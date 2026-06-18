import runpod, base64, torch
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="uzair0/Katib-ASR",
    torch_dtype=torch.bfloat16,
    device="cuda",
    chunk_length_s=30,
)

def handler(job):
    inp = job["input"]
    audio_b64 = inp.get("audio_base64", "")
    if not audio_b64:
        return {"error": "no audio_base64 provided"}
    audio_bytes = base64.b64decode(audio_b64)
    result = asr(
        audio_bytes,
        generate_kwargs={"language": "pashto", "task": "transcribe"},
    )
    return {"text": result["text"]}

runpod.serverless.start({"handler": handler})
