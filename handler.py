import runpod, base64, tempfile, os, torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained(
    "uzair0/Katib-ASR", torch_dtype=torch.bfloat16
).to("cuda")
model.config.forced_decoder_ids = None

def handler(job):
    inp = job.get("input", {})
    audio_b64 = inp.get("audio_base64", "")
    if not audio_b64:
        return {"error": "no audio_base64 provided"}
    tmp_path = None
    try:
        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        speech, sr = sf.read(tmp_path)
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to("cuda", torch.bfloat16)

        forced = processor.get_decoder_prompt_ids(language="pashto", task="transcribe")
        ids = model.generate(input_features, forced_decoder_ids=forced, max_new_tokens=440)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

runpod.serverless.start({"handler": handler})
