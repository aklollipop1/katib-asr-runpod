FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir transformers==4.46.3 tokenizers==0.20.3 accelerate runpod soundfile protobuf sentencepiece

RUN python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='uzair0/Katib-ASR')"

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
