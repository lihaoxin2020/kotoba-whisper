# %%
from datasets import load_dataset, load_from_disk
import torch

# %%
ds = load_from_disk("output")

# %%
ds = load_from_disk("output-pseudolabel")

# %%
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "kotoba-tech/kotoba-whisper-v1.1"
cache_dir = "/workspace/.cache"
# %%
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, cache_dir=cache_dir
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)
dataset = load_dataset("japanese-asr/ja_asr.reazonspeech_test", cache_dir=cache_dir, split="test")
# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]
# %%
sample = dataset[0]["audio"]
# result = pipe(sample, generate_kwargs={"task": "translate", "language": "ja"})
result = pipe(sample)
print(result["text"])

# %%
whisper_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3", cache_dir=cache_dir)

# %%
