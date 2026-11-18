import io
import logging
import os
import time

import torch
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from diffusers import Flux2KleinPipeline
from diffusers.quantizers import PipelineQuantizationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("klein-server")

app = FastAPI()

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
LOAD_MODE = os.environ.get("KLEIN_LOAD_MODE", "int8").lower()

load_started_at = time.perf_counter()
if LOAD_MODE == "nf4":
    logger.info(
        "Caricamento FLUX.2 Klein 4B: transformer NF4, text encoder FP16, tutto su GPU..."
    )
    quantization_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
        },
        components_to_quantize=["transformer"],
    )
    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="cuda",
    )
elif LOAD_MODE == "int8":
    logger.info("Caricamento FLUX.2 Klein 4B in INT8 direttamente sulla GPU...")
    quantization_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_8bit",
        quant_kwargs={"load_in_8bit": True},
        components_to_quantize=["transformer", "text_encoder"],
    )
    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="cuda",
    )
elif LOAD_MODE == "offload":
    logger.info("Caricamento FLUX.2 Klein 4B con CPU offload...")
    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
else:
    raise RuntimeError(
        f"KLEIN_LOAD_MODE non valido: {LOAD_MODE!r}. "
        "Usare 'nf4', 'int8' o 'offload'."
    )

# Riduce l'uso di VRAM durante la decodifica finale ad alta risoluzione
pipe.vae.enable_slicing()
logger.info(
    "Modello caricato in modalità %s in %.2f secondi",
    LOAD_MODE,
    time.perf_counter() - load_started_at,
)


@app.middleware("http")
async def log_request_time(request: Request, call_next):
    started_at = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "%s %s fallita dopo %.2f secondi",
            request.method,
            request.url.path,
            time.perf_counter() - started_at,
        )
        raise

    logger.info(
        "%s %s -> %s in %.2f secondi",
        request.method,
        request.url.path,
        response.status_code,
        time.perf_counter() - started_at,
    )
    return response


def make_generator(seed: int):
    if seed == -1:
        return None
    return torch.Generator("cuda").manual_seed(seed)


def png_response(image: Image.Image) -> Response:
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    return Response(content=image_bytes.getvalue(), media_type="image/png")


def validate_steps(steps: int) -> None:
    if not 1 <= steps <= 4:
        raise HTTPException(
            status_code=400,
            detail="FLUX.2 Klein è distillato per un massimo di 4 step; usare un valore tra 1 e 4",
        )

@app.post("/img2img")
async def generate_img2img(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    steps: int = Form(4),
    guidance_scale: float = Form(1.0),
    seed: int = Form(-1)
):
    validate_steps(steps)
    logger.info(
        "Parametri img2img | prompt=%r | steps=%d | guidance_scale=%s | seed=%d",
        prompt,
        steps,
        guidance_scale,
        seed,
    )

    input_bytes = await image.read()
    init_image = Image.open(io.BytesIO(input_bytes)).convert("RGB")
    
    generator = make_generator(seed)

    # Esecuzione in modalità inferenza pura (più veloce e meno VRAM)
    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()
    with torch.inference_mode():
        output_image = pipe(
            prompt=prompt,
            image=init_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
    logger.info(
        "Inferenza img2img completata in %.2f secondi | picco VRAM %.2f GiB",
        time.perf_counter() - started_at,
        torch.cuda.max_memory_allocated() / 1024**3,
    )

    return png_response(output_image)


@app.post("/text2img")
async def generate_text2img(
    prompt: str = Form(...),
    width: int = Form(512),
    height: int = Form(512),
    steps: int = Form(4),
    guidance_scale: float = Form(1.0),
    seed: int = Form(-1),
):
    validate_steps(steps)
    if not 64 <= width <= 1024 or width % 16 != 0:
        raise HTTPException(400, "width deve essere tra 64 e 1024 e multiplo di 16")
    if not 64 <= height <= 1024 or height % 16 != 0:
        raise HTTPException(400, "height deve essere tra 64 e 1024 e multiplo di 16")

    logger.info(
        "Parametri text2img | prompt=%r | width=%d | height=%d | "
        "steps=%d | guidance_scale=%s | seed=%d",
        prompt,
        width,
        height,
        steps,
        guidance_scale,
        seed,
    )

    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()
    with torch.inference_mode():
        output_image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=make_generator(seed),
        ).images[0]
    logger.info(
        "Inferenza text2img completata in %.2f secondi | picco VRAM %.2f GiB",
        time.perf_counter() - started_at,
        torch.cuda.max_memory_allocated() / 1024**3,
    )

    return png_response(output_image)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
