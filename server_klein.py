import asyncio
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import torch
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from diffusers import Flux2KleinPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from huggingface_hub.errors import HfHubHTTPError

from lora_manager import LoraManager

print("PyTorch:", torch.__version__)
print("Runtime CUDA:", torch.version.cuda)
print("CUDA disponibile:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print("cuDNN:", torch.backends.cudnn.version())
    free, total = torch.cuda.mem_get_info()
    print("VRAM libera GiB:", free / 1024**3)
    print("VRAM totale GiB:", total / 1024**3)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("klein-server")

app = FastAPI()
inference_lock = asyncio.Lock()

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
LOAD_MODE = os.environ.get("KLEIN_LOAD_MODE", "int8").lower()
lora_manager = LoraManager(MODEL_ID)

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
pipe.vae.enable_tiling()
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


async def read_rgb_image(upload: UploadFile) -> Image.Image:
    try:
        contents = await upload.read()
        with Image.open(io.BytesIO(contents)) as source:
            return source.convert("RGB")
    except Exception as exc:
        raise HTTPException(
            400, f"File {upload.filename or '<senza nome>'!r} non è un'immagine valida"
        ) from exc


def validate_steps(steps: int) -> None:
    if not 1 <= steps <= 4:
        raise HTTPException(
            status_code=400,
            detail="FLUX.2 Klein è distillato per un massimo di 4 step; usare un valore tra 1 e 4",
        )


def parse_lora_payload(raw_payload: str | None) -> dict | None:
    if raw_payload is None or not raw_payload.strip():
        return None
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"Payload lora non è JSON valido: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(400, "Payload lora deve essere un oggetto JSON")

    allowed = {"index", "weight", "parameters"}
    unknown = set(payload) - allowed
    if unknown:
        raise HTTPException(400, f"Campi lora non supportati: {', '.join(sorted(unknown))}")
    index = payload.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise HTTPException(400, "lora.index deve essere un intero non negativo")
    weight = payload.get("weight", 1.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise HTTPException(400, "lora.weight deve essere un numero")
    if not -4.0 <= float(weight) <= 4.0:
        raise HTTPException(400, "lora.weight deve essere compreso tra -4.0 e 4.0")
    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict):
        raise HTTPException(400, "lora.parameters deve essere un oggetto JSON")
    allowed_parameters = {"weight_name", "subfolder", "revision"}
    unknown_parameters = set(parameters) - allowed_parameters
    if unknown_parameters:
        raise HTTPException(
            400,
            "Parametri LoRA non supportati: " + ", ".join(sorted(unknown_parameters)),
        )
    for name, value in parameters.items():
        if value is not None and not isinstance(value, str):
            raise HTTPException(400, f"lora.parameters.{name} deve essere una stringa")
    return {"index": index, "weight": float(weight), "parameters": parameters}


def hub_error(exc: Exception) -> HTTPException:
    logger.exception("Operazione Hugging Face LoRA fallita")
    if isinstance(exc, IndexError):
        return HTTPException(404, str(exc))
    if isinstance(exc, HfHubHTTPError) and exc.response is not None:
        status = 404 if exc.response.status_code == 404 else 502
        return HTTPException(status, f"Hugging Face: {exc}")
    return HTTPException(502, f"Impossibile completare l'operazione LoRA: {exc}")


@app.get("/loras")
async def list_loras(refresh: bool = False):
    try:
        items = await asyncio.to_thread(lora_manager.list_catalog, refresh)
    except Exception as exc:
        raise hub_error(exc) from exc
    return {
        "base_model": MODEL_ID,
        "count": len(items),
        "items": items,
    }


@app.post("/loras/{index}/download")
async def download_lora(index: int, revision: str | None = None):
    try:
        item, local_path = await asyncio.to_thread(
            lora_manager.download, index, revision=revision
        )
    except Exception as exc:
        raise hub_error(exc) from exc
    return {
        "index": item.index,
        "repo_id": item.repo_id,
        "downloaded": True,
        "local_path": str(local_path),
        "revision": revision,
    }


@asynccontextmanager
async def exclusive_inference(endpoint: str):
    if inference_lock.locked():
        logger.warning("Richiesta %s rifiutata: inferenza già in corso", endpoint)
        raise HTTPException(
            status_code=429,
            detail="È già in corso una generazione; riprovare al suo completamento",
            headers={"Retry-After": "1"},
        )

    await inference_lock.acquire()
    try:
        yield
    finally:
        inference_lock.release()


@app.post("/img2img")
async def generate_img2img(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    steps: int = Form(4),
    guidance_scale: float = Form(1.0),
    seed: int = Form(-1),
    lora: str | None = Form(None),
):
    validate_steps(steps)
    lora_payload = parse_lora_payload(lora)
    logger.info(
        "Parametri img2img | prompt=%r | steps=%d | guidance_scale=%s | seed=%d",
        prompt,
        steps,
        guidance_scale,
        seed,
    )

    init_image = await read_rgb_image(image)
    
    generator = make_generator(seed)

    async with exclusive_inference("img2img"):
        started_at = time.perf_counter()

        def run_inference():
            # Il thread lascia libero il server di rifiutare subito richieste duplicate.
            adapter_attempted = False
            try:
                if lora_payload is not None:
                    parameters = lora_payload["parameters"]
                    item, local_path = lora_manager.download(
                        lora_payload["index"], revision=parameters.get("revision")
                    )
                    load_parameters = {
                        name: value
                        for name, value in parameters.items()
                        if name in {"weight_name", "subfolder"} and value is not None
                    }
                    # Anche un caricamento interrotto può aver modificato alcuni layer.
                    adapter_attempted = True
                    pipe.load_lora_weights(
                        str(local_path), adapter_name="request_lora", **load_parameters
                    )
                    pipe.set_adapters(
                        "request_lora", adapter_weights=lora_payload["weight"]
                    )
                    logger.info(
                        "LoRA attivo | index=%d | repo=%s | weight=%s | parameters=%s",
                        item.index,
                        item.repo_id,
                        lora_payload["weight"],
                        parameters,
                    )
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    return pipe(
                        prompt=prompt,
                        image=init_image,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]
            finally:
                if adapter_attempted:
                    pipe.unload_lora_weights()

        try:
            output_image = await asyncio.to_thread(run_inference)
        except (IndexError, HfHubHTTPError) as exc:
            raise hub_error(exc) from exc
    logger.info(
        "Inferenza img2img completata in %.2f secondi | picco VRAM %.2f GiB",
        time.perf_counter() - started_at,
        torch.cuda.max_memory_allocated() / 1024**3,
    )

    return png_response(output_image)


@app.post("/multi-img2img")
async def generate_multi_img2img(
    images: list[UploadFile] = File(...),
    prompt: str = Form(...),
    width: int = Form(512),
    height: int = Form(512),
    steps: int = Form(4),
    guidance_scale: float = Form(1.0),
    seed: int = Form(-1),
    lora: str | None = Form(None),
):
    validate_steps(steps)
    if not 2 <= len(images) <= 4:
        raise HTTPException(400, "Inviare da 2 a 4 immagini nel campo form 'images'")
    if not 64 <= width <= 1024 or width % 16 != 0:
        raise HTTPException(400, "width deve essere tra 64 e 1024 e multiplo di 16")
    if not 64 <= height <= 1024 or height % 16 != 0:
        raise HTTPException(400, "height deve essere tra 64 e 1024 e multiplo di 16")
    lora_payload = parse_lora_payload(lora)
    reference_images = [await read_rgb_image(image) for image in images]

    logger.info(
        "Parametri multi-img2img | images=%d | prompt=%r | width=%d | height=%d | "
        "steps=%d | guidance_scale=%s | seed=%d",
        len(reference_images),
        prompt,
        width,
        height,
        steps,
        guidance_scale,
        seed,
    )

    async with exclusive_inference("multi-img2img"):
        started_at = time.perf_counter()

        def run_inference():
            adapter_attempted = False
            try:
                if lora_payload is not None:
                    parameters = lora_payload["parameters"]
                    item, local_path = lora_manager.download(
                        lora_payload["index"], revision=parameters.get("revision")
                    )
                    load_parameters = {
                        name: value
                        for name, value in parameters.items()
                        if name in {"weight_name", "subfolder"} and value is not None
                    }
                    adapter_attempted = True
                    pipe.load_lora_weights(
                        str(local_path), adapter_name="request_lora", **load_parameters
                    )
                    pipe.set_adapters(
                        "request_lora", adapter_weights=lora_payload["weight"]
                    )
                    logger.info(
                        "LoRA multi-image attivo | index=%d | repo=%s | weight=%s",
                        item.index,
                        item.repo_id,
                        lora_payload["weight"],
                    )

                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    return pipe(
                        prompt=prompt,
                        image=reference_images,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        generator=make_generator(seed),
                    ).images[0]
            finally:
                if adapter_attempted:
                    pipe.unload_lora_weights()

        try:
            output_image = await asyncio.to_thread(run_inference)
        except (IndexError, HfHubHTTPError) as exc:
            raise hub_error(exc) from exc

    logger.info(
        "Inferenza multi-img2img completata in %.2f secondi | picco VRAM %.2f GiB",
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

    async with exclusive_inference("text2img"):
        started_at = time.perf_counter()

        def run_inference():
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                return pipe(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=make_generator(seed),
                ).images[0]

        output_image = await asyncio.to_thread(run_inference)
    logger.info(
        "Inferenza text2img completata in %.2f secondi | picco VRAM %.2f GiB",
        time.perf_counter() - started_at,
        torch.cuda.max_memory_allocated() / 1024**3,
    )

    return png_response(output_image)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
