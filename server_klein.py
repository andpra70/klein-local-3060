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
from diffusers import AutoPipelineForImage2Image, Flux2KleinPipeline, FluxPipeline
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

BACKEND = os.environ.get("KLEIN_BACKEND", "klein").lower()
if BACKEND not in {"klein", "flux1"}:
    raise RuntimeError("KLEIN_BACKEND deve essere 'klein' oppure 'flux1'")
DEFAULT_MODEL_IDS = {
    "klein": "black-forest-labs/FLUX.2-klein-4B",
    "flux1": "black-forest-labs/FLUX.1-dev",
}
MODEL_ID = os.environ.get("KLEIN_MODEL_ID", DEFAULT_MODEL_IDS[BACKEND])
LOAD_MODE = os.environ.get("KLEIN_LOAD_MODE", "int8").lower()
DEFAULT_STEPS = int(os.environ.get("KLEIN_DEFAULT_STEPS", "28" if BACKEND == "flux1" else "8"))
DEFAULT_GUIDANCE_SCALE = float(
    os.environ.get("KLEIN_DEFAULT_GUIDANCE_SCALE", "3.5" if BACKEND == "flux1" else "1.0")
)
MAX_STEPS = int(os.environ.get("KLEIN_MAX_STEPS", "50"))
if not 1 <= DEFAULT_STEPS <= MAX_STEPS:
    raise RuntimeError(
        "KLEIN_DEFAULT_STEPS deve essere compreso tra 1 e KLEIN_MAX_STEPS"
    )
lora_manager = LoraManager(MODEL_ID)


load_started_at = time.perf_counter()
pipeline_class = Flux2KleinPipeline if BACKEND == "klein" else FluxPipeline
if LOAD_MODE == "nf4":
    logger.info(
        "Caricamento %s: quantizzazione NF4 su GPU...",
        MODEL_ID,
    )
    components_to_quantize = (
        ["transformer"]
        if BACKEND == "klein"
        else ["transformer", "text_encoder_2"]
    )
    quantization_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
        },
        components_to_quantize=components_to_quantize,
    )
    pipe = pipeline_class.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="cuda",
    )
elif LOAD_MODE == "int8":
    logger.info(
        "Caricamento %s in INT8 direttamente sulla GPU...",
        MODEL_ID,
    )
    components_to_quantize = (
        ["transformer", "text_encoder"]
    ) if BACKEND == "klein" else ["transformer", "text_encoder_2"]
    quantization_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_8bit",
        quant_kwargs={"load_in_8bit": True},
        components_to_quantize=components_to_quantize,
    )
    pipe = pipeline_class.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="cuda",
    )
elif LOAD_MODE == "offload":
    logger.info("Caricamento %s con CPU offload...", MODEL_ID)
    pipe = pipeline_class.from_pretrained(
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
img2img_pipe = AutoPipelineForImage2Image.from_pipe(pipe) if BACKEND == "flux1" else pipe
logger.info(
    "Modello %s caricato con backend %s in modalità %s in %.2f secondi",
    MODEL_ID,
    BACKEND,
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
    if not 1 <= steps <= MAX_STEPS:
        raise HTTPException(
            status_code=400,
            detail=f"steps deve essere compreso tra 1 e {MAX_STEPS}",
        )


FLUX1_PRESETS = {
    "adult": [
        {"repo_id": "lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1", "weight": 0.75}
    ],
    "anatomy": [
        {
            "repo_id": "rorito/testSCG-Anatomy-Flux1",
            "weight": 0.8,
            "parameters": {"weight_name": "scg-anatomy-female-v2.safetensors"},
        }
    ],
    "latex": [
        {
            "repo_id": "Muapi/latex-under-clothing-flux1.d",
            "weight": 0.8,
            "parameters": {"weight_name": "latex-under-clothing-flux1.d.safetensors"},
        }
    ],
}
FLUX1_PRESETS["adult-latex"] = FLUX1_PRESETS["adult"] + FLUX1_PRESETS["latex"]


def parse_lora_payload(raw_payload: str | None, preset: str | None = None) -> list[dict]:
    if preset:
        if BACKEND != "flux1":
            raise HTTPException(400, "I preset adult/anatomy/latex richiedono il backend flux1")
        if preset not in FLUX1_PRESETS:
            raise HTTPException(400, f"Preset sconosciuto: {preset!r}")
        if raw_payload and raw_payload.strip():
            raise HTTPException(400, "Specificare preset oppure lora, non entrambi")
        return [dict(item) for item in FLUX1_PRESETS[preset]]
    if raw_payload is None or not raw_payload.strip():
        return []
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"Payload lora non è JSON valido: {exc.msg}") from exc
    payloads = payload if isinstance(payload, list) else [payload]
    if not 1 <= len(payloads) <= 4:
        raise HTTPException(400, "lora deve contenere da uno a quattro adapter")
    parsed = []
    for position, item in enumerate(payloads):
        if not isinstance(item, dict):
            raise HTTPException(400, f"lora[{position}] deve essere un oggetto JSON")
        allowed = {"index", "repo_id", "weight", "parameters"}
        unknown = set(item) - allowed
        if unknown:
            raise HTTPException(400, f"Campi lora non supportati: {', '.join(sorted(unknown))}")
        index = item.get("index")
        repo_id = item.get("repo_id")
        if (index is None) == (repo_id is None):
            raise HTTPException(400, "Ogni LoRA deve specificare uno solo tra index e repo_id")
        if index is not None and (isinstance(index, bool) or not isinstance(index, int) or index < 0):
            raise HTTPException(400, "lora.index deve essere un intero non negativo")
        if repo_id is not None and (not isinstance(repo_id, str) or "/" not in repo_id):
            raise HTTPException(400, "lora.repo_id deve avere forma organizzazione/modello")
        weight = item.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not -4 <= float(weight) <= 4:
            raise HTTPException(400, "lora.weight deve essere un numero tra -4 e 4")
        parameters = item.get("parameters", {})
        allowed_parameters = {"weight_name", "subfolder", "revision"}
        if not isinstance(parameters, dict) or set(parameters) - allowed_parameters:
            raise HTTPException(400, "lora.parameters contiene parametri non supportati")
        if any(value is not None and not isinstance(value, str) for value in parameters.values()):
            raise HTTPException(400, "I valori di lora.parameters devono essere stringhe")
        parsed.append({"index": index, "repo_id": repo_id, "weight": float(weight), "parameters": parameters})
    return parsed


def activate_loras(lora_payloads: list[dict]) -> bool:
    if not lora_payloads:
        return False
    adapter_names = []
    adapter_weights = []
    load_attempted = False
    try:
        for position, payload in enumerate(lora_payloads):
            parameters = payload["parameters"]
            if payload["repo_id"] is not None:
                repo_id = payload["repo_id"]
                declared_bases = lora_manager.declared_base_models(repo_id)
                if declared_bases and MODEL_ID.lower() not in {
                    base.lower() for base in declared_bases
                }:
                    raise ValueError(
                        f"LoRA {repo_id!r} incompatibile con {MODEL_ID!r}; "
                        f"modelli base dichiarati: {', '.join(declared_bases)}"
                    )
                local_path = lora_manager.download_repo(
                    repo_id, revision=parameters.get("revision")
                )
            else:
                item, local_path = lora_manager.download(
                    payload["index"], revision=parameters.get("revision")
                )
                repo_id = item.repo_id
            adapter_name = f"request_lora_{position}"
            load_parameters = {
                name: value
                for name, value in parameters.items()
                if name in {"weight_name", "subfolder"} and value is not None
            }
            load_attempted = True
            pipe.load_lora_weights(str(local_path), adapter_name=adapter_name, **load_parameters)
            adapter_names.append(adapter_name)
            adapter_weights.append(payload["weight"])
            logger.info("LoRA attivo | repo=%s | weight=%s", repo_id, payload["weight"])
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
        return True
    except Exception:
        if load_attempted:
            pipe.unload_lora_weights()
        raise


def hub_error(exc: Exception) -> HTTPException:
    logger.exception("Operazione Hugging Face LoRA fallita")
    if isinstance(exc, IndexError):
        return HTTPException(404, str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(400, str(exc))
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


@app.get("/status")
async def status():
    return {
        "backend": BACKEND,
        "model_id": MODEL_ID,
        "load_mode": LOAD_MODE,
        "default_steps": DEFAULT_STEPS,
        "default_guidance_scale": DEFAULT_GUIDANCE_SCALE,
        "presets": sorted(FLUX1_PRESETS) if BACKEND == "flux1" else [],
        "multi_img2img": BACKEND == "klein",
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
    steps: int = Form(DEFAULT_STEPS),
    guidance_scale: float = Form(DEFAULT_GUIDANCE_SCALE),
    seed: int = Form(-1),
    lora: str | None = Form(None),
    preset: str | None = Form(None),
):
    validate_steps(steps)
    lora_payloads = parse_lora_payload(lora, preset)
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
            adapters_active = False
            try:
                adapters_active = activate_loras(lora_payloads)
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    return img2img_pipe(
                        prompt=prompt,
                        image=init_image,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]
            finally:
                if adapters_active:
                    pipe.unload_lora_weights()

        try:
            output_image = await asyncio.to_thread(run_inference)
        except Exception as exc:
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
    steps: int = Form(DEFAULT_STEPS),
    guidance_scale: float = Form(DEFAULT_GUIDANCE_SCALE),
    seed: int = Form(-1),
    lora: str | None = Form(None),
    preset: str | None = Form(None),
):
    validate_steps(steps)
    if BACKEND != "klein":
        raise HTTPException(501, "multi-img2img è disponibile soltanto con backend klein")
    if not 2 <= len(images) <= 4:
        raise HTTPException(400, "Inviare da 2 a 4 immagini nel campo form 'images'")
    if not 64 <= width <= 1024 or width % 16 != 0:
        raise HTTPException(400, "width deve essere tra 64 e 1024 e multiplo di 16")
    if not 64 <= height <= 1024 or height % 16 != 0:
        raise HTTPException(400, "height deve essere tra 64 e 1024 e multiplo di 16")
    lora_payloads = parse_lora_payload(lora, preset)
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
            adapters_active = False
            try:
                adapters_active = activate_loras(lora_payloads)

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
                if adapters_active:
                    pipe.unload_lora_weights()

        try:
            output_image = await asyncio.to_thread(run_inference)
        except Exception as exc:
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
    steps: int = Form(DEFAULT_STEPS),
    guidance_scale: float = Form(DEFAULT_GUIDANCE_SCALE),
    seed: int = Form(-1),
    lora: str | None = Form(None),
    preset: str | None = Form(None),
):
    validate_steps(steps)
    lora_payloads = parse_lora_payload(lora, preset)
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
            adapters_active = False
            try:
                adapters_active = activate_loras(lora_payloads)
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
            finally:
                if adapters_active:
                    pipe.unload_lora_weights()

        try:
            output_image = await asyncio.to_thread(run_inference)
        except Exception as exc:
            raise hub_error(exc) from exc
    logger.info(
        "Inferenza text2img completata in %.2f secondi | picco VRAM %.2f GiB",
        time.perf_counter() - started_at,
        torch.cuda.max_memory_allocated() / 1024**3,
    )

    return png_response(output_image)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
