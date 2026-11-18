# Klein image-to-image server

Server HTTP locale basato su FastAPI e sul modello
[`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).
Supporta la modifica di immagini tramite `POST /img2img` e la generazione da
testo tramite `POST /text2img`. Entrambi gli endpoint restituiscono un PNG.

## Requisiti

- Linux x86_64.
- GPU NVIDIA compatibile con CUDA e driver NVIDIA aggiornati.
- Circa 13 GB di VRAM secondo la scheda ufficiale del modello.
- Almeno 30 GB di spazio libero consigliati per ambiente Python e cache del modello.
- Connessione Internet per l'installazione e il primo download del modello.

> **Nota per RTX 3060 12 GB:** il modello supera la VRAM disponibile. Il server
> abilita il CPU offload, mantenendo nella RAM di sistema i componenti non in uso.
> Questo evita il caricamento dell'intera pipeline sulla GPU, ma riduce la velocità.

## 1. Installare Miniconda

Se `conda --version` funziona già, passare alla sezione successiva.

Scaricare e avviare l'installer ufficiale:

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Durante l'installazione accettare l'inizializzazione della shell. Poi chiudere e
riaprire il terminale, oppure eseguire:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

Controllare l'installazione:

```bash
conda --version
```

## 2. Creare l'ambiente Conda

Entrare nella directory del progetto e creare un ambiente con Python 3.11:

```bash
cd /home/andrea/devel/progetti/miei/ai/st/klein
CONDA_SOLVER=classic conda create -n flux-klein python=3.11 pip -y
conda activate flux-klein
```

L'uso di `CONDA_SOLVER=classic` evita l'errore
`libmambapy has no attribute QueryFormat` riscontrato con installazioni Conda in
cui i componenti di `libmamba` non sono allineati.

## 3. Installare PyTorch e Klein

Verificare prima che il driver riconosca la GPU:

```bash
nvidia-smi
```

Installare tutte le dipendenze elencate in `requirements.txt` tramite lo script
fornito:

```bash
./install.sh
```

Lo script aggiorna `pip`, installa PyTorch, Diffusers e le dipendenze del server,
quindi verifica automaticamente che PyTorch possa utilizzare CUDA. Il valore
`CUDA disponibile` deve essere `True`.

Il server usa normalmente la quantizzazione NF4 tramite `bitsandbytes` soltanto
per il transformer. Il text encoder resta in FP16 per preservare la qualità del
text-to-image. Questa combinazione consente ai componenti di rimanere sulla GPU
da 12 GB, evitando il lento trasferimento CPU/GPU a ogni richiesta.

## 4. Avviare il server

Con l'ambiente attivo:

```bash
conda activate flux-klein
cd /home/andrea/devel/progetti/miei/ai/st/klein
python server_klein.py
```

Al primo avvio Diffusers scarica automaticamente il modello da Hugging Face e
lo salva nella cache locale. Il download può richiedere diversi minuti. Agli
avvii successivi viene utilizzata la copia in cache.

Quando il caricamento è terminato, il server ascolta su:

```text
http://127.0.0.1:8000
```

La documentazione interattiva FastAPI è disponibile su:

```text
http://127.0.0.1:8000/docs
```

Lasciare questo terminale aperto durante i test. Per fermare il server premere
`Ctrl+C`.

## 5. Testare il server

Aprire un secondo terminale, entrare nella directory del progetto e assicurarsi
che esista un'immagine chiamata `immagine_partenza.jpg`. Quindi eseguire:

```bash
curl --fail-with-body -X POST "http://127.0.0.1:8000/img2img" \
  -F "image=@immagine_partenza.jpg" \
  -F "prompt=trasforma in stile cyberpunk al neon, alta qualità" \
  -F "steps=4" \
  -F "guidance_scale=1.0" \
  -F "seed=42" \
  --output risultato.png
```

Se la richiesta termina correttamente, il risultato viene scritto in
`risultato.png`. Verificarlo con:

```bash
file risultato.png
```

I parametri `steps`, `guidance_scale` e `seed` sono facoltativi. Un `seed`
uguale a `-1` genera ogni volta un risultato non deterministico.

### Test text-to-image

Con il server avviato, eseguire:

```bash
./test2.sh
```

Il risultato predefinito viene salvato in `risultato_text2img.png`. È possibile
specificare nome del file e prompt:

```bash
./test2.sh output.png "un piccolo robot in una foresta, luce cinematografica"
```

L'endpoint `POST /text2img` accetta `prompt`, `width`, `height`, `steps`,
`guidance_scale` e `seed` come campi form. Larghezza e altezza devono essere
comprese tra 64 e 1024 e multiple di 16.

## Log dei tempi

Il terminale del server mostra il tempo impiegato per caricare il modello, per
la sola inferenza, il picco di VRAM e il tempo complessivo della richiesta HTTP.
Esempio:

```text
Inferenza text2img completata in 18.42 secondi | picco VRAM 9.34 GiB
POST /text2img -> 200 in 18.45 secondi
```

### Modalità di caricamento

La modalità predefinita `nf4` quantizza soltanto il transformer a 4 bit e lascia
il text encoder in FP16:

```bash
./go.sh
```

La vecchia modalità `int8` quantizza anche il text encoder e può produrre
immagini text-to-image corrotte; è mantenuta soltanto per confronto:

```bash
KLEIN_LOAD_MODE=int8 ./go.sh
```

Se la quantizzazione non fosse compatibile con il sistema, è possibile tornare
alla modalità lenta con CPU offload:

```bash
KLEIN_LOAD_MODE=offload ./go.sh
```

La modalità offload richiede più trasferimenti tra RAM e GPU e può quindi
raddoppiare il tempo totale rispetto al solo denoising.

## Risoluzione dei problemi

### Solver `libmamba` non riconosciuto

Creare l'ambiente usando il solver classico:

```bash
CONDA_SOLVER=classic conda create -n flux-klein python=3.11 pip -y
```

Per renderlo il solver predefinito:

```bash
conda config --set solver classic
```

### `CUDA disponibile: False`

Controllare che `nvidia-smi` funzioni e reinstallare la build CUDA di PyTorch.
Il toolkit CUDA di sistema non è normalmente necessario per le wheel PyTorch,
ma il driver NVIDIA deve essere compatibile.

### `CUDA out of memory`

Chiudere gli altri programmi che utilizzano la GPU e controllare la memoria con:

```bash
nvidia-smi
```

La scheda ufficiale indica circa 13 GB di VRAM per FLUX.2 Klein 4B. Il server
usa già `pipe.enable_model_cpu_offload()` per funzionare su una GPU da 12 GB.
L'offload richiede più RAM di sistema ed è più lento del caricamento completo
sulla GPU. Se l'errore persiste, chiudere i processi che usano la GPU e ridurre
le dimensioni dell'immagine di input.

### Il download del modello fallisce

Verificare connessione, spazio disponibile e permessi sulla cache di Hugging
Face. Per modelli che richiedono autenticazione, installare la CLI ed eseguire
il login:

```bash
python -m pip install "huggingface_hub[cli]"
hf auth login
```

## Riferimenti

- [FLUX.2 Klein 4B su Hugging Face](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [Installazione PyTorch](https://pytorch.org/get-started/locally/)
- [Installazione Conda su Linux](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html)
