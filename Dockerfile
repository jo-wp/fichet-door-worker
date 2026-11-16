# Worker RunPod serverless : remplacement de porte avec Qwen-Image-Edit-2511 (Diffusers).
# Les poids (~58 Go) ne sont PAS dans l'image : ils viennent du cache de modèles RunPod
# (champ "Model" de l'endpoint = Qwen/Qwen-Image-Edit-2511) ou d'un network volume.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY handler.py .
CMD ["python", "-u", "handler.py"]
