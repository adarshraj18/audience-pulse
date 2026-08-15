# Lightweight image for the deployed app. No TensorFlow: inference runs on
# the NumPy forward pass in app/model.py, so the whole image stays small and
# starts fast. Binds to $PORT if the host sets one (Render and most PaaS
# providers do), falling back to 7860 (Hugging Face Spaces' Docker convention)
# otherwise.
FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY model/ model/

EXPOSE 7860

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
