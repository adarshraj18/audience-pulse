# Lightweight image for the deployed app. No TensorFlow: inference runs on
# the NumPy forward pass in app/model.py, so the whole image stays small and
# starts fast. Listens on 7860 to match Hugging Face Spaces' Docker SDK
# convention.
FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY model/ model/

EXPOSE 7860
ENV PORT=7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
