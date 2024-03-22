import hashlib
import io
from typing import List, Tuple, Union

import aiohttp
import redis
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from nn import NN
from PIL import Image
from starlette.responses import FileResponse

REDIS_CONFIG = {"host": "localhost", "port": 6379, "decode_responses": True}
ONNX_MODEL_PATH = "model.onnx"

r = redis.Redis(**REDIS_CONFIG)
nn = NN(ONNX_MODEL_PATH)

def error(message: str, sha1: str = "", url: str = "") -> dict:
  if url:
    return {"invert": -1, "sha1": sha1, "error": message, "url": url}
  return {"invert": -1, "sha1": sha1, "error": message}

def create_app() -> FastAPI:
  app = FastAPI()
  app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
  )

  @app.get("/")
  async def read_index():
    return FileResponse('index.html')

  @app.post("/api/file")
  async def process_files(files: List[UploadFile]) -> List[dict]:
    results = []
    for file in files:
      content = await file.read()
      if not content:
        results.append(error("No image provided"))
        continue
      if file.content_type not in ["image/jpg", "image/jpeg", "image/png"]:
        results.append(error("Only jpg, jpeg, and png (non-transparent) images are supported"))
        continue
      sha1 = hashlib.sha1(content).hexdigest()
      if (invert := r.get(sha1)) is not None:
        results.append({"invert": int(invert), "sha1": sha1, "error": ""})
        continue
      try:
        image = Image.open(io.BytesIO(content)).convert('RGB')
      except IOError:
        results.append(error("Invalid image format", sha1=sha1))
        continue
      except Exception as e:
        results.append(error("An unexpected error occurred while processing the image", sha1=sha1))
        continue
      invert = nn.pred(image)
      r.set(sha1, invert)
      results.append({"invert": invert, "sha1": sha1, "error": ""})
    return results

  async def fetch_image(url: str) -> Union[dict, Tuple[bytes, str]]:
    try:
      async with aiohttp.ClientSession() as session:
        async with session.get(url, max_redirects=10) as response:
          # Check for non-success status
          if response.status != 200:
            return error("Failed to fetch image from URL")
          content = await response.read()
          content_type = response.headers.get('Content-Type', 'application/octet-stream')
          return content, content_type
    except Exception as e:
      return error(f"Error: An unexpected error occurred while fetching the URL")

  @app.post("/api/url")
  async def process_urls(urls: list[str]) -> List[dict]:
    results = []
    for url in urls:
      res = await fetch_image(url)
      if type(res) == dict:
        results.append(res)
        continue
      content, content_type = res
      if not content:
        results.appned(error("No image provided"))
        continue
      if content_type not in ["image/jpg", "image/jpeg", "image/png"]:
        results.append(error("Only jpg, jpeg, and png (non-transparent) images are supported"))
        continue
      sha1 = hashlib.sha1(content).hexdigest()
      if (invert := r.get(sha1)) is not None:
        results.append({"invert": int(invert), "sha1": sha1, "error": "", "url": url})
        continue
      try:
        image = Image.open(io.BytesIO(content)).convert('RGB')
      except IOError:
        results.append(error("Invalid image format", sha1=sha1, url=url))
        continue
      except Exception as e:
        results.append(error("An unexpected error occurred while processing the image", sha1=sha1, url=url))
        continue
      invert = nn.pred(image)
      r.set(sha1, invert)
      results.append({"invert": invert, "sha1": sha1, "error": "", "url": url})
    return results

  @app.post("/api/sha1")
  async def process_sha1s(sha1s: list[str]) -> List[dict]:
    results = []
    for sha1 in sha1s:
      if (invert := r.get(sha1)) is not None:
        results.append({"invert": int(invert), "sha1": sha1, "error": ""})
      else:
        results.append(error("No image found with the provided SHA1 hash"))
    return results

  return app

app = create_app()
