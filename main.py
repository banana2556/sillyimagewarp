from fastapi import FastAPI, HTTPException, Request
import httpx
import base64
import re
import os
import time

app = FastAPI()

# ==========================================
# 設定：只需要兩個環境變數
#   API_URL = 你的 API base，例如 https://your-newapi-domain.com/v1
#   API_KEY = 你的金鑰
# 其餘端點路徑都從 base 自動推導
# ==========================================
API_URL = os.getenv("API_URL", "https://your-newapi-domain.com/v1").rstrip("/")
API_KEY = os.getenv("API_KEY", "your_newapi_token_here")

MODELS_ENDPOINT = f"{API_URL}/models"
IMAGES_ENDPOINT = f"{API_URL}/images/generations"
CHAT_ENDPOINT = f"{API_URL}/chat/completions"

# 沒選 VAE 時的預設模式
DEFAULT_MODE = "images"
# ST 沒帶模型時的最後備援（正常情況不會用到）
FALLBACK_MODEL = "gpt-4o"
# 取不到模型清單時，顯示在 ST 下拉的提示文字（讓使用者知道設定有問題）
NO_MODELS_LABEL = "無法取得模型，請檢查設定"
# 取樣方法 / 排程器等假端點統一用這個，避免被誤會成真的有作用
PLACEHOLDER = "（此設定無效，由 API 端決定）"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# 記住 SillyTavern 透過 /options 設定的值（模型、VAE），當作 override_settings 的備援
STORED_OPTIONS = {}

# 進度狀態（時間模擬用）：上游不回真實進度，這裡用「已過時間 / 預估秒數」算個會跑動的百分比
ESTIMATED_SECONDS = 25.0  # 預估一次生圖大概多久（影響進度條速度）
PROGRESS_STEPS = 20       # 假裝有幾個取樣步數，讓 ST 顯示 step x/20
PROGRESS_STATE = {"active": False, "start": 0.0}


# ==========================================
# 1. SillyTavern 設定讀取 / 寫入
# ==========================================
@app.get("/sdapi/v1/options")
async def get_options():
    # 回傳記住的設定，沒有就空的
    return STORED_OPTIONS


@app.post("/sdapi/v1/options")
async def set_options(request: Request):
    # SillyTavern 選模型 / VAE 時會打這裡，把它記下來
    try:
        payload = await request.json()
        if isinstance(payload, dict):
            STORED_OPTIONS.update(payload)
        print(f"[options] 收到設定: {payload}")  # 除錯：ST 設定模型/VAE 時的內容
    except Exception as e:
        print(f"[options] 解析失敗: {e}")
    return {}


# ==========================================
# 2. 模型列表：直接去打你 API 的 /models，回真實清單給 ST
# ==========================================
@app.get("/sdapi/v1/sd-models")
async def get_models():
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(MODELS_ENDPOINT, headers=HEADERS)
            resp.raise_for_status()
            data = resp.json()

        # OpenAI 格式是 {"data": [{"id": "..."}]}，也相容直接給 list 的情況
        raw = data.get("data", data) if isinstance(data, dict) else data
        result = []
        for m in raw:
            mid = m.get("id") if isinstance(m, dict) else str(m)
            if not mid:
                continue
            result.append({
                "title": mid,
                "model_name": mid,
                "hash": None,
                "sha256": None,
                "filename": mid,
                "config": None,
            })
        if result:
            return result
    except Exception as e:
        print(f"[sd-models] 取得模型列表失敗，改用備援假模型: {e}")

    # 取不到就回一個提示用的項目，讓使用者在下拉直接看到設定有問題
    return [{
        "title": NO_MODELS_LABEL,
        "model_name": NO_MODELS_LABEL,
        "hash": None,
        "sha256": None,
        "filename": NO_MODELS_LABEL,
        "config": None,
    }]


# ==========================================
# 3. VAE 端點：拿來當「模式開關」（VAE 本來就是假資料）
#    在 ST 的 VAE 下拉選 mode-chat 就走 chat，否則走 images
# ==========================================
@app.get("/sdapi/v1/sd-vae")
async def get_vae():
    return [
        {"model_name": "mode-images", "filename": "mode-images"},
        {"model_name": "mode-chat", "filename": "mode-chat"},
    ]


# ==========================================
# 4. 其他假資料端點（消掉 ST 啟動時的一堆 404）
# ==========================================
@app.get("/sdapi/v1/samplers")
async def get_samplers():
    return [{"name": PLACEHOLDER, "aliases": [], "options": {}}]


@app.get("/sdapi/v1/schedulers")
async def get_schedulers():
    return [{"name": PLACEHOLDER, "label": PLACEHOLDER}]


@app.get("/sdapi/v1/sd-modules")
async def get_modules():
    return []


@app.get("/sdapi/v1/upscalers")
async def get_upscalers():
    return [{
        "name": PLACEHOLDER,
        "model_name": None,
        "model_path": None,
        "model_url": None,
        "scale": 4,
    }]


@app.get("/sdapi/v1/latent-upscale-modes")
async def get_latent_upscale_modes():
    return [{"name": PLACEHOLDER}]


@app.get("/sdapi/v1/progress")
async def get_progress():
    # 上游不回真實進度，用「已過時間 / 預估秒數」模擬一個會跑動的進度條
    if PROGRESS_STATE["active"]:
        elapsed = time.time() - PROGRESS_STATE["start"]
        progress = min(elapsed / ESTIMATED_SECONDS, 0.95)  # 封頂 95%，等真的生完再到 100%
        eta_relative = max(ESTIMATED_SECONDS - elapsed, 0.0)
        job_count, job, job_no = 1, "generating", 0
        sampling_step = int(progress * PROGRESS_STEPS)
    else:
        progress, eta_relative = 0.0, 0.0
        job_count, job, job_no = 0, "", 0
        sampling_step = 0

    return {
        "progress": progress,
        "eta_relative": eta_relative,
        "state": {
            "skipped": False,
            "interrupted": False,
            "job": job,
            "job_count": job_count,
            "job_timestamp": "0",
            "job_no": job_no,
            "sampling_step": sampling_step,
            "sampling_steps": PROGRESS_STEPS if PROGRESS_STATE["active"] else 0,
        },
        "current_image": None,
        "textinfo": None,
    }


# ==========================================
# 5. 核心生圖
# ==========================================
@app.post("/sdapi/v1/txt2img")
async def txt2img(request: Request):
    try:
        payload = await request.json()
        prompt = payload.get("prompt", "")

        # 把收到的 prompt 印進 log
        print(f"[txt2img] 收到 Prompt: {prompt}")

        # 負面提示詞：OpenAI 不支援，拼到後面
        negative_prompt = payload.get("negative_prompt", "")
        if negative_prompt:
            prompt += f" \n(Please DO NOT include: {negative_prompt})"

        override = payload.get("override_settings") or {}

        # --- 除錯：印出 ST 實際送來的內容，定位模型為何沒帶進來 ---
        print(f"[debug] payload keys = {list(payload.keys())}")
        print(f"[debug] override_settings = {override}")
        print(f"[debug] STORED_OPTIONS = {STORED_OPTIONS}")

        # 決定模型：override > 記住的 > 備援
        model = (
            override.get("sd_model_checkpoint")
            or STORED_OPTIONS.get("sd_model_checkpoint")
            or FALLBACK_MODEL
        )

        # 決定模式：看選的 VAE
        vae = override.get("sd_vae") or STORED_OPTIONS.get("sd_vae") or ""
        mode = "chat" if "chat" in str(vae).lower() else DEFAULT_MODE

        # 尺寸：讀 ST 傳的 width/height，不再寫死
        width = payload.get("width", 1024)
        height = payload.get("height", 1024)
        size = f"{width}x{height}"

        print(f"[txt2img] 模型={model} 模式={mode} 尺寸={size}")

        # 開始：啟動進度模擬
        PROGRESS_STATE["active"] = True
        PROGRESS_STATE["start"] = time.time()
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                if mode == "chat":
                    base64_image = await generate_via_chat(client, model, prompt)
                else:
                    base64_image = await generate_via_images(client, model, prompt, size)
        finally:
            # 結束：不管成功失敗都把進度關掉
            PROGRESS_STATE["active"] = False

        return {
            "images": [base64_image],
            "parameters": payload,
            "info": "{}",
        }

    except Exception as e:
        print(f"[txt2img] 發生錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- images 端點：OpenAI /images/generations ----
async def generate_via_images(client, model, prompt, size):
    openai_payload = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }
    resp = await client.post(IMAGES_ENDPOINT, json=openai_payload, headers=HEADERS)
    resp.raise_for_status()
    item = resp.json()["data"][0]

    if item.get("b64_json"):
        return item["b64_json"]
    if item.get("url"):
        return await download_as_base64(client, item["url"])
    raise ValueError("images 回應裡找不到 b64_json 或 url")


# ---- chat 端點：/chat/completions，從文字回應抽圖 ----
async def generate_via_chat(client, model, prompt):
    chat_payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = await client.post(CHAT_ENDPOINT, json=chat_payload, headers=HEADERS)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    # content 可能是字串，也可能是 OpenAI 多模態的 list
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    return await extract_image_base64(client, content)


# ---- 從一段文字裡抽出圖片，回傳 base64 ----
async def extract_image_base64(client, text):
    text = text or ""

    # 1) base64 data URI：data:image/png;base64,xxxx
    m = re.search(r"data:image/\w+;base64,([A-Za-z0-9+/=]+)", text)
    if m:
        return m.group(1)

    # 2) Markdown 圖片連結：![alt](https://...)
    m = re.search(r"!\[[^\]]*\]\((https?://[^\s)]+)\)", text)
    if m:
        return await download_as_base64(client, m.group(1))

    # 3) 純網址
    m = re.search(r"https?://[^\s)\"']+", text)
    if m:
        return await download_as_base64(client, m.group(0))

    # 4) 整段就是 base64（去掉空白後嘗試解碼）
    stripped = re.sub(r"\s+", "", text)
    if len(stripped) > 100:
        try:
            base64.b64decode(stripped, validate=True)
            return stripped
        except Exception:
            pass

    raise ValueError(f"chat 回應裡抽不到圖片，原始內容: {text[:200]}")


# ---- 下載圖片 URL 並轉 base64 ----
async def download_as_base64(client, url):
    img_resp = await client.get(url)
    img_resp.raise_for_status()
    return base64.b64encode(img_resp.content).decode("utf-8")


# ==========================================
# 6. 健康檢查
# ==========================================
@app.get("/")
def read_root():
    return {"status": "SillyTavern to NewAPI Middleware is running!"}
