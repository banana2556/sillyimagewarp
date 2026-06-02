from fastapi import FastAPI, HTTPException, Request
import httpx
import base64
import os

app = FastAPI()

# 從環境變數讀取你的 NewAPI 設定（部署到 Render 時再填入）
NEW_API_URL = os.getenv("NEW_API_URL", "https://your-newapi-domain.com/v1/images/generations")
NEW_API_KEY = os.getenv("NEW_API_KEY", "your_newapi_token_here")
NEW_API_MODEL = os.getenv("NEW_API_MODEL", "gpt-4o") # 你在 NewAPI 設定的模型名稱

# ==========================================
# 1. 處理 SillyTavern 的設定讀取 (打招呼，解決 404 錯誤)
# ==========================================
@app.get("/sdapi/v1/options")
async def get_options():
    # 回傳一個空的 JSON，讓 SillyTavern 收到 HTTP 200 OK 即可
    return {}

# ==========================================
# 2. 處理 SillyTavern 獲取模型列表 (偽裝一個假模型給它選)
# ==========================================
@app.get("/sdapi/v1/sd-models")
async def get_models():
    return [
        {
            "title": "NewAPI_Proxy_Model",
            "model_name": "NewAPI_Proxy_Model",
            "hash": "123456",
            "filename": "newapi.safetensors"
        }
    ]

# ==========================================
# 3. 核心生圖邏輯 (將 A1111 格式轉換為 OpenAI 格式)
# ==========================================
@app.post("/sdapi/v1/txt2img")
async def txt2img(request: Request):
    try:
        # 接收 SillyTavern (A1111 格式) 的請求
        payload = await request.json()
        prompt = payload.get("prompt", "")
        
        # 處理負面提示詞 (將其拼接到正面提示詞後方，因為 OpenAI 不支援負面參數)
        negative_prompt = payload.get("negative_prompt", "")
        if negative_prompt:
            prompt += f" \n(Please DO NOT include: {negative_prompt})"

        # 轉換為 OpenAI / DALL-E 的請求格式
        openai_payload = {
            "model": NEW_API_MODEL,
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024"
        }
        headers = {
            "Authorization": f"Bearer {NEW_API_KEY}",
            "Content-Type": "application/json"
        }

        # 發送請求給 NewAPI (使用 httpx 處理非同步請求)
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(NEW_API_URL, json=openai_payload, headers=headers)
            response.raise_for_status() # 如果發生錯誤(如 401 授權失敗)，這裡會拋出異常
            data = response.json()

            # 取得圖片結果
            # OpenAI 格式通常預設回傳圖片 URL，但 A1111 需要的是 Base64 字串
            # 所以我們要把圖片下載下來轉成 Base64
            image_url = data["data"][0]["url"]
            img_response = await client.get(image_url)
            img_response.raise_for_status()
            
            base64_image = base64.b64encode(img_response.content).decode('utf-8')

        # 包裝成 A1111 格式回傳給 SillyTavern
        return {
            "images": [base64_image],
            "parameters": payload,
            "info": "{}"
        }

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 4. Render 健康檢查用的端點
# ==========================================
@app.get("/")
def read_root():
    return {"status": "SillyTavern to NewAPI Middleware is running!"}