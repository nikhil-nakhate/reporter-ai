"""
Main FastAPI application for Reporter AI webapp.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.config import DEFAULT_LLM_MODEL
from app.utils import check_qwen_dependencies, create_error_response

from app.services.article_service import ArticleService
from app.services.llm_service import LLMService
from app.services.character_service import CharacterService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Reporter AI", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
character_service = CharacterService()
article_service = ArticleService()
llm_service = LLMService()

# Request models
class GenerateRequest(BaseModel):
    persona: str
    article_url: str
    llm_model: str = DEFAULT_LLM_MODEL

class FetchArticleRequest(BaseModel):
    url: str

# API Routes
@app.get("/")
async def root():
    """Root endpoint - serve the main HTML page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(
            html_path,
            media_type="text/html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {"status": "ok", "message": "Reporter AI Server"}

@app.get("/api/personas")
async def get_personas():
    """Get list of available personas."""
    try:
        personas = character_service.get_available_personas()
        return {"personas": personas}
    except Exception as e:
        logger.error(f"Error getting personas: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/llms")
async def get_llms():
    """Get list of available LLM models."""
    try:
        models = llm_service.get_available_models()
        # Add debug info for Qwen
        qwen_debug = {}
        try:
            from app.services.qwen_adapter import QWEN_AVAILABLE, import_error
            qwen_debug["import_available"] = QWEN_AVAILABLE
            qwen_debug["adapters_initialized"] = len(llm_service.qwen_adapters)
            qwen_debug["import_error"] = str(import_error) if import_error else None
            qwen_debug["dependencies"] = check_qwen_dependencies()
        except Exception as e:
            qwen_debug["import_available"] = False
            qwen_debug["error"] = str(e)
        
        return {
            "models": models,
            "debug": {
                "qwen": qwen_debug
            }
        }
    except Exception as e:
        logger.error(f"Error getting LLMs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/fetch-article")
async def fetch_article(request: FetchArticleRequest):
    """Fetch and extract content from an article URL."""
    try:
        content = await article_service.fetch_article(request.url)
        return {
            "success": True,
            "content": content,
            "url": request.url
        }
    except Exception as e:
        logger.error(f"Error fetching article: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def generate_bulletin(request: GenerateRequest):
    """Generate a news bulletin from an article URL with streaming."""
    try:
        # Validate persona
        if not character_service.persona_exists(request.persona):
            raise HTTPException(status_code=400, detail=f"Persona '{request.persona}' not found")
        
        # Fetch article content
        logger.info(f"Fetching article from: {request.article_url}")
        article_content = await article_service.fetch_article(request.article_url)
        
        # Get character prompts
        character = character_service.get_character(request.persona)
        system_prompt = character.get_full_prompt()
        task_prompt = character.get_task_prompt()
        
        # Create the user prompt
        user_prompt = f"{task_prompt}\n\nArticle content:\n{article_content}"
        
        # Stream the response
        async def generate_stream():
            try:
                async for chunk in llm_service.stream_generate(
                    model_id=request.llm_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                ):
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                error_msg = create_error_response(str(e))
                yield f"data: {error_msg}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating bulletin: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reporter AI Web App")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("Reporter AI Server")
    print(f"Open your browser to: http://{args.host}:{args.port}")
    print("="*50 + "\n")
    
    uvicorn.run(app, host=args.host, port=args.port)

