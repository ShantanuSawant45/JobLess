from fastapi import APIRouter
from app.config import get_settings
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
import redis.asyncio as redis
import httpx

router = APIRouter()

@router.get("/health")
async def health_check():
    settings = get_settings()
    status = {
        "postgres": "unreachable",
        "redis": "unreachable",
        "qdrant": "unreachable"
    }

    # Check Postgres
    try:
        engine = create_async_engine(settings.database_url)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        status["postgres"] = "ok"
    except Exception as e:
        status["postgres"] = f"error: {str(e)}"
    finally:
        try:
            await engine.dispose()
        except:
            pass

    # Check Redis
    try:
        r = redis.from_url(settings.redis_url)
        await r.ping()
        status["redis"] = "ok"
    except Exception as e:
        status["redis"] = f"error: {str(e)}"
    finally:
        try:
            await r.close()
        except:
            pass

    # Check Qdrant
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{settings.qdrant_url}/readyz", timeout=2.0)
            if res.status_code == 200:
                status["qdrant"] = "ok"
            else:
                status["qdrant"] = f"status {res.status_code}"
    except Exception as e:
        status["qdrant"] = f"error: {str(e)}"

    return status
