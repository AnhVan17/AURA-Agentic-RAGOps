import time
import logging
import functools
from app.settings import APPSETTINGS

logging.basicConfig(level=APPSETTINGS.log_level)
logger = logging.getLogger("telemetry")


def timeit_stage(stage_name: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = (time.perf_counter() - t0) * 1000
                logger.info(f"stage={stage_name} ms={dt:.1f}")
        return wrapper
    return deco
