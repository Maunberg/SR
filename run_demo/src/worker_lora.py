from pathlib import Path
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from typing import List, Callable, Any
from fastapi.responses import StreamingResponse
from time import sleep
import asyncio
import logging
from .model_lora import SALMONN_mutigpu
import torch
import queue
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Task queue setup
task_queue = queue.PriorityQueue()
worker_thread = None

# Simple FastAPI app
app = FastAPI()

class ModelGeneartionRequest(BaseModel):
    audio: List[List[float]]
    dialog: List[dict]

class Task:
    def __init__(self, priority: int, func: Callable[[], Any], future: asyncio.Future):
        self.priority = priority
        self.func = func
        self.future = future

    def __lt__(self, other):
        # Lower priority number means higher priority
        return self.priority < other.priority

def worker():
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def set_result(future, result, main_loop):
        await asyncio.sleep(0)  # Ensure we're in an async context
        if not future.done():
            main_loop.call_soon_threadsafe(future.set_result, result)

    async def set_exception(future, exc, main_loop):
        await asyncio.sleep(0)  # Ensure we're in an async context
        if not future.done():
            main_loop.call_soon_threadsafe(future.set_exception, exc)
    
    while True:
        try:
            # Use timeout to make queue.get() interruptible
            task = task_queue.get(timeout=10.0)
            if task is None:
                # None is the signal to stop the worker
                logger.info("Worker received stop signal")
                break
                
            try:
                result = task.func()
                # Set the result using call_soon_threadsafe
                loop.run_until_complete(set_result(task.future, result, task.loop))
            except Exception as e:
                logger.error(f"Task execution error: {str(e)}", exc_info=True)
                # Set the exception using call_soon_threadsafe
                loop.run_until_complete(set_exception(task.future, e, task.loop))
            finally:
                task_queue.task_done()
        except queue.Empty:
            # Just continue the loop if timeout occurred
            continue
        except Exception as e:
            logger.error(f"Worker thread error: {str(e)}", exc_info=True)
            # Don't exit the worker thread on general exceptions

logger.info("Worker function defined")

# Model paths
whisper_path = Path("/home/stc/gradio/models/whisper-large-v3-russian/")
beats_path = Path("/home/stc/gradio/models/beats.pt")
vicuna_path = Path("/home/stc/gradio/models/T-lite-instruct-0.1/")
lora_path = Path("/home/stc/gradio/models/ars_last_ckpt/")
connector_path = Path("/home/stc/gradio/models/connector_last_ars_checkpoint.pt")

for path in [whisper_path, beats_path, vicuna_path, lora_path, connector_path]:
    assert path.exists()
logger.info(f"lora_path {lora_path} {lora_path.exists()}")   
logger.info(f"whisper_path {whisper_path.exists()}")   
logger.info(f"beats_path {beats_path.exists()}")   
logger.info(f"vicuna_path {vicuna_path.exists()}") 
logger.info(f"connector_path {connector_path.exists()}")

lora_alpha = 32
low_resource = False

logger.info("Loading model...")
model = SALMONN_mutigpu(
    whisper_path=whisper_path,
    beats_path=beats_path,
    vicuna_path=vicuna_path,
    connector_path=connector_path,
    lora_path=lora_path,
    low_resource=low_resource, 
)

model.eval()
logger.info("Model loaded successfully")

# Start worker thread on module import
logger.info("Starting worker thread...")
worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()
logger.info("Worker thread started successfully")

@app.get("/")
async def read_root():
    return {"message": "This is root!"}

@app.get("/api/version")
async def version():
    return {
        "model_version": "CryFish_241021",
        "worker_version": "0.1.0"
    }

@app.get("/api/is_alive")
async def is_alive():
    return "ok"

@app.get("/api/description")
async def description():
    return {"description": "2 gpu worker with torch backend"}

@app.post("/api/audiochat")
async def audiochat(MGR: ModelGeneartionRequest):
    print("test")
    logger.info(f"recieved request, generating now...")
    # Get the current running event loop
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    
    user_message = MGR.dialog[0]
    user_prompt = user_message["text"]
    user_audio = MGR.audio[0]
    
    prompt = user_prompt
    audio = user_audio
    sr = 16000

    def task_func():
        logger.info(f"generation starts")
        with torch.inference_mode():
            result = model.generate(prompts=[prompt], audios=[audio], srs=[sr])
            logger.info(f"generation ends")
            return result

    # Pass the loop to the task
    task = Task(priority=0, func=task_func, future=future)
    task.loop = loop  # Add the loop as an attribute to the task
    task_queue.put(task)

    try:
        ans, time_dict, input_output = await future
        logger.info(f"finished generation for '{prompt}'")
        logger.info(f"New answer: '{ans}'")
        log = ''
        for key in time_dict:
            log += f'{key}: {time_dict[key]:.3f}; '
        logger.info(log)
        generation = str(ans[0])
        return {
            'wall_time': time_dict,
            'text': generation,
            'audio': None,
            'model_version': '241021',
            'worker_version': '0.1.0'
        }
    except Exception as e:
        logger.error(f"Error in audiochat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# When you run with uvicorn, this will keep running until manually stopped
# Command to run: uvicorn worker_lora:app --host 0.0.0.0 --port 8000