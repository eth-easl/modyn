from fastapi import FastAPI, Request
from llm_score import LLMScore

app = FastAPI()
evaluator = LLMScore(config={"use_api": False, "model": "meta-llama/Llama-2-13b-chat-hf"})


@app.post("/evaluate")
async def evaluate(request: Request):
    data = await request.json()
    scores = evaluator._evaluate_locally(data["prompts"], data["responses"])
    return {"scores": scores}
