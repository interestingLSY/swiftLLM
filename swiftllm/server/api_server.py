import traceback
import os

import argparse
import asyncio
import fastapi
import uvicorn

import swiftllm

TIMEOUT_KEEP_ALIVE = 5  # in seconds

app = fastapi.FastAPI()
engine = None

@app.post("/generate")
async def generate(req: fastapi.Request) -> fastapi.Response:
    """
    Generate completion for the request.

    The request should be a JSON object with fields that match the `RawRequest`
    class plus the following fields:
    - `stream`: boolean, whether to stream the output or not
    """
    req_dict = await req.json()
    raw_request = swiftllm.RawRequest(
        prompt = req_dict["prompt"],
        output_len = req_dict["output_len"]
    )

    if req_dict.get("stream", False):
        generator = engine.add_request_and_stream(raw_request)
        async def wrapper():
            async for step_output in generator:
                yield f"{step_output.token_id}\n"
        return fastapi.responses.StreamingResponse(
            wrapper(),
            media_type="text/plain"
        )
    else:
        # TODO Abort the request when the client disconnects
        (_, output_token_ids) = await engine.add_request_and_wait(raw_request)
        return fastapi.responses.JSONResponse(
            content={"output_token_ids": output_token_ids}
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    swiftllm.EngineConfig.add_cli_args(parser)

    args = parser.parse_args()
    args = vars(args)

    host = args.pop("host")
    port = args.pop("port")
    engine = swiftllm.Engine(swiftllm.EngineConfig(**args))

    uvicorn_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)

    async def main_coroutine():
        await engine.initialize()

        uvicorn_task = asyncio.create_task(uvicorn_server.serve())
        engine_task = asyncio.create_task(engine.start_all_event_loops())

        try:
            await engine_task
        except:  # pylint: disable=broad-except
            traceback.print_exc()
            uvicorn_task.cancel()
            os._exit(1) # Kill myself, or it will print tons of errors. Don't know why.
    
    asyncio.run(main_coroutine())
