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
    - `decode`: boolean, whether to decode the output tokens (default: True)
    """
    req_dict = await req.json()
    raw_request = swiftllm.RawRequest(
        prompt = req_dict["prompt"],
        output_len = req_dict["output_len"]
    )
    # Whether to decode the output
    decode_output = req_dict.get("decode", False)

    if req_dict.get("stream", False):
        generator = engine.add_request_and_stream(raw_request)
        async def wrapper():
            output_token_ids = []
            prev_decoded = ""
            
            async for step_output in generator:
                token_id = step_output.token_id
                output_token_ids.append(token_id)
                
                if decode_output:
                    # Only decode new tokens
                    try:
                        # Decode current token individually
                        new_token = await engine.tokenization_engine.decode.remote([token_id], skip_special_tokens=False)
                        # Handle special case: some tokens need context from previous token
                        if not new_token and len(output_token_ids) > 1:
                            # Try decoding last two tokens
                            last_two = await engine.tokenization_engine.decode.remote(
                                output_token_ids[-2:], skip_special_tokens=False
                            )
                            # Extract new content from combined result
                            if last_two and len(last_two) > len(prev_decoded):
                                new_token = last_two[len(prev_decoded):]
                        
                        prev_decoded += new_token
                        yield f"{prev_decoded}\n"
                    except Exception:
                        # Fallback to full decoding if incremental fails
                        decoded = await engine.tokenization_engine.decode.remote(output_token_ids, skip_special_tokens=True)
                        prev_decoded = decoded
                        yield f"{decoded}\n"
                else:
                    # Output token ID without decoding
                    yield f"{token_id}\n"
        
        return fastapi.responses.StreamingResponse(
            wrapper(),
            media_type="text/plain"
        )
    else:
        # TODO Abort the request when the client disconnects
        (_, output_token_ids) = await engine.add_request_and_wait(raw_request)
        
        response_content = {"output_token_ids": output_token_ids}
        
        if decode_output:
            decoded = await engine.tokenization_engine.decode.remote(output_token_ids, skip_special_tokens=True)
            response_content["output"] = decoded
        
        return fastapi.responses.JSONResponse(content=response_content)

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
