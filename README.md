# SwiftLLM

A tiny yet powerful LLM inference system tailored for researching purpose.

## Why SwiftLLM

There are so many open source frameworks for LLM, including [HuggingFace Transformers](https://github.com/huggingface/transformers), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), and [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII). Why SwiftLLM?

The reason is that, those frameworks are tailored for **production**, instead of **researching**. They are equipped with numerous features, such as 100+ model supports, various hardward supports, LoRA, quantization, multimodal, prefix caching, beam search, and so on. While being an all-in-one solution for production, their codebase is too big and complex to understand and modify (for example, vLLM has 100k+ lines of code), making it hard to use them for researching purpose. Also, their historical burden is also a problem.

SwiftLLM is designed to be a tiny yet powerful LLM inference system tailored for **researching purpose**. "Tiny" means that it only keeps features that are essential for researching, "powerful" means that it has no compromise on performance, and finally "swift" means that it is easy to understand and modify. While supporting basic features (see the list below) and being able to achieve equivalent performance to vLLM, the codebase of SwiftLLM is less than 5k lines of code, written in Python and Triton (a DSL for writing CUDA kernels), making it easy to read, modify, debug, test, and extend.

## Feature List

Currently, SwiftLLM supports the following features:

- Iterational Scheduling and Selective Batching (proposed in <TODO>)
- PagedAttenton (proposed in <TODO>)
- LLaMA / LLaMA2 / LLaMA3 models
- Piggybacking prefill and decoding (proposed in <TODO>)
- Flash attention (proposed in <TODO>)
- Paged attention v2 (flash decoding)