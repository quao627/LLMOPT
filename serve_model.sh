python -m vllm.entrypoints.openai.api_server \
          --model /orcd/scratch/seedfund/001/multimodal/qua/huggingface/hub/LLMOPT-Qwen2.5-14B \
            --served-model-name LLMOPT-Qwen2.5-14B \
              --host 0.0.0.0 \
                --port 8000 \
                  --trust-remote-code