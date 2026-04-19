from vllm import LLM, SamplingParams
import os

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

class QwenAPI:
    def __init__(self, model_path, tensor_parallel_size=1):
        # ⚠️ Correction principale : réduire max_model_len pour ne pas dépasser la mémoire GPU
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=32768,  # réduit de 32768 → 8192
        )

        sample_params = dict(
            temperature=0.1,
            repetition_penalty=1.1,
            top_k=50,
            top_p=0.98,
            max_tokens=1024
        )

        self.sampling_params = SamplingParams(**sample_params)
        print(f"sampling_params: {sample_params}")

    def get_response(self, system_prompt: str, user_prompt: str):
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        outputs = self.llm.chat(conversation, sampling_params=self.sampling_params, use_tqdm=False)
        response = outputs[0].outputs[0].text
        return response


if __name__ == "__main__":
    llm = QwenAPI("Qwen/Qwen2.5-Coder-32B-Instruct")
    system_prompt = "You are a helpful assistant."
    user_prompt = "Where is the capital of China?"
    response = llm.get_response(system_prompt, user_prompt)
    print(response)
