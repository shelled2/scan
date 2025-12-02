import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_llama3_8b(model_id: str = "mcpotato/42-eicar-street",
                   device: str = None):
    """
    Loads the Llama 3.1 8B Instruct model with PyTorch + Transformers, returns a pipeline.
    """
    # Torch dtype choice (many use bfloat16 for llama 3.1)
    torch_dtype = torch.bfloat16

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load model (Causal LM)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if device is None else device
    )

    # Create a pipeline for text generation
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto" if device is None else device,
        model_kwargs={"torch_dtype": torch_dtype}
    )
    return gen_pipeline

if __name__ == "__main__":
    pipeline_llama = load_llama3_8b()
    prompt = "You are a helpful assistant.\nUser: Hello, how are you?\nAssistant:"
    out = pipeline_llama(prompt, max_new_tokens=128, do_sample=True, temperature=0.7)
    print(out[0]["generated_text"])
