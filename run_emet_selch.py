import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "./emetselch_gpt2large"  # Fine-tuned model folder
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# For GPT-2 style models with no pad token
tokenizer.pad_token = tokenizer.eos_token

# We'll keep a conversation history list
conversation_history = []

def chat(user_input, history, max_length=1024):
    # Append the user’s line
    history.append(f"User: {user_input}")
    
    # Build the context
    context = "\n".join(history) + "\nEmet-Selch:"

    # Tokenize the context
    input_ids = tokenizer.encode(context, return_tensors="pt")
    # If it's too long, you may want to truncate older history, but omitted here for brevity

    # Generate a response (sampling for creativity/snark)
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.2
    )

    # -------------------------------
    #  NEW: Decode only newly generated tokens
    # -------------------------------
    # input_ids.shape[-1] = length of the input (in tokens)
    new_tokens = output_ids[0][input_ids.shape[-1]:]  # skip the input part
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Store Emet-Selch's reply in the history
    history.append(f"Emet-Selch: {response}")
    return response, history

def run_chatbot():
    global conversation_history
    print("Emet-Selch Chatbot running. Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting Emet-Selch chatbot. Farewell!")
            break
        
        response, conversation_history = chat(user_input, conversation_history, max_length=1024)
        print(f"Emet-Selch: {response}\n")

if __name__ == "__main__":
    run_chatbot()