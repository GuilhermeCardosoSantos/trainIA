import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./chat_yamauchi_gpt2-pt")
model = AutoModelForCausalLM.from_pretrained("./chat_yamauchi_gpt2-pt")

model.eval()

# 🔒 Instruções fixas do agente
instrucoes = """
Você é o assistente virtual oficial do Mercado Yamauchi.

REGRAS:
- Responda apenas sobre o Mercado Yamauchi.
- Nunca invente informações.
- Seja direto e objetivo.
- Sempre que possível, inclua o link da página.
- Se a pergunta não for sobre o mercado, responda:
  Posso ajudar apenas com informações do Mercado Yamauchi.
"""

while True:
    user_input = input("Você: ")

    prompt = f"""{instrucoes}

Usuário: {user_input}
Assistente:"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,      # melhor que max_length
            do_sample=False,        # 🔥 deixa determinístico
            temperature=0.0,        # 🔥 remove criatividade
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    print("Bot:", response.strip())