import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "microsoft/DialoGPT-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

model.eval()

# 🔒 Regras do agente
system_prompt = """
Você é o assistente virtual oficial do Mercado Yamauchi.

REGRAS IMPORTANTES:
- Responda apenas perguntas relacionadas ao Mercado Yamauchi.
- Se a pergunta não for sobre o mercado, responda:
  "Posso ajudar apenas com informações do Mercado Yamauchi."
- Seja objetivo e responda em português (Brasil).
- Sempre que possível, direcione para uma página do site.
"""

# 📚 Base de conhecimento fixa
base_conhecimento = """
INFORMAÇÕES DO MERCADO YAMAUCHI:

Horário: 8h às 20h todos os dias.
Localização: Centro da cidade.
Frutas: Temos frutas frescas diariamente.
Telefone: (11) 99999-9999
Site: www.mercadoyamauchi.com

Páginas:
Horários → /horarios
Produtos → /produtos
Frutas → /produtos/frutas
Contato → /contato
"""

messages = [
    {"role": "system", "content": system_prompt}
]

while True:
    user_input = input("Você: ")

    if user_input.lower() == "sair":
        break

    # 🔥 Monta prompt estruturado
    structured_input = f"""
Base de conhecimento:
{base_conhecimento}

Pergunta do cliente:
{user_input}

Resposta:
"""

    messages.append({"role": "user", "content": structured_input})

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.5,  # 🔥 mais baixo = mais controlado
            top_p=0.8,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    bot_reply = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("Bot:", bot_reply.strip())

    messages.append({"role": "assistant", "content": bot_reply})