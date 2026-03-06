import random
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# =====================================================================
#  Funções auxiliares para gerar exemplos de forma dinâmica
# =====================================================================

def gerar_saudacoes(n=300):
    """Gera variações realistas e profissionais de saudações"""
    respostas_base = [
        "Olá! Sou o assistente do Mercado Yamauchi. Como posso ajudar?",
        "Olá! Sou o assistente do Mercado Yamauchi. Em que posso ajudar?",
        "Olá! Sou o assistente do Mercado Yamauchi. Como posso ajudá-lo hoje?",
        "Bom dia! Sou o assistente do Mercado Yamauchi. Como posso ajudar?",
        "Bom dia! Sou o assistente do Mercado Yamauchi. Em que posso ser útil?",
        "Boa tarde! Sou o assistente do Mercado Yamauchi. Como posso ajudar?",
        "Boa tarde! Sou o assistente do Mercado Yamauchi. Em que posso ajudá-lo?",
        "Boa noite! Sou o assistente do Mercado Yamauchi. Como posso ajudar?",
        "Olá! Tudo bem por aqui. Sou o assistente do Mercado Yamauchi. Como posso ajudar?",
        "Olá! Estou à disposição. Sou o assistente do Mercado Yamauchi. Como posso ajudar?",
        "Claro! Sou o assistente do Mercado Yamauchi. Como posso ajudá-lo hoje?",
        "Com certeza. Sou o assistente do Mercado Yamauchi. Em que posso ser útil?"
    ]

    inicios = [
        "oi", "olá", "oie", "oii", "oiii", "eae", "e aí", "alô", "alo",
        "bom dia", "bom dia!", "bom diaa", "bom diaaa",
        "boa tarde", "boa tarde!", "boa tardee",
        "boa noite", "boa noite!", "boa noitee",
        "tudo bem", "tudo bem?", "tá bem?", "tudo certo?", "blz?",
        "alguém aí?", "tem alguém?", "tem alguem ai?", "tem alguem?",
        "atendimento", "atendimento por favor",
        "preciso de ajuda", "pode me ajudar?", "me ajuda por favor",
        "gostaria de informações", "quero informação", "preciso de informação"
    ]

    complementos = [
        "", " por favor", " tudo bem", " tudo bem?", " como vai", " como vai?",
        " você aí?", " tá aí?", " pode falar?", " urgente", " agora", " hoje",
        " mercado yamauchi", " yamauchi", " do yamauchi"
    ]

    exemplos = []
    while len(exemplos) < n:
        inicio = random.choice(inicios)
        comp = random.choice(complementos)
        usuario = f"Usuário: {inicio}{comp}"
        resposta = random.choice(respostas_base)
        exemplos.append(f"{usuario}\nAssistente: {resposta}")

    return exemplos


def gerar_exemplos_fixos(categoria, pares):
    """Gera exemplos a partir de uma lista fixa de pares usuário → assistente"""
    return [f"Usuário: {u}\nAssistente: {a}" for u, a in pares]


# =====================================================================
#  Construção do dataset completo
# =====================================================================

todos_textos = []

# 1. Saudações → geradas dinamicamente
todos_textos.extend(gerar_saudacoes(n=400))

# 2. Horários
horarios = [
    ("qual o horário de funcionamento?", "Funcionamos todos os dias das 8h às 20h. Mais detalhes em /horarios"),
    ("que horas abre?", "Abrimos às 8h todos os dias. Mais detalhes em /horarios"),
    ("que horas fecha?", "Fechamos às 20h todos os dias. Mais detalhes em /horarios"),
    ("vocês abrem domingo?", "Sim, funcionamos todos os dias, inclusive domingos, das 8h às 20h. Veja /horarios"),
    ("abre feriado?", "Sim, permanecemos abertos em feriados no mesmo horário: 8h às 20h. Veja /horarios"),
    ("funciona sábado?", "Sim, aberto sábado das 8h às 20h. Mais detalhes em /horarios"),
    ("horario hoje", "Hoje funcionamos das 8h às 20h. Veja todos os horários em /horarios"),
    ("ate que horas voces estao abertos", "Estamos abertos até às 20h todos os dias. Veja /horarios"),
]
todos_textos.extend(gerar_exemplos_fixos("horários", horarios * 8))

# 3. Localização
localizacao = [
    ("onde fica o mercado?", "Estamos localizados no centro de Sorocaba. Mais informações em /contato"),
    ("qual o endereço?", "Nosso endereço completo está disponível em /contato"),
    ("como chegar ai?", "Estamos no centro da cidade. Veja instruções e mapa em /contato"),
    ("tem estacionamento?", "Sim, contamos com estacionamento próximo. Detalhes em /contato"),
    ("cep do mercado", "O CEP e endereço completo estão em /contato"),
]
todos_textos.extend(gerar_exemplos_fixos("localização", localizacao * 10))

# 4. Frutas / Hortifrúti
frutas = [
    ("tem frutas?", "Sim, temos frutas frescas todos os dias. Veja nossa seção em /produtos/frutas"),
    ("vende frutas?", "Sim! Trabalhamos com frutas frescas diariamente. Confira em /produtos/frutas"),
    ("tem banana hoje?", "Sim, temos banana fresca. Veja todos os produtos em /produtos/frutas"),
    ("tem laranja?", "Sim, laranja fresca disponível. Mais informações em /produtos/frutas"),
    ("tem morango?", "Temos morango fresco. Confira nossa oferta em /produtos/frutas"),
    ("tem maçã?", "Sim, maçã vermelha e verde disponíveis. Veja em /produtos/frutas"),
]
todos_textos.extend(gerar_exemplos_fixos("frutas", frutas * 12))

# 5. Fora do escopo (resposta única – repetir bastante ajuda o modelo a aprender o padrão)
off_topic = [
    "quanto é 2+2", "me conta uma piada", "quem ganhou o jogo ontem",
    "qual a capital da frança", "fala sobre politica", "me ajuda com matematica",
    "qual a previsão do tempo", "conta uma historia", "quem é o presidente",
    "qual sua idade", "voce é homem ou mulher"
]
off_topic_exemplos = [
    f"Usuário: {pergunta}\nAssistente: Posso ajudar apenas com informações sobre o Mercado Yamauchi."
    for pergunta in off_topic
] * 20
todos_textos.extend(off_topic_exemplos)

# =====================================================================
#  Criar o Dataset Hugging Face
# =====================================================================

dataset = Dataset.from_dict({"text": todos_textos})
print(f"Dataset criado com {len(dataset)} exemplos.")

# =====================================================================
#  Modelo e tokenização
# =====================================================================

model_name = "pierreguillou/gpt2-small-portuguese"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.tie_word_embeddings = False  # silencia o aviso de tied weights

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset.set_format("torch")

# =====================================================================
#  Data collator & treinamento
# =====================================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False   # causal LM
)

training_args = TrainingArguments(
    output_dir="./results_yamauchi",
    per_device_train_batch_size=4,          # ajuste conforme sua GPU/CPU
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=5e-5,
    warmup_steps=100,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    fp16=True,                              # remova se estiver em CPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Iniciar treinamento
trainer.train()

# Salvar modelo final e tokenizer
trainer.save_model("./chat_yamauchi_gpt2-pt")
tokenizer.save_pretrained("./chat_yamauchi_gpt2-pt")

print("Treinamento concluído. Modelo salvo em ./chat_yamauchi_gpt2-pt")