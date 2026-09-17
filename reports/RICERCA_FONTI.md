# FlyTOLLM — fonti tecniche raccolte

Obiettivo: architettura GPT-3-style (embedding + transformer decoder-only + MLA semplificata) su connectome mosca. File traccia fonti per componente.

## Fonte primaria (da seguire per implementazione)

**"Build a Large Language Model (From Scratch)" — Sebastian Raschka**
- PDF: già in `flytollm/` (root)
- Repo companion: `flytollm/sources/LLMs-from-scratch/` (da github.com/rasbt/LLMs-from-scratch)
- Capitoli chiave:
  - `ch04/01_main-chapter-code/` — GPT decoder-only base, embedding token+posizionale, multi-head attention standard
  - `ch04/03_kv-cache/` — KV-cache per inferenza
  - `ch04/04_gqa/` — Grouped Query Attention
  - `ch04/05_mla/` — **Multi-Head Latent Attention semplificata, PyTorch, quello che serve**
  - `ch04/06_swa/`, `07_moe/`, `08_deltanet/`, `09_dsa/`, `10_kv-sharing/` — varianti avanzate opzionali
- Sito libro: https://sebastianraschka.com/llms-from-scratch/
- Pagina MLA: https://sebastianraschka.com/llms-from-scratch/ch04/05_mla/

Fonte passo-passo: copre encoder embedding (token+posizionale), blocchi transformer decoder-only, attention (standard → MLA), pretraining loop, in PyTorch puro leggibile.

## Encoder embedding

- Cap. 2 Raschka: tokenizzazione, vettorizzazione, embedding table + positional embedding
- Base: embedding layer (lookup table) + positional encoding (learned o sinusoidale) sommati prima del primo blocco transformer

## Transformer / architettura GPT-3

- Paper GPT-3: "Language Models are Few-Shot Learners" (Brown et al. 2020)
  - PDF: `flytollm/sources/papers/gpt3_language_models_few_shot_learners.pdf`
  - arxiv: https://arxiv.org/abs/2005.14165
  - Architettura: base GPT-2 (pre-normalization, init modificata, tokenizzazione reversibile), alterna attention densa e sparse a bande locali stile Sparse Transformer. 175B parametri, 96 layer nella full (i modelli piccoli useranno pochi layer, es. 6-12).
- Paper transformer originale: "Attention Is All You Need" (Vaswani et al. 2017)
  - PDF: `flytollm/sources/papers/attention_is_all_you_need.pdf`
  - arxiv: https://arxiv.org/abs/1706.03762
  - GPT usa solo lo stack **decoder** (mascherato, autoregressivo), non l'encoder-decoder completo — "encoder embedding" da noi = solo il layer di embedding in ingresso, non un encoder Transformer separato.

## MLA — Multi-Head Latent Attention

- Paper: DeepSeek-V2 "A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
  - PDF: `flytollm/sources/papers/deepseek_v2_mla.pdf`
  - arxiv: https://arxiv.org/abs/2405.04434
- Idea: comprime Q/K/V in spazio latente a bassa dimensione (d_c << h_n·d_h) prima della KV-cache, poi riproietta alla dimensione originale per l'attention. Riduce banda/memoria KV-cache in decodifica autoregressiva, qualità pari o superiore a MHA (meglio di GQA).
- Implementazione pronta: `flytollm/sources/LLMs-from-scratch/ch04/05_mla/` (PyTorch, sullo stesso scheletro GPT del cap. 4)
- Report recente (opzionale, hardware-focused): DeepSeek-V3 Technical Report — arxiv.org/abs/2412.19437 (non scaricato, solo riferimento)
- Spiegazione divulgativa: https://machinelearningmastery.com/a-gentle-introduction-to-multi-head-latent-attention-mla/

## Decoder (blocco transformer decoder-only)

- Raschka cap. 3-4: coding attention mechanisms + GPT model completo (embedding → N blocchi decoder con masked self-attention → layer norm finale → head output)
- Ogni blocco decoder in `ch04/01_main-chapter-code/`: LayerNorm → Masked Multi-Head Attention (poi sostituibile con MLA) → residual → LayerNorm → FeedForward (GELU) → residual

## Dataset training

**TinyStories** (Eldan & Li, Microsoft Research) — scelto per iniziare
- Sorgente: https://huggingface.co/datasets/roneneldan/TinyStories
- Versione: **V2 (generata con GPT-4)**, qualità superiore alla V1, consigliata dall'autore
- File in `flytollm/dataset/`:
  - `TinyStoriesV2-GPT4-train.txt` (~2.23 GB)
  - `TinyStoriesV2-GPT4-valid.txt` (~22.5 MB)
- Perché adatto: vocabolario ristretto (livello bambino 3-4 anni), frasi semplici, coerenza narrativa base, zero sintassi complessa — "senso comune inglese senza grandi capacità di sentenza" per un primo modello piccolo.
- Paper (non scaricato): "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?" — arxiv.org/abs/2305.07759

## Struttura cartella flytollm/

```
flytollm/
├── Build a Large Language Model (From Scratch) (Raschka).pdf   [libro, fonte primaria]
├── RICERCA_FONTI.md                                             [questo file]
├── sources/
│   ├── LLMs-from-scratch/        [repo companion clonato, codice PyTorch]
│   └── papers/
│       ├── gpt3_language_models_few_shot_learners.pdf
│       ├── deepseek_v2_mla.pdf
│       └── attention_is_all_you_need.pdf
└── dataset/
    ├── TinyStoriesV2-GPT4-train.txt
    └── TinyStoriesV2-GPT4-valid.txt
```

## Prossimo step (non ancora fatto)

Collegare architettura connectome mosca (vedi memoria progetto separata) alla struttura GPT decoder+MLA sopra — decidere come il connectome entra (es. inizializzazione pesi, struttura di sparsità nei layer, o modulo separato).
