# Revisione ricerca — 2026-09-12

Oggetto: `RICERCA_OTTIMIZZAZIONI_ARCHITETTURA.md`. Esito: **utile; adozione selettiva in specifica v0.2**. No cambio codice/run/installazione.

## Confermato

- MHA4×64, RoPE, memoria128, BPE4k/d256, weight tying, FP32, grafo centrale. Ricerca no dimostra architettura alternativa migliore.
- Contabilità KV corretta: `2×8×128×256×4 = 2 MiB`, esclusi intermedi/autograd. GQA no priorità memoria qui.
- Attivazioni: potenziale FP32 per sottopasso B8/N166700/L64/T8 ≈2605 MiB; spike bool ≈651 MiB. Componenti, no previsione esatta picco LM.
- Decay raw: deduzione corretta da `softplus(raw)` locale e aggiornamento AdamW. Per raw<0, decay verso zero aumenta magnitudine. Piano: decay zero iniziale; eventuale prior esplicito.
- CSR backward: confermato doc PyTorch2.14 **e docstring build locale** `torch/sparse/__init__.py`. Supporto documentato, no benchmark CUDA locale. Esperimento accettato; migrazione no approvata auto.
- SDPA: allineamento causale non quadrato richiede cura; cache solo passato usa maschera validità e `is_causal=False`. Rafforza vincolo già presente.
- Backward locale `grad_s` dipende da pesi/gradiente uscita anche se spike=0: saltare sorgenti silenti altera estimatore. No ottimizzazione event-driven ingenua.
- Checkpoint/cache funzionali, correnti per token, reset slot e normalizzazione gradient accumulation: requisiti adottati.

## Precisazioni al report

- **CSR non vittoria garantita:** builtin autograd può salvare input denso FP32; custom attuale salva spike bool. Storage forse quadruplica; verificare storage reale/alias, workspace, conversioni, trasposta, tempo completo. No tensore batch×archi integrale nel codice; osservare allocazioni backend.
- **Pooling:** somma/√n no recupera info persa da aggregazione. Gruppi uguali = riscalare media; LayerNorm successiva può quasi annullare differenza (epsilon escluso). Gruppi/feature diversi cambiano confronto. Ipotesi da testare, no fix accertato.
- **Gate zero:** blocca inizialmente gradiente ramo attenzione a monte, non necessariamente del gate stesso; può sbloccarsi aggiornandolo. Preferenza gate piccolo non nullo = scelta avvio, no prova che zero impedisca apprendimento.
- **Cache2MiB:** sola capacità K/V; cache funzionale con copie/slice trattenute e autograd può costare di più. Evitare concatenazioni crescenti non controllate.
- **8-bit optimizer:** qui non verificata compatibilità binaria specifica; beneficio memoria stimato modesto vs attivazioni. Rinviato, no installazione.
- **Copertura ricerca:** ampia, non esaustiva; nessuno speedup/accuratezza accertato, nessuna conferma che tutti gli archi entrino nel training.

## Decisione

Adottati: correzioni optimizer/autograd, diagnostica, ordine profilo→sparse/checkpoint→più archi, esperimenti tokenizer/porte controllati.
Rinviati: cambi memoria attenzione, SNN alternativi, quantizzazione, custom kernel. Aggiornamento: CPU offload escluso; CUDA Graph richiesto già in fase2.2, poi da verificare su LM completo.
Prossimo: implementazione minima corretta del LM secondo v0.2; no training lungo prima di verifica apprendimento e risorse.

## Fonti ricontrollate

- [CSR backward PyTorch2.14](https://docs.pytorch.org/docs/2.14/generated/torch.sparse.mm.html)
- [AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)
- [SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
- Codice locale: `fly_core.py`; contabilità derivata da dimensioni nel piano e confronto con `RISULTATI_BENCHMARK.md`.
