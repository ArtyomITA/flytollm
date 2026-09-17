# TinyStories / Spikformer — verifica 2026-09-12

Ricerca senza modifiche architetturali. Decisione successiva approvata e implementata 2.4: split 4+4, corrente post token+feedback. Varianti qui restano proposte.

## Tokenizer

[TinyStories, nota 2](https://arxiv.org/html/2305.07759v2): tokenizer GPT-Neo, top 10.000 token. No caratteri; lingua semplice ≠ token monolettera.

Locale: BPE byte-level 4.096, `dataset/prepared_v1/tokenizer-4096.json`; byte alfabeto base, fusioni in frammenti/parole. Verifica:

`Once | upon | a | time | , | a | little | girl | found | a | beautiful | butterfly | .`

13 token; spazi iniziali omessi per leggibilità. 256 dim embedding, indipendenti da lunghezza testo del token.

Probe esistente, stesso testo 10.520 storie:

| Vocabolario | Token | Byte/token | Parametri embedding |
|---|---:|---:|---:|
| 2.048 | 2.304.748 | 3,675 | 524.288 |
| 4.096 | 2.131.789 | 3,974 | 1.048.576 |
| 8.192 | 2.068.703 | 4,095 | 2.097.152 |

8k taglia token del 2,96% vs 4k, ma raddoppia embedding/testa condivisa. Compressione ≠ accuratezza. 4k resta baseline. Fonte: `dataset/prepared_v1/tokenizer_comparison.json`.

## Spikformer originale

[Paper](https://arxiv.org/pdf/2209.15425): classificazione visiva, patch, Q/K/V a spike, attenzione senza softmax. Energia stimata via costo MAC/AC e firing; no misura tempo su GTX 1080.

[Codice ImageNet verificato](https://github.com/ZK-Zhou/spikformer/blob/621268981827af47aa666c1b5f774de4281d90ad/imagenet/model.py):

- Immagine ripetuta 4 volte; proiezioni conv, BatchNorm, LIF.
- SSA: `Q @ (K.T @ V)`, scala 0.125, LIF/proiezione; norm1/norm2 dichiarate nel blocco ma non usate nel forward verificato.
- Residui SSA e MLP; tau 2, `detach_reset=True`; soglia 0.5 per LIF subito dopo aggregazione attenzione.
- Operatori `@` su tensori: binarietà logica ≠ kernel GPU che salta zeri.

## Valutazione per noi — inferenze progettuali

1. Input ripetuto: precedente per drive persistente; non confronta nostra fase post token+feedback vs solo feedback.
2. SSA è variante, non backend equivalente a softmax. RoPE standard rende Q/K continui e negativi: rappresentazione binaria originaria non si conserva da sola.
3. Query singola vs 128 ricordi: nostra matrice attenzione 1×128 per testa, non 128×128. Ricomputare K.T@V può costare di più. Accumulatore incrementale richiederebbe causalità/eviction e nuova verifica; finestra esatta implica conservare info per rimuovere contributo più vecchio.
4. BatchNorm su asse token potrebbe introdurre dipendenza dal futuro in training; LN sui nostri vettori continui non contraddice il risultato su spike. Non sostituire in automatico.
5. T4, leak e detach_reset: possibili ablazioni costo/stabilità, ma alterano dinamica o gradiente. Testare solo come varianti autorizzate dopo baseline integrata.
6. Residuo attorno al nucleo rischia scorciatoia che indebolisce contributo del connettoma. Non aggiungere come ottimizzazione gratuita presunta.

Priorità proposta: baseline 2.4; profilare core/attenzione2.5; apprendimento 3; ottimizzazioni mirate 4. Nessun vantaggio throughput/accuracy di SSA sul nostro LM finora dimostrato.
