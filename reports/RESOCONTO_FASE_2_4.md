# Fase2.4 completata — 2026-09-12

Scelta approvata: **4+4; corrente token + feedback in fase post**. No altro cambio architettura. LM collega core, interfacce, attenzione;4.410.893 parametri su CNS soglia10.

## Implementato

- `fly_lm.py`: step, sequenze causali, stato core/cache, BOS/reset/PAD, CE mascherata, detach TBPTT.
- `CUDATokenStep`: inferenza+selezione catturate; generazione greedy/campionata, EOS, congelamento slot terminati.
- `lm_io.py`: finestre TinyStories senza attraversare storie; costruzione CNS; salvataggio+ricostruzione modello.
- `verify_phase24.py`: training forward+CE+backward+Adam in CUDA Graph, confronto GPU, generazione, checkpoint sotto guardie.

Contratto: [LM_API.md](LM_API.md).

## Verifiche

- Cinque test GPU: sequenza/token+causalità; reset/PAD/CE; gradienti+detach; checkpoint/confini storie; campionamento/EOS in CUDA Graph.
- Toy: batch2, L4, maschere/BOS variabili; tre replay training. CNS:166.700 nodi,2.753.975 archi, batch1,L4, tre replay su primi4 input token storie train0/1/2.
- Gradienti finiti/non nulli+aggiornamenti in tutti i gruppi, incluse Q/K/V+magnitudini core. Parametri, gradienti, stati Adam confrontati vs riferimento GPU.
- CNS: errore max parametri4,77e-7, gradienti1,49e-7; inferenza logits3,58e-7. Greedy+campionamento temp0,8:8 replay verificati ciascuno, più continuazioni6 token.
- Checkpoint ricaricato: output GPU coincide nelle tolleranze. File `results/phase24_cns_t10.smoke.pt` e toy analogo: **tre update diagnostici, non modelli preaddestrati**.
- Picco allocatore CNS355,52MiB, GPU totale campionata1.318MiB, RAM sistema76%. Cleanup0MiB allocati/riservati; GPU totale pre/post633MiB.
- Replay training CNS ~89–90ms/finestra di4 token. Microprova, non throughput sostenuto né tempo stimato pretraining.

Evidenze: `results/phase24_toy.json`, `results/phase24_cns_t10.json`, log JSONL+hash sorgenti incorporati. No fallback CPU né guardia scattata.

## Ancora da verificare

Qualità linguistica non dimostrata. Smoke con finestre indipendenti; training prolungato con carry post-update, profilo dettagliato, checkpoint attivazioni, eviction/reset su LM completo+config più grandi restano2.5/3/4. Checkpoint attuale salva modello/config; optimizer/RNG/stato per ripresa esatta previsti5.

## Valutazioni aggiunte al piano

| Fase | Valutazioni |
|---|---|
| 2.5 | Costo separato core/attenzione/interfacce/cache; firing, saturazione, rapporto corrente token/feedback, gate, gradienti |
| 3 | Stabilità+uso memoria; eventuale confronto token+memoria/solo memoria dopo discussione |
| 4 | T, leak, detach_reset; SSA se giustificata dal profilo; causalità/normalizzazione/RoPE; BPE8k solo dopo confronto costo/qualità |
| 5 | Congelare+registrare tutte scelte approvate; tokenizer immutabile per run |
| 6 | Benchmark+ablazioni varianti effettivamente sperimentate; tempi hardware misurati |

Inserimento nel piano ≠ autorizzazione a cambiare architettura: prima proposta+domanda all'utente.

| Fase | Stato |
|---|---|
| 1 — Dati/tokenizer/protocollo | ✅ Completata |
| 2.1 — API nucleo/stato | ✅ Completata |
| 2.2 — Embedding/interfacce/readout | ✅ Completata |
| 2.3 — Attenzione/memoria | ✅ Completata |
| 2.4 — Integrazione LM/generazione | ✅ Completata |
| **2.5 — Verifica completa** | **Prossima** |
| 3 — Apprendimento/stabilità | Da fare |
| 4 — Ottimizzazione/più archi | Da fare |
| 5 — Pretraining | Da fare |
| 6 — Benchmark/ablazioni | Da fare |