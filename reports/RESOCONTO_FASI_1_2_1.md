# Resoconto — fasi1 e 2.1

2026-09-12. **Richiesta fatta. Prossima: 2.2. Pretraining LM non iniziato.**

## Fase1 — completata

- SHA256 train/valid locale = SHA256 LFS upstream ufficiale.
- Parser streaming UTF8; confini storie espliciti; dedup per hash whitespace-normalizzato.
- Esclusi train: 205 vuoti,1 duplicato,1 coda incompleta. Valid:1 vuoto, primo frammento, coda incompleta. Sovrapposizioni train/heldout:0. Duplicati semantici non garantiti.
- Validation ufficiale ripartita per hash: development80% / test interno20%. Test non usato per fitting/scelta modello.

| Split | Storie | Token inclusi BOS/EOS |
|---|---:|---:|
| Train | 2.717.493 | 555.504.351 |
| Validation | 22.157 | 4.496.926 |
| Test interno | 5.471 | 1.112.001 |

- Corpus selezionato pretokenizzato: uint16, offset uint64, byte-count uint32. Binari totali **1.155.168.032 byte**. Reader mmap read-only, chiusura esplicita Windows.
- Tokenizer:42.690 storie train fitting; altre10.520 train confronto. Matrice E appresa futura, non creata qui.

| Vocab | Byte/token sul probe | Mediana token/storia | Parametri E futuri, d256 |
|---|---:|---:|---:|
| 2.048 | 3,675 | 193 | 524.288 |
| **4.096 scelto** | **3,974** | **179** | **1.048.576** |
| 8.192 | 4,095 | 174 | 2.097.152 |

- 4k:−7,5% token vs2k; 8k:altro−3,0% vs4k, E doppia. Compromesso; accuracy LM non dimostrata.
- Round-trip: tutte10.520 probe per candidato + edge case Unicode/spazi/speciali. Encoding fast vs standard:256 probe identici. Binari finali:32 round-trip per split.
- Verifica completa: hash binari, offset, BOS/EOS, range ID, conteggi byte. `verification.json`:ok.
- Protocollo congelato:512 storie test,128 validation,64 completamenti; task memoria separati; baseline unigram/bigram/GRU/decoder e GPT-2 storico con revision fissa. Punteggi test non calcolati.
- Dipendenze: tokenizers0.23.1 nella .venv; lock in `requirements-phase1.lock.txt`. PyTorch/CUDA invariati.

Artefatti: [manifest](dataset/prepared_v1/manifest.json), [verifica](dataset/prepared_v1/verification.json), [audit](dataset/prepared_v1/audit.json), [confronto tokenizer](dataset/prepared_v1/tokenizer_comparison.json), [protocollo](PROTOCOLLO_VALUTAZIONE.md).

## Fase2.1 — completata

- `Core.initial_state`, `advance`, `reset_state`, `detach_state` fatti. `advance` dà stato+rate, nessuna loss interna.
- Stato persistente; pre/post componibili con correnti diverse; reset/freeze per slot; TBPTT esplicita; stato ricevuto non mutato.
- `window` storico usa nuova API: loss sintetica/benchmark preservati. Topologia, segni, LIF, surrogato invariati.
- **13 test CPU passati:**6 regressioni nucleo,4 API/gradienti,3 pipeline dati.
- CUDA toy eager:ok. CUDA toy graph:equivalenza multi-update/gradienti/Adam ok.
- CUDA CNS:166.700 nodi,2.753.975 archi,soglia10,B1,L4,T8;3 update, gradienti finiti/non nulli. Picco allocatore ~174MiB; totale GPU campionato884MiB; cleanup allocatore0/reserved0. Non throughput linguistico.
- Guardie rispettate, nessun abort nei run finali. Test CPU/GPU separati; smoke ridotto, nessuna prova nuova B8/L64.

Contratto: [CORE_API.md](CORE_API.md). Risultati: [CPU](results/phase21_cpu_tests.json), [CNS](results/phase21_cns_t10.json), [CUDA Graph toy](results/phase21_toy_graph.json).

## Fasi mancanti

| Fase | Stato | Risultato da costruire |
|---|---|---|
| 1 | **Completata** | Dati/tokenizer/protocollo pronti |
| 2.1 | **Completata** | API nucleo verificata |
| 2.2 | **Prossima** | Embedding, porte sparse, pooling, readout legato |
| 2.3 | Da fare | MHA/RoPE/cache/maschere |
| 2.4 | Da fare | Integrazione LM, CE, generazione |
| 2.5 | Da fare | Test integrati e smoke LM completo |
| 3 | Da fare | Apprendimento/stabilità linguistica |
| 4 | Da fare | Efficienza e recupero archi |
| 5 | Da fare | Pretraining controllato |
| 6 | Da fare | Benchmark/ablazioni/contributo mosca |

**Fase2:1/5 sottofasi fatte.** Storie/token pronti ≠ modello linguistico pronto. Embedding/attenzione/readout ancora assenti.