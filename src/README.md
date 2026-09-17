# Indice del codice

Copia dei sorgenti del progetto (17 settembre 2026, sera). Percorsi relativi alla radice del progetto: i dati vanno in `dataset/`, i risultati in `results/`.

## Modello
- `fly_graph.py` — carica il connettoma male-cns (annotazioni, neurotrasmettitori, pesi), costruisce e mette in cache il grafo a soglia (nodi, archi, conteggi, segni, sensoriali).
- `fly_core.py` — nucleo LIF ricorrente su tutto il grafo: propagazione sparsa (gather + index_add), reset alla soglia, gradiente surrogato.
- `fly_core_fast.py` — core veloci: gather a chunk, FP16, CSR, kernel CUDA fuso in cupy (`FusedCore`, `FusedVariantCore`).
- `fly_core_variants.py` — varianti della dinamica, combinabili con `+`: `apl` (k-WTA sulle Kenyon cell), `graded_ol` (lobo ottico graduato), `tau_type` (leak per tipo cellulare), `reversal` (sinapsi a conduttanza), `bias_type` (riposo per tipo).
- `fly_interfaces.py` — porte d'ingresso (embedding → corrente nei neuroni porta, fan-in 8), lettura (neuroni discendenti/motori → gruppi → proiezione), gate del feedback.
- `fly_attention.py` — attention causale esterna (4 teste, cache 128, RoPE).
- `fly_lm.py` — il language model: sottopassi pre/post attention, stato ricorrente, loss, generazione.
- `fly_rewire.py` — null model e varianti del grafo: rimescolamento a gradi conservati, configuration model, Erdős-Rényi, entro superclasse, pesi/segni permutati, segni morbidi, soglia relativa (`relthr`, `relthr5`), monoammine negative, riordino dei nodi.
- `optimizer_variants.py` — Muon / ROOT diagnostici e `HybridMuon` (Muon sulle matrici dense, Adam sul resto, catturabile nel CUDA Graph).
- `phase3_t45.py` — motore di training con cattura CUDA Graph e verifica di equivalenza eager/graph (`FairCapture`).
- `phase3_variants.py` — varianti della testa (H1 separata).
- `text_dataset.py`, `prepare_text.py`, `lm_io.py`, `lm_checkpointing.py` — dati TinyStories, tokenizer BPE 4096, I/O e checkpoint.

## Training e controlli
- `pretrain_resumable.py` — pretraining con ripresa, telemetria, curve DEV, checkpoint (usato dal modello principale e dai controlli).
- `pretrain_control.py` — trainer dei controlli: grafo (`--rewire-kind`), porte (`--ports anatomical|random_matched|olfactory|visual|auditory|shortpath`), lettura (`--readout chunks|anatomical|fru|hub|cx`), dinamica (`--core-variant`), sottopassi (`--pre-steps/--post-steps`), scala dei pesi, lr dedicato per i parametri per tipo, Muon, core veloce (`--fast-mode fused`).
- `pretrain_night_queue.py`, `bench_runtime.py` — coda con supervisore (guardie RAM/VRAM/timeout, watchdog), pause tra i run.
- `baseline_lm.py`, `phase6_baselines_plus.py` — GRU, Transformer, n-grammi a pari dati.
- `pause_runs.py` — pausa/ripresa vera dei run (sospensione dei processi).

## Suite di test (fase 6-7)
- `phase6_controls_queue.py`, `phase6_rewired_8000_runner.py`, `phase6b_queue.py` — controlli di fase 6 (rimescolato, configuration model, seed).
- `phase6c_queue.py`, `phase6d_queue.py`, `phase6e_queue.py`, `phase6f_promote.py` — varianti a 2000 update (null, porte, lettura, dinamica), regioni specializzate, rerun, promozioni a 8000.
- `phase6c_inference.py` — suite di inferenza sui checkpoint: profilo KV, attività, lesioni per popolazione e rich club, rumore, sonde, raggio spettrale, persistenza.
- `phase6c_reachability.py`, `phase6d_hops.py` — distanze in sinapsi porte → lettura.
- `phase6c_microbench.py`, `phase6s_speed.py` — micro-benchmark e suite di velocità.
- `phase6_state_probe.py`, `phase6c_summary.py`, `phase6_record.py` — sonda lineare sullo stato, riepiloghi, registrazione automatica dei risultati nel file della suite.
- `phase7_queue.py` … `phase7i_queue.py` — fase 7: baseline standard, leve (sottopassi, porte corte, archi, eccitabilità, conduttanza, monoammine, scala, Muon), promozioni a 8000.

## Fase 8 (corti sul candidato per il default: standard + Muon 1e-3)
- `fly_lm_variants.py` — `LoopedFlyLM`: attention letta più volte per token dentro il giro dei sottopassi (cache per lettura, identificativo di passo, input injection per concatenazione), ablazione dell'iniezione del token, attention spenta, schedule di profondità durante il training con un solo CUDA Graph (sottopassi congelati + contatore su GPU). Con una lettura riproduce `FlyLM.step`.
- `fly_core_variants.py` (aggiornato) — flag nuovi: `cond_ports` (porte a conduttanza), `gain_group` (guadagno per superclasse), `soft_gw` (gradiente del peso allargato ai presinaptici vicini alla soglia), `homeo` (omeostasi di risveglio per tipo cellulare, senza gradiente), `arousal` (eccitabilità globale schedulata, a impulsi o a onda).
- `pretrain_control.py` (aggiornato) — opzioni `--attn-reads --attn-kv --attn-step-id --attn-inject --token-injection --depth-schedule --core-lr --freeze-core --ports-seed --ports-count --homeo --arousal`.
- `phase8_selftest.py` — self-test CPU su un grafo minuscolo: equivalenza col modello base, gradiente finito su ogni parametro, identità all'init delle varianti, omeostasi, arousal, schedule di profondità.
- `phase8a_queue.py`, `phase8b_queue.py` — code dei corti (la 8b è riprendibile: salta le run già fatte).
- `phase8_cpu_analyses.py` — raggio spettrale all'init (con la correzione dell'oscillazione di periodo 2) e raggiungibilità con segno porte → lettura.
- `phase8_coverage.py` — quanto cervello impara: variazione relativa dei pesi sinaptici, archi mossi, scarti di soglia, leak.
- `phase8_c15_revert_weights.py` — CE di un checkpoint con le sinapsi rimesse ai valori iniziali.
- `phase8_n0_diagnostic.py`, `phase8_depth_transfer.py` — cosa cambia fra letture ipotetiche dell'attention; trasferimento fra profondità a inferenza.

## Fasi precedenti
- `phase3_*`, `phase35_*`, `verify_*`, `test_*`, `summarize_*`, `bench_throughput.py`, `train_diagnostic.py`, `graph_specialization_probe.py`, `analyze_graph_specialization.py`, `anatomy_fanout_tests.py`, `audit_step0.py`, `generate_phase35.py` — diagnostica, verifiche e test unitari delle fasi 2-3 (vedi `reports/PIANO_FASI.md`).
