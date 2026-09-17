# Fase2.2 completata — 2026-09-12

**Modello solo GPU FP32 + CUDA Graph.** CPU: dati, mappe, supervisione. No offload; riferimento eager stessa GPU.

## Implementato

- Embedding4.096×256 da zero, condiviso con testa output.
- Ingresso:17.937 porte sensoriali, fan-in8, copertura bilanciata256 componenti.
- Lettura:2.241 neuroni motori/discendenti/efferenti;77 gruppi,154 medie potenziale/rate →256.
- Reiniezione separata, stesse porte sensoriali, gate trainabile0.1; pronta per attenzione futura.
- Parametri artificiali: **1.394.262**, inclusi1.048.576 embedding. Core soglia10:2.753.975 parametri; totale4.148.237. Attenzione futura esclusa.
- Tutti166.700 nodi mantenuti. Lettura/ingresso disgiunti; no scorciatoia token→logits.

Scelte/API: [INTERFACCE_TESTO.md](INTERFACCE_TESTO.md). Codice: `fly_interfaces.py`; controlli: `test_interfaces.py`, `verify_phase22.py`. Supervisore esteso per scegliere modulo worker; benchmark precedente preservato.

## Verificato sulla GTX1080

- Cinque test GPU: riferimento sparso forward/gradienti; pooling; gradienti core/interfacce e weight tying/ricaricamento; archi azzerati; mappe deterministiche. Annotazioni su host.
- Toy12 nodi + CNS soglia10: **tre replay ciascuno**, batch2, T8, input/target/feedback diversi. CUDA Graph include forward, CE, backward, Adam.
- Parametri, gradienti, momenti, step Adam vs riferimento GPU. Gradienti finiti/non nulli, aggiornamenti non nulli ogni gruppo.
- CNS: max errore parametri4.77e-7, gradienti2.98e-7. No fallback, no guardia.
- Allocazione GPU max cattura/replay CNS:316.01MiB; riservata450MiB. Max uso GPU totale watchdog852MiB; RAM78%.
- Cleanup: allocato0MiB, riservato0MiB; dopo uscita worker GPU totale628MiB, uguale al preflight.
- Media tre replay CNS44.8ms, **solo microprova interfacce**, con snapshot verifica in memoria. Non throughput LM completo, non stima pretraining, non confronto CUDA Graph/eager.

Problemi risolti: riferimenti autograd allo stream impedivano cattura; eliminati prima. Workspace cuBLAS32.5MiB rimaneva dopo eliminazione tensori; rilasciato via API della build locale, senza modificare librerie.

Evidenze: `results/phase22_toy.json`, `results/phase22_cns_t10.json`, log JSONL. Hash codice/risultati: `results/phase22_manifest.json`. Hash annotazioni, ordine bodyId e mappe nel risultato CNS.

## Limiti

Tre update su fixture sintetiche. No embedding preaddestrato o checkpoint linguistico; nessuna qualità dimostrata. Loss su tre batch diversi non misura curva apprendimento.

Gruppi per classe/lato e blocchi bodyId: convenzione artificiale, non regioni spaziali. Raggiungibilità completa, utilità biologica e alternative pooling da misurare. Porta feedback verificata con vettori sintetici; attenzione assente.

## Fasi

- [x] 1 — dati, tokenizer, protocollo.
- [x] 2.1 — API core/stato.
- [x] 2.2 — embedding, porte, readout; CUDA Graph verificato.
- [ ] 2.3 — MHA4×64, RoPE, cache causale128.
- [ ] 2.4 — LM integrato, CE sequenziale, generazione; cattura CUDA Graph completa.
- [ ] 2.5 — causalità/reset/PAD, gradienti e risorse LM completo.
- [ ] 3 — dimostrare apprendimento.
- [ ] 4 — efficienza e recupero archi.
- [ ] 5 — pretraining.
- [ ] 6 — benchmark e ablazioni.