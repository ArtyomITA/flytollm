# Fase3 — apprendimento e stabilità

**Eseguito fino3.5. Fase3 aperta: apprendimento misurabile; memoria insufficiente, generazione degenerata.**

## 3.1 — Protocollo

Campioni/semi/budget/criteri fissi; train/validation separati; test finale chiuso. CE mascherata, BOS/reset, skip optimizer senza target, carry detached, telemetria e guardie verificati.

## 3.2 — Overfit

32 posizioni target, due storie;360 update Adam. CE8,24→2,02; accuracy19/32=59,375%. Core/interfacce/attenzione aggiornati. **Pipeline impara su campione; generalizzazione non dimostrata.**

## 3.3 — Memoria

| Ritardo | Accuracy validation | Criterio |
|---|---:|---:|
| 1 token | 32,66% | ≥80% |
| 4 token | 9,82% | ≥80% |
| 8 token | 9,375% | ≥80% |

Chance8 simboli:12,5%. Tutti sotto criterio; ritardi4/8 vicini al caso su campione piccolo. Nessuna prova memoria utile a quelle distanze. Stress inferenza256 token finito; cache128 gestita bene. Correttezza cache ≠ capacità appresa.

TBPTT8 tronca gradiente fra finestre; cache128 conserva stati oltre quel tratto. Possibile limite, non causa dimostrata. Budget/LR, interfacce e feedback restano ipotesi da isolare.

## 3.4 — Ottimizzatori

24 run:4 varianti ×3 LR ×2 semi;96 update/run, dati/init comparabili.

| Variante | Miglior LR | CE validation media |
|---|---:|---:|
| Adam | 0,0003 | 5,3124 |
| AdamW | 0,0003 | 5,3084 |
| Muon + AdamW | 0,001 | 5,2344 |
| ROOT fallback + AdamW | 0,001 | 5,2317 |

Muon/ROOT migliorano1,27–1,74% nei singoli semi; criterio≥2% in entrambi non superato. Indizio favorevole, nessuna adozione. ROOT usa coefficienti fallback. Tempi confusi da carico TFT: nessuna conclusione speedup.

## 3.5 — Validation e generazione

64 storie train,8126 target,512 update Adam;179s training. CE validation8,376→5,918 (−29,3%); ultime12 storie escluse da selezione optimizer:8,372→5,910. Nessun NaN/Inf, allarme drift assente.

**Greedy24 su entrambi i prompt fissi: `........................`.** Criterio numerico passato, linguaggio utile assente. CE migliora su contesti reali; generazione deve invece sopravvivere ai propri errori. Le due prove misurano aspetti diversi.

## Decisione

- Architettura invariata; baseline Adam conservata. Nessun pretraining avviato.
- Prossima priorità: diagnosi3.3 + collasso greedy3.5; poi rivalutare passaggio4.
- Primo lavoro utile: distribuzione logits/entropia, frequenza punto nei dati, dipendenza da prompt/stato, confronto controlli semplici. Prima di aumentare training, isolare dove manca informazione.
- Variare TBPTT/budget/LR solo con protocollo esplicito; modifiche architetturali sempre presentate all'utente prima di applicare.
- Calcolo GPU FP32/CUDA Graph; paging autorizzato, guardie attive. Tentativi abortiti conservati e esclusi dai risultati. Worker validi ripuliti; griglia con memoria residente solo fra casi.

## Evidenze

[3.1](RESOCONTO_FASE_3_1.md), [3.2](RESOCONTO_FASE_3_2.md), [3.3](RESOCONTO_FASE_3_3.md), [3.4](RESOCONTO_FASE_3_4.md), [3.5](RESOCONTO_FASE_3_5.md), [griglia CSV](results/phase34_grid.csv), [aggregato](results/phase3_extended_summary.json), [piano/stati](PIANO_FASI.md).