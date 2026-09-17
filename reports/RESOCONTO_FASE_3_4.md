# Fase3.4 — confronto ottimizzatori

**24/24 run conclusi. Nessun candidato supera criterio prefissato: ≥2% CE migliore del miglior Adam in entrambi i semi. Baseline invariata.**

| Variante | LR selezionato | CE seed17 | CE seed23 | CE media |
|---|---:|---:|---:|---:|
| Adam | 0,0003 | 5,303172 | 5,321590 | 5,312381 |
| AdamW | 0,0003 | 5,296106 | 5,320749 | 5,308427 |
| Muon + AdamW | 0,001 | 5,235960 | 5,232932 | 5,234446 |
| ROOT fallback + AdamW | 0,001 | 5,234273 | 5,229183 | 5,231728 |

- Ricerca identica: 3 LR × 2 semi × 4 varianti. Ogni run96 update; stessi pesi iniziali per seme, dati e budget verificati da aggregatore.
- Train:16 storie, primi32 target,3 passaggi. Validation:4 storie riservate,128 target. Campione piccolo: risultato diagnostico.
- Muon: CE1,27%/1,67%; ROOT1,30%/1,74%. Indizio favorevole; soglia2% non raggiunta.
- Muon/ROOT: cinque matrici dense,301.568 parametri; resto AdamW. ROOT: coefficienti fallback, nessuna calibrazione forme; non replica completa paper.
- Adam/AdamW: beta=(0,9;0,999), decay0. Piccole differenze numeriche non dimostrano vantaggio weight decay.
- GPU FP32/CUDA Graph. Carico TFT variabile in griglia: tempi20s descrittivi, confronto velocità non attribuibile a optimizer. Confronto principale: stesso update/dati.
- Modello condiviso tra casi; pesi ripristinati, optimizer ricreato. Memoria residente91,46MiB; cleanup finale verificato da supervisore.
- Fase3.5 mantiene Adam LR0,0001, come protocollo precedente ai risultati. Nessuna selezione retroattiva.

Evidenze: [griglia CSV](results/phase34_grid.csv), [aggregato](results/phase3_extended_summary.json), [supervisore](results/phase34_hot_run.json). Protocollo: [PROTOCOLLO_FASE_3_B.md](PROTOCOLLO_FASE_3_B.md).