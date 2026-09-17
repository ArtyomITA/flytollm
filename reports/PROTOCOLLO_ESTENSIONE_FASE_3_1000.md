# Estensione fase3 — 13 settembre

Richiesta: finire prove extra Luna. 200 update non escludono capacità con budget più grande. Fase3 non chiude da sola.

- T0: contratti già ok; ricontrolla solo harness nuovi.
- T1: GRU d4/d8, 1000 update, protocollo T1b uguale; curva scarsa → calibrazione ancora aperta.
- R0: scomponi gradiente embedding condiviso, 8 coppie simboli DEV +8 coppie testo train; zero update.
- R1: pooled154 congelate, 1024 train/1024 DEV; solo norm/proiezione/output-norm originali, Adam0,001, 100 update =102.400 presentazioni. Head4096 tied ferma. Curva ogni10 update.
- R2: ricattura circuito dopo R1; stessi DEV e32 contesti train non usati dal fit R1. Questi 32 nuovi per calibrazione, non dati del tutto nuovi per checkpoint T2.
- P0: annotazioni/raggiungibilità su grafo intero; sensitività testo train. Zero pruning.
- P1/P2/D1/D2/H1 approvati esplicito da utente: canali per classe/lato; pooling77 ordinato classe/sottoclasse/lato; surrogate scala2 vs4; solo bias input0,5 vs0,6; output untied copiato da E (+1.048.576 parametri). Una modifica per run; adozione dopo, su evidenza.
- T5 esteso: L8/L16, d1/d8, 1000 update ciascuno da stesso checkpoint T2 e Adam nuovo; stessi stream estesi T1b, curva200/500/1000. Stesso update ogni16 posizioni. Budget stima ~45–55min totale; confronto a budget e dati uguali, non prova di convergenza finale.
- T6: pilot100k target nuovi, DEV10k/30k/100k, baseline frequenze+bigram, stabilità/generazione; replica se buono, audit64 tenuto per una volta.

GPU FP32/CUDA Graph; CPU I/O/statistiche. Un worker GPU per volta. Paging ok; watchdog RAM/commit/VRAM uguali. Per run1000update limite tempo1200s, heartbeat≤32update; stop se5 update di fila>2s. Modello e risultati storici tenuti. Nessun download/install serve.

Realtime: ricerca Luna fatta a parte; visualizzazione non in questa suite. Controlla unità coordinate soma a parte da unità skeleton; eventi/ownership ring devono bloccare overwrite durante D2H.
