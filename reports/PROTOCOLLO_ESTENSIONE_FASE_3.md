# Estensione fase3 — 13 settembre

Richiesta: completare prove Luna. 200 update no escludono capacità budget maggiore. No chiusura auto fase3.

- T0: contratti già verificati; ricontrollare solo nuovi harness.
- T1: GRU d4/d8,1000 update,protocollo T1b; curva insufficiente → calibrazione aperta.
- R0: scomporre gradiente embedding condiviso,8 coppie simboli DEV +8 coppie testo train; no update.
- R1: pooled154 congelate,1024 train/1024 DEV; solo norm/proiezione/output-norm originali,Adam0,001,100 update =102.400 presentazioni. Head4096 tied immutata. Curva ogni10 update.
- R2: ricatturare circuito dopo R1; stessi DEV e32 contesti train non usati da fit R1. Ultimi nuovi per calibrazione, non dati nuovi per checkpoint T2.
- P0: annotazioni/raggiungibilità su grafo completo; sensitività testo train. No pruning.
- P1/P2/D1/D2/H1 approvati utente: canali per classe/lato; pooling77 ordinato classe/sottoclasse/lato; surrogate scala2 vs4; solo bias input0,5 vs0,6; output untied copiato da E (+1.048.576 parametri). Una modifica/run; adozione poi su evidenza.
- T5 esteso: L8/L16,d1/d8,1000 update ciascuno da stesso checkpoint T2 e Adam nuovo; stream estesi T1b,curva200/500/1000. Update ogni16 posizioni. Budget ~45–55min; confronto a budget/dati identici, non prova convergenza.
- T6: pilot100k nuovi target,DEV10k/30k/100k,baseline frequenze+bigram,stabilità/generazione; replica se promettente,audit64 riservato una volta.

GPU FP32/CUDA Graph; CPU I/O/statistiche. Un worker GPU alla volta. Paging autorizzato; RAM/commit/VRAM watchdog invariati. Run1000update limite1200s,heartbeat≤32update; stop5 update consecutivi>2s. Modello e risultati storici preservati. No download/install.

Realtime: ricerca Luna completata separatamente; visualizzazione non in questa suite. Verificare unità coordinate soma separate da unità skeleton; eventi/ownership ring devono impedire overwrite durante D2H.

Aggiornamento utente: T5 a2000update,curve200/500/1000/2000,timeout1800s. GRU d4/d8 a2000update,4000stream nuovi. Run L8d1 avviato a1000 resta controllo storico separato; no resume con Adam azzerato. Varianti P1/P2/D1/D2/H1 autorizzate per prova, non adozione.
