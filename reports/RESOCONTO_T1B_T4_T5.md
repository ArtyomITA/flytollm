# T1b/T4/T5 — calibrazione e memoria a grafo intero

T1b, T4, T5 conclusi. Calibrazione d1 GRU ok; nessun vantaggio L16 run mosca. Nessuna riduzione grafo, modifica porte/readout o nuova attenzione. Suite soluzioni: [SUITE_SOLUZIONI_FASE_3.md](SUITE_SOLUZIONI_FASE_3.md). Protocollo: [PROTOCOLLO_T1B_T4_T5.md](PROTOCOLLO_T1B_T4_T5.md).

## T1b — calibrazione d1 risolta localmente

Stessa GRU416.128 param, Adam0,003, B2/BPTT40, seed89.1000 update,2000 stream nuovi,62.000 target validi; primi400 stream identici vecchio T1. Nessun grid o cambio soglia.

| Budget | Train accuracy | DEV accuracy | DEV CE | Training s |
|---|---:|---:|---:|---:|
|200 update storico|30,80%|30,80%|1,21070|1,93|
|1000 update T1b|100%|100%|0,00242|10,01|

DEV64 stream fissi,1.984 target; train eval primi64 stream. Calibrazione: controllo positivo d1 esiste a questo budget. Non significa1000 minimo necessario, né garantisce d4/d8 o mosca.80% non abbassato; nessuna scelta architetturale dedotta dal solo controllo.

## T4 — contenuto cache e dipendenza locale

Checkpoint simboli: T2 d0 finale.16 train+16DEV stream,512 target/split. Probe K senza RoPE e V del token appena scritto: DEV75,98%/74,02%; controlli etichette permutate11,72%/13,67%; train probe100%.51200 target ripetuti/probe; non task linguistico o memoria lunga.

Interventi da identico stato reale a ogni token; propaga al passo successivo solo ramo reale. Cache vuota, contenuto invertito fra posizioni valide con de/ri-rotazione RoPE, riordino congiunto K/V/posizioni come controllo invariante. Nessuna nuova info dal futuro.

| DEV simboli | CE | Accuracy | Argmax cambiati rispetto reale |
|---|---:|---:|---:|
|Reale|2,151006|17,77%|0%|
|Vuota|2,153623|19,92%|15,23%|
|Contenuto invertito|2,151277|18,55%|1,37%|
|Riordino congiunto|2,151006|17,77%|0%|

CE e accuracy danno segnali diversi: cache reale meglio in CE ma peggio in accuracy vs vuota. Non dichiarare utilità generale da questo test.

Checkpoint testo: storico `phase35_adam_s17.diagnostic.pt`,16DEV TinyStories,primi64 target/storia,1.024 target. Prefissi più corti del report storico128: CE non confrontabile direttamente con quel numero.

| DEV TinyStories | CE | ΔCE vs reale | Accuracy | Argmax cambiati |
|---|---:|---:|---:|---:|
|Reale|5,711880|0|7,52%|0%|
|Vuota|5,712096|+0,000216|7,52%|0,20%|
|Contenuto invertito|5,711734|−0,000146|7,52%|0%|
|Riordino congiunto|5,711880|≈0|7,52%|0%|

**Interpretazione:** cache contiene segnale distinguibile; dipendenza utile linguistica molto piccola in questo checkpoint e intervento one-step. Non prova inutilità attenzione: perturbazioni non accumulate, nessun riaddestramento senza cache, checkpoint poco allenato e generazione storicamente degenerata.

Controllo invariante: max errore logits1,91e−6 simboli/2,38e−6 testo, zero argmax modificati. Rami perturbati finiti; cleanup0MiB. Prima raccolta per coppia B2; [verifica aggiuntiva per storia](results/phase3_t4_detail.json) completa16 storie e riproduce CE aggregata entro1e−5, senza update o nuovi dati. Cache vuota peggiora10 storie e migliora6; contenuto invertito migliora9 e peggiora7. Effetti piccoli e misti, nessuna significatività statistica dichiarata. Variazioni controllo congiunto = rumore numerico.

## T5 — confronto a update uguali concluso

200 update per config, checkpoint T2 finale condiviso, Adam0,0001. Un update ogni16 posizioni: L8 accumula due loss pesate sul numero totale target e fa detach interno; L16 conserva gradiente16. Stessi stream/target/update,4.152 target d1 e3.214 d8. Stesso elenco primi200 segmenti T1;66 coppie complete più2 segmenti ultima coppia. Nessun optimizer step intermedio in L8.

| Ritardo / finestra | Train accuracy | DEV accuracy | Train CE | DEV CE | Training s | Picco MiB |
|---|---:|---:|---:|---:|---:|---:|
|d1 / L8|27,02%|27,82%|2,02077|2,01442|130,08|489,18|
|d1 / L16|26,41%|26,26%|2,02214|2,01879|131,41|569,69|
|d8 / L8|11,85%|12,57%|2,10876|2,10774|132,76|489,18|
|d8 / L16|11,72%|12,63%|2,10897|2,10957|133,17|569,69|

Accuracy4096 e ristretta8 identiche. Train eval32 stream, DEV64 distinti. Un seme: differenze piccole non dimostrano superiorità generale. d8: un target corretto in più su1536 con L16, ma CE peggiore; nessun recupero pratico. Tutti risultati sotto80% storico.

Gradiente CE ultimo token finestra16 vs corrente sensoriale sorgente d8: **L8=0**, **L16=5,36e−6 RMS**. Troncamento rimosso nel secondo caso; apprendimento non migliora utilmente. Amplitudini misurate sui rispettivi checkpoint finali, non prova isolata di dinamica a pesi uguali. Su d1 il token sorgente è entro entrambe le finestre: RMS2,82e−4/2,89e−4.

**Decisione:** non adottare L16, non raddoppiare ancora la finestra. Più orizzonte gradiente non risolve da solo lettura/budget. Circa stesso tempo perché entrambi elaborano16 posizioni/update; L16 picco allocator+16,46%. Non è confronto col vecchio training che faceva update ogni8.

Nota: sequenze BOS+32 hanno coda con PAD. Finestre finali hanno solo2 target per B2; identiche fra L8/L16 e contano come update. Non chiamare16 posizioni padded «16 target validi».

T1b aveva62.000 target e GRU/BPTT40; T5 mosca solo4.152/3.214 target, LR diverso. Confronti non equivalenti: successo GRU non dimostra che questo budget mosca debba bastare. d8 non ha ancora controllo GRU calibrato. Test isola il fattore detach, non tutte le cause del mancato apprendimento.

Verifica eager/Graph loss, gradienti e pesi passata per tutti4 casi, warmup ripristinato; finitezza per update e cleanup0MiB. Hash iniziale/dati ed elenco segmenti identici in ogni coppia L8/L16. Nessuna modifica ai moduli produzione. Log include warning conversione tensore→scalare nella verifica; nessun fallback o errore numerico.

## Ricerca e decisione

Luna high ha aggiornato ricerca su tying, gradienti, pooling/porte per ruolo e dinamica, senza pruning. [Report](REPORT_LUNA_T23_SOLUZIONI.md). Prima head originale a budget adeguato; head untied aggiungerebbe1.048.576 param e resta proposta successiva.

Correzioni porte/pooling = proposte da discutere; traiettorie migliori per TinyStories vanno misurate su train e verificate DEV. Non sono attributi linguistici innati di neuroni o regioni.

Riepilogo finale:8 worker validi (T1b,2T4,4T5,1dettaglio T4), hash verificati e cleanup0 per tutti. Calibrazione d1 risolta; T4/T5 eseguiti, capacità mosca ancora insufficiente. Nessun R1/R2 o pilot eseguito: restano prossime azioni proposte, fase3 non dichiarata chiusa.

Evidenze: [T1b](results/phase3_t1b_d1.json), [T4 simboli](results/phase3_t4_symbols.json), [T4 testo](results/phase3_t4_text.json), [T5 L8/d1](results/phase3_t5_l8_d1.json), [L16/d1](results/phase3_t5_l16_d1.json), [L8/d8](results/phase3_t5_l8_d8.json), [L16/d8](results/phase3_t5_l16_d8.json), [riepilogo verificato](results/phase3_t1b_t45_summary.json).