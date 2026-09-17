# T1b/T4/T5 — protocollo prima dei risultati

Mandato: preserva grafo intero; calibra lettura/porte/dinamica, no ridurre nodi. T4/T5 ok; varianti architetturali dopo servono opinione utente.

## T1b
- GRU T1 identica, seed89, B2/40, Adam0,003, FP32 GPU Graph. d1:1000 update,2000 stream nuovi×32; primo400 identico T1, prosegue stesso RNG93014. DEV93015 identico64 stream. Train eval primi64. Un run, no grid;80% DEV = calibrazione positiva. Budget62.000 target,5× T1. No promuove auto d4/d8.
- Nuovo file/versione; storico T1 immutato. Audit ID invariati, no testo audit aperto.

## T4
- Checkpoint T2 d0 finale:16 stream train+16DEV da manifest T2,32 token ciascuno. Probes K senza RoPE/V del token appena appeso: fit512 train,512DEV, affine8, Adam0,03/100update; controllo permutato.
- Checkpoint linguistico storico `phase35_adam_s17.diagnostic.pt`:16 DEV storici, prefisso max64 target/storia. No fit testo.
- Interventi one-step da **stesso stato reale**: cache reale; vuota; contenuto K/V invertito nel tempo entro slot validi (K de-rotato, permutato, ri-rotato a posizioni originali); permutazione congiunta K/V/posizioni/valid = controllo negativo matematicamente invariante. No usa semplice permutazione congiunta come test memoria temporale.
- Stato per token successivo aggiorna SOLO ramo reale; perturbazioni no accumulate. Futura informazione sempre esclusa; maschere e next_position preservati nel trattamento contenuto. No update pesi.
- CE/accuracy, RMS/max differenza logits, frazione argmax cambiati, differenze CE per storia. Risultati = dipendenza locale, no vantaggio causale provato da riaddestramento.

## T5
- Quattro casi: d1/d8 × L8/L16. Stesso checkpoint iniziale T2 finale (identità già diagnosticata), Adam0,0001, seed89, FP32 GPU Graph, B2.200 update/config; no grid o seme extra.
- Stream esatti T1 train93014 (manifest originale), nuovo checkpoint optimizer; DEV64 T1; train-eval primi32 stream. BOS+32 simboli, padding a48:3 segmenti16/coppia. Primi200 segmenti con target, stesso elenco fra L8/L16. Ultima coppia parziale esplicita.
- L8: backward prima metà pesato n1/(n1+n2), detach stato/cache, backward seconda metà pesato n2/(n1+n2), clip1 una volta, optimizer una volta. L16: backward unico su tutti target16. Identiche forward, target, ordine, frequenza update; differisce solo orizzonte gradiente.
- Verifica GPU eager/Graph loss, gradienti e pesi con ripristino warmup/optimizer; test finestra con n1 diverso da n2. Conta update/target/hash; no confronti solo su epoche.
- Riporta train/DEV, CE4096 e accuracy8/4096; tempo/picco/finitezza/cleanup. Sensibilità CE ultimo token vs correnti16, con detach8 vs16, su coppia DEV fissa. No prova memoria appresa da sola.
- Tetto600s per worker, heartbeat32update o più spesso, guardie paging ok, stop5update>2s; una GPU alla volta. No pretraining/T6 implicito.

## Decisioni
- T1b≥80%: task d1 calibrato per quella GRU/config/budget; no parità budget con mosca.
- T4 cache vuota peggiora: evidenza uso utile locale; inversione cambia ma no migliora/peggiora stabile: dipendenza, no qualità. Controllo congiunto deve avere errore numerico piccolo; se errore sostanziale, indaga test.
- T5 vantaggio L16 sotto budget pari: candidato training da valutare; nessun vantaggio → no aumenta ancora finestra senza nuova ipotesi.80% storico invariato.
- Suite soluzioni T1–T5 separata, aggiornata con ricerca Luna e risultati; grafo intero = riferimento permanente.
