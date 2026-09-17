# T0/T1 — risultati

T0 passato. T1 eseguito entro budget; controllo memoria d1 NON calibrato. Mosca invariata; nessun T2 avviato.

## T0: contratti

- Oracle FIFO100% su d0/1/4/8:2048/1984/1792/1536 target DEV.
- Chunk8/16 equivalenti; BOS reset; PAD congela; perturbare futuro non cambia prefisso.
- Generatore storico d1/4/8 coincide oracle. d0 aggiunto nuovo generatore: vecchio helper non prevedeva caso; nessuna correzione retroattiva risultati.
- `story_batch` reale: batch misto, PAD, EOS previsto, nessun target fra storie. Loss produzione catturata:4 target validi, CE uniforme=ln4096. Verifica fine storie train18/81.
-64 IDs audit riservati nel manifest; esclusi DEV storico, elenco validation protocollo valutazione e primi82 IDs. Testi audit non valutati; test finale chiuso.
- Run autorevole: `results/phase3_t0_complete.json`. Prima esecuzione `phase3_t0.json` precede aggiunta check batch testo; conservata preliminare, non conteggiata replica.

## T1: GRU di controllo

416.128 parametri: embedding4096×32 → GRU64 → head4096. Adam0,003;200 update; B2;400 stream nuovi×32 simboli, stesso elenco fra ritardi. Train eval64 stream già visti; DEV64 distinti. Accuracy tabella su4096 classi; ristretta8 identica.

| Ritardo | Train accuracy | DEV accuracy | Train CE | DEV CE | Target training | Training s |
|---|---:|---:|---:|---:|---:|---:|
|0|100%|100%|0,00935|0,00940|12.800|1,98|
|1|30,80%|30,80%|1,21845|1,21070|12.400|1,93|
|4/8|—|Non avviati|—|—|0|—|

Chance fra8=12,5%; costante400 DEV13,38% d0 /13,36% d1. Oracle100%.

**Interpretazione:** identità d0 apprendibile con questo controllo. d1 sopra caso, sotto80% prefissato: budget/configurazione controllo non ancora calibrati. Non identifica quale limite pesi di più; non dimostra impossibilità grafo. d4/d8 saltati per protocollo, nessun inseguimento LR/epoche.

BPTT40 intero nella GRU, contro TBPTT8/16 storico mosca; LR/dati/architettura differenti. Tempi misurano solo controllo; nessun rapporto prestazioni equo fra modelli. Un seme; nessuna stima variabilità. Train accuracy calcolata su sottoinsieme64 fissato, non su tutti400 stream.

## T1: baseline testo

Fit solo train18..81, massimo128 target/storia;8.126 target. DEV16 storico,2.048 target. Stesso tokenizer4096; unigram add-one; bigram backoff10 verso unigram. Nessun tuning DEV.

| Baseline | Train CE | DEV CE | Train accuracy | DEV accuracy |
|---|---:|---:|---:|---:|
|Unigramma|5,861|6,012|7,38%|7,32%|
|Bigramma|2,724|4,803|36,88%|25,10%|

Bigramma: conosce solo token corrente; migliora CE DEV di1,209 nat/token vs unigramma. Evidenza che contesto locale offre segnale su questi dati. Mosca storica CE5,918–6,149: manca ancora vantaggio su questa baseline semplice. Nessuna conclusione su vero pretraining da soli64 prefissi.

## Esecuzione e limiti

- GPU FP32, CUDA Graph train/eval; nessun offload. CPU prepara dati/conteggi e controlli deterministici.
- GRU equivalenza loss/pesi con riferimento GPU eager passata; loss/gradienti/pesi finiti per200 update; picco allocator38,86MiB. Preparazione/cattura4,05s d0,2,90s d1, escluse dai tempi training.
- Tutti3 run autorevoli `ok=true`, guardie superate, allocator finale0MiB. Guardie paging autorizzate; nessun arresto RAM/VRAM.
- Warning PyTorch durante riferimento: stream AccumulateGrad diverso; equivalenza passata, zero fallback. Warning conversione loss a scalare nel logging; nessun errore numerico osservato. Log conservati. Non attribuire a questi microtempi prestazioni produzione.
- Hash sorgenti/protocollo/manifest registrati; stream esatti salvati. Checkpoint GRU non salvato: controllo usa metriche finali, nessun resume previsto.

## Decisione

T0 chiuso. T1 chiuso come esperimento; calibrazione d1 resta aperta. Prossimo utile **T2/T3: token→readout e dinamica**, già informati da d0 positivo. Prima di nuovo confronto memoria, calibrare controllo d1 con protocollo separato. T4–T7 non eseguiti. Criteri storici80% immutati; fase3 resta aperta.

Evidenze: [manifest](configs/phase3_t01_manifest.json), [protocollo](PROTOCOLLO_T0_T1.md), [T0 e baseline testo](results/phase3_t0_complete.json), [GRU d0](results/phase3_t1_d0.json), [GRU d1](results/phase3_t1_d1.json).