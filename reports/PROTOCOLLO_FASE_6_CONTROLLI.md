# Fase 6 — controlli: grafo rimescolato e baseline standard

Data: 14 settembre 2026. Fissato prima dei risultati. Richiesta utente: prima delle ottimizzazioni, eseguire i controlli mancanti della fase 6 (grafo rimescolato allenato; GRU e Transformer piccolo a pari budget). Nessuna modifica al modello mosca, ai suoi file sorgente o ai checkpoint esistenti. `STOP_PRETRAINING` resta il marker del pretraining lungo; i controlli hanno marker separato `STOP_PHASE6`.

## Domande

1. **Il cablaggio reale conta?** Stessa pipeline mosca in tutto (166.700 nodi, LIF, segni Dale, porte sensoriali, readout 2.241, attention MHA4/KV128, H1 head separata, T 4+4, init artificiale seed 17), con la sola topologia sostituita da un grafo casuale a gradi conservati.
2. **La mosca vale qualcosa in assoluto?** Modelli standard senza connettoma (GRU, decoder Transformer piccolo) sugli stessi dati, ordine, target, valutazione.

## Controllo 1 — grafo rimescolato a gradi conservati

- Base: grafo soglia 10, 2.753.975 archi, `src`/`dst` compatti su 166.700 body ID ordinati.
- Rimescolamento: permutazione casuale della sola colonna `dst` (seed 41), poi correzione iterativa di self-loop e archi duplicati con scambi di destinazione fra coppie disgiunte. Risultato: grado uscente e grado entrante di ogni nodo **identici** all'originale; nessun self-loop; nessun duplicato; peso di ogni arco resta sulla riga della sua sorgente; segno = segno del neurone presinaptico, quindi legge di Dale conservata; `sensory`, porte di ingresso, popolazioni di lettura e gruppi readout **identici** (dipendono dai nodi, non dagli archi).
- Cambia: quali nodi sono collegati, reciprocità, clustering, somma dei pesi entranti per nodo (la normalizzazione iniziale delle magnitudini in `build_cns` viene ricalcolata sul grafo rimescolato, come per ogni grafo).
- Verifica automatica prima del training: uguaglianza bincount in/out, zero self-loop, zero duplicati, array pesi invariato, frazione di archi originali sopravvissuti (attesa ≈ 0) e reciprocità prima/dopo registrate in `results/phase6_rewire_stats.json`.
- Codice: `fly_rewire.py` (grafo), `pretrain_control.py` (trainer che riusa `pretrain_resumable.run` invariato sostituendo solo il costruttore del grafo). I file in `SOURCES` di `pretrain_resumable.py` non vengono toccati: i checkpoint esistenti restano riprendibili.
- Training: identico al confronto 3.5+4 e alla notte: `--threshold 10 --head separate --seed 17`, batch 2, 16 posizioni/update, TBPTT 8, Adam 1e-4, clip 1, storie complete dalla 130, reset a ogni coppia, CUDA Graph FP32. Hash dell'init artificiale (`artificial_initial_sha256`) atteso uguale a quello del pilota H1-10: stessi pesi di partenza per embedding, porte, attention, readout.
- Budget: 2000 update (confronto con `pretrain_pilot_h10`, CE 5,087 seed 17; replica seed 23 5,239), poi ripresa a 8000 (confronto con `pretrain_night8000_h10`, CE 3,959). Stessi target attesi: 52.673 e 208.271.
- Un solo seed di rimescolamento: la variabilità del rimescolamento non è stimata. La replica seed 23 della mosca dà l'unico ordine di grandezza della variabilità da init.

## Controllo 2 — baseline standard (PROTOCOLLO_VALUTAZIONE.md)

- **GRU**: embedding 4096×256 condiviso con l'uscita, 2 strati GRU hidden 256, proiezione 256, dropout 0. Stato ricorrente portato fra finestre e staccato ogni 8 posizioni come la mosca (TBPTT 8, due segmenti da 8 per update, un solo passo Adam). Reset a inizio storia; slot PAD congelati.
- **Transformer**: d 256, 2 blocchi pre-LayerNorm, MHA 4 teste, FFN 1024 GELU, RoPE, contesto 128, embedding condiviso, dropout 0. A ogni update riceve le 16 nuove posizioni più il prefisso della stessa storia fino a 128 token; loss solo sulle 16 nuove; gradiente attraverso tutto il prefisso visibile. È un vantaggio strutturale rispetto al TBPTT 8 della mosca e va dichiarato, non compensato.
- Dati: stesso generatore di coppie (`pretrain_resumable.next_pair`, storie dalla 130, coppie sequenziali, 16 posizioni per update), stesso seed 17, stesso conteggio unigramma/bigramma congelato a 2000 update. Uguaglianza dei target cumulativi (52.673 a 2000, 208.271 a 8000) verificata e riportata.
- Valutazione: DEV16 storie riservate, prefisso 128, CE e accuracy come per la mosca; generazioni greedy e a temperatura 0,8 (seed 93018) sugli stessi 3 prompt.
- Learning rate: 1e-4 come la mosca (confronto stretto) e 1e-3 (LR più consueto per questi modelli). Nessuna ricerca oltre queste due; entrambe riportate.
- Esecuzione: GPU FP32 eager (nessuna cattura CUDA Graph: modelli standard, non servono), stesse guardie del supervisore, un worker per volta.
- Budget: fino a 8000 update con valutazione a 2000 e 8000.

## Sequenza (`phase6_controls_queue.py`)

1. `phase6_rewired_h10_2000` — controllo 1 a 2000.
2. `phase6_gru_lr1e-4`, `phase6_transformer_lr1e-4`, `phase6_gru_lr1e-3`, `phase6_transformer_lr1e-3` — 8000 update ciascuno, eval a 2000/8000.
3. `phase6_rewired_h10_8000` — ripresa del punto 1 fino a 8000.
4. `results/phase6_controls_summary.json` — tabella CE DEV a pari update/target: mosca reale (pilota, replica, notte), grafo rimescolato, GRU, Transformer, unigramma, bigramma.

Stop morbido: file `STOP_PHASE6` nella radice. Ogni job registra `ok`, cleanup allocator 0, picco VRAM, tempo per update. Output preesistenti non vengono sovrascritti.

Log: valutazione DEV ogni 500 update per tutti i job (curva a 500/1000/…/8000, generazioni a ogni valutazione); telemetria di training ogni 16 update (loss, target, norma gradiente, tempo; per la mosca anche quantili di potenziale e spike medio) nei JSONL `results/<job>.jsonl` e nei `results/<job>.console.log`; stato vivo della coda in `results/phase6_controls_live.json`.

Marker: `STOP_PRETRAINING` era il freno della notte del 13 settembre. Con autorizzazione dell'utente (14 settembre) è stato rinominato `STOP_PRETRAINING.night_stop_2026-09-13` per lasciar girare il trainer sui controlli. Il pretraining continuo a 26.247 update non viene ripreso da questa coda.

GRU: il segmento di 8 posizioni passa intero in cuDNN; una lane oltre la fine della sua storia continua a evolvere su embedding PAD, ma le sue uscite sono mascherate e lo stato viene scartato al reset della coppia successiva.

## Lettura dei risultati, fissata ora

- Rimescolato ≈ reale (differenza entro la variabilità seed 17/23, ≈0,15 nat a 2000): il cablaggio anatomico non contribuisce in questa configurazione. Non prova che non possa contribuire con altra lettura/porte.
- Rimescolato peggiore del reale oltre quella variabilità: la topologia reale porta vantaggio misurabile in questa pipeline; un solo seed di rimescolamento, replica necessaria prima di dichiararlo robusto.
- Rimescolato migliore: la topologia reale è un vincolo sfavorevole per questo compito; risultato da riportare così com'è.
- Baseline: se GRU o Transformer battono la mosca a pari target, la mosca non ha ancora vantaggio assoluto; se la mosca batte le baseline, resta da verificare a pari parametri e tempo (i parametri differiscono: mosca 5,46M, GRU ≈2,7M, Transformer ≈2,6M).
- Nessun risultato qui chiude la fase 3 né autorizza modifiche architetturali.

## Limiti dichiarati

Un seed per configurazione; DEV16 già usato per sviluppo; test finale e audit64 non consultati; LR delle baseline non ottimizzato oltre due valori; Transformer con orizzonte di gradiente più lungo della mosca; tempi per update non confrontabili fra architetture. Confronto a pari dati, non a pari tempo.
