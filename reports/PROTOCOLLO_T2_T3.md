# T2/T3 — diagnosi fissata prima dei run

Autorizza utente: ricerca Luna high parallelo, root continua T2/T3. Mosca invariata.

- Soglia10, nodi166700, porte originali, T4+4, B2/L8, Adam0,0001, seed89. d0, head4096 originale, max200 update.80 stream train nuovi bilanciati (4 occorrenze/simbolo400..407 per32 posizioni); seed93016. DEV32 stream distinti seed93017. BOS+32+7PAD:5 finestre8/coppia,40 coppie,2.560 target. No ricerca LR.
- Checkpoint iniziale/finale; hash e variazione parametri per modulo. Controllo GPU Graph/eager ereditato da engine verificato; nuova cattura confrontata direttamente con forward originale.
- Raccolta checkpoint finale: primi32 stream train e32DEV =2.048 posizioni, più8 token isolati con reset. No storia testo/audit aperta.
- Tappe: embedding normalizzato, corrente sensoriale, voltage/rate uscite pre-pooling (pre/post),154 pooled feature (pre/post),256 rappresentazione (pre/post),logits4096. Tutti nodi porte, no sottocampionamento nascosto.
- Probe affine8 classi congelando mosca: full-batch train1024 esempi, Adam0,03,100 update, seed91. Standardizza per feature stimata solo train, std min1e-6. Fit e inferenza GPU FP32/CUDA Graph. Fit controlli: etichette train permutate seed92, val DEV vera; controllo costante; embedding controllo positivo. No tuning DEV.
- Separabilità: energia centroidi token / energia variazione intra-token, train eDEV separati. Non =mutua informazione; probe negativo non prova assenza info.
- T3: stesse sequenze/checkpoint. Tre gruppi funzionali disgiunti: ingressi sensoriali, uscite selezionate, resto. NON regioni anatomiche. Potenziale pre-reset LIF: media/RMS/min/max, quota vicino soglia[0,9;1,1], quota>1. Voltage finale e firing medio pre/post: medie/RMS/estremi/frazioni; RMS corrente token e feedback già moltiplicato gate.
- Gradiente CE ultimo token finestra8 vs corrente di ognuno degli8 token: RMS/max/quota nonzero per gruppo, ultimo token vs precedenti. Stato iniziale vuoto; BOS primo token. Include percorso attenzione stesso segmento; no gradiente oltre8. Anche gradienti parametri per modulo.
- Generazione solo diagnosi dinamica: prompt8 simboli DEV, max16 greedy, stop EOS/PAD. Modello allenato identità: non val linguistica.
- Budget worker600s, fase90s, heartbeat<=32update; raccolta progressi ogni coppia; guardie paging già autorizzate. Un worker GPU alla volta, cleanup0. No modifica architetturale senza domanda utente.
- Se strumentazione/cattura fallisce: correggere strumento, conservare tentativo. No altro training salvo errore invalidante. Report distingue prove, ipotesi e limiti.
