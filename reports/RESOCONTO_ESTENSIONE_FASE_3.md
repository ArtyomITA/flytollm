# Fase3 — estensione in corso

200update misura comportamento a quel budget; no esclude apprendimento dopo. Utente autorizza varianti isolate P1/P2/D1/D2/H1; chiede T5 e controlli GRU fino2000update. Protocollo: [estensione](PROTOCOLLO_ESTENSIONE_FASE_3.md).

## Risultati verificati

| Prova | DEV | Lettura |
|---|---|---|
| GRU d4,1000update |44,14%;CE1,4085|Sopra caso12,5%;no calibrata criterio80%|
| GRU d8,1000update |24,67%;CE1,8820|Sopra caso;no calibrata criterio80%|
| R1 lettore originale,offline |18,85%→75,59%;CE2,1534→1,0249|Stessa head4096 tied;102.400 presentazioni su1024 feature train congelate|
| R2 circuito completo |74,61%;CE1,0565|Guadagno sopravvive cache/feedback;d0 simbolico,no linguaggio|
| R2 altri32 contesti train |75%;CE1,0709|Nuovi per fit R1,già visti da checkpoint T2|
| R0 embedding lookup/output |coseno medio ~−0,05 simboli;~+0,01 testo|Opposizione debole locale;no diagnosi generale contro weight tying|

Tutti worker sopra: `ok=true`,cleanup0MiB. Sorgenti e checkpoint nei JSON. R1 equivalenza update eager/capture verificata. R0:8 coppie/split,CEultimo token,8 posizioni; coseni stesso spazio E. No generalizzare ad altri checkpoint/task.

R1 allena solo norm/proiezione/output-norm originali. No parametro aggiunto,no neurone rimosso. Mostra lettore originale può sfruttare più segnale se allenato separato; no dimostra ancora che training congiunto raggiunga stesso risultato.

## In corso / ancora da verificare

T5 L8d1 completato:200update26,76%,500update50,55%,1000update66,68% (CE0,94984);20.674target,646,83s training,cleanup0MiB. Confronti separati fino2000 in coda. Protocollo originale1000 recuperato in `PROTOCOLLO_ESTENSIONE_FASE_3_1000.md`,hash identico a quello del worker prima estensione utente. Valore200 diverso da precedente27,82%: run singolo no stima variabilità. Attendere confronto completo.

Code: `phase3_suite_driver.py`,stato `results/phase3_suite_live.json` quando parte prima prova dopo run attuale. Sequenza seriale:GRU2000→P0→riferimento+5varianti→smoke pilot→T5quattro2000→pilot100k. Stop su errore; test non eseguito resta non eseguito. Replica/audit dopo valutazione pilot; audit64 non aperto.

Luna realtime: [report](REPORT_LUNA_VISUALIZZAZIONE_REALTIME.md). WebGL2,8.192 nodi campionati,2Hz,ring pinned con ownership/eventi. Campionamento vista no riduce modello. Coordinate soma: unità da verificare; no ereditare auto unità skeleton. Visualizzazione non ancora implementata.

| Fase | Stato |
|---|---|
|1 — Dati/tokenizer/protocollo|✅ Completata|
|2.1 — API nucleo/stato|✅ Completata|
|2.2 — Embedding/interfacce/readout|✅ Completata|
|2.3 — Attenzione/memoria|✅ Completata|
|2.4 — Integrazione LM/generazione|✅ Completata|
|2.5 — Verifica completa|✅ Completata|
|3.1 — Protocollo/controlli|Controlli estesi eseguiti;GRU d4 calibrato,d8 ancora no|
|3.2 — Apprendimento/lettura|Overfit verificato;R1/R2 riusciti su simboli|
|3.3 — Memoria/stabilità|L8/L16 d1 a2000 eseguiti;d8 lunghi rinviati|
|3.4 — Ottimizzatori|✅ Confronti eseguiti;Adam mantenuto|
|3.5 — Validazione linguistica|Confronto TinyStories2000update/candidata in corso;replica/audit aperti|
|4 — Ottimizzazione/più archi|Unita3.5: tied10/H1-10/H1-5;no adozione prima risultati|
|5 — Pretraining|Da fare|
|6 — Benchmark/ablazioni|Da fare|

## Aggiornamento controlli e varianti

GRU2000update: d4 DEV99,55%,CE0,05246; d8 DEV32,42%,CE1,69044. Cleanup0MiB entrambi. d4 calibrato; d8 no.

P0:2204/2241 uscite raggiungibili,16108/17937 ingressi con cammino verso uscita. T7 solo topologia: soglia5→2232 uscite e17078 ingressi; stessi166700nodi. Dimostra cammini aggiuntivi,no qualità linguistica.

Varianti200update da identico checkpoint iniziale: base18,07%DEV/CE2,15387;P1 18,55%/2,15783;P2 29,00%/2,09461;D1 16,21%/2,15728. Tutti cleanup0MiB. P2 ordine effettivo classe/lato/sottoclasse/bodyID con capacità gruppi preservata. Nessuna variante adottata; no verdetto convergenza da200update.

Preferenza utente: più parametri entro VRAM; soglia5 candidata7,90Mparametri totali (6,24Marchi). Confronto5 successivo a pari dati/update; no modifica confronti10 già partiti.

Reel: Luna effort max incaricata; ricerca esempi cervello mosca su scacchi/doomscrolling/Beat Saber e verificaRLHF. Consegna richiesta MP4,no solo storyboard.

D2:DEV25,10%,CE2,14011;H1:DEV47,46%,CE2,04533,+1.048.576parametri,total5.459.469. Cleanup0MiB. Rate medio D2~3,43% vsbase~4,03% (stesso protocollo/maschere): più firing no obiettivo sufficiente.

T5 L8d1/2000 completato:41.352target,DEV68,75%,CE0,79938,training1361,93s,peak490,35MiB,cleanup0. Curva200/500/1000/2000:27,37/52,07/65,47/68,95%. Ripetizione finale stessa config68,75%: piccola oscillazione va quantificata prima di interpretare differenze marginali. `index_add_` suCUDA può essere non deterministico secondo [PyTorch2.14](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.index_add_.html); nel nostro nucleo causa plausibile,non ancora isolata sperimentalmente.

Smoke T6/128target passato,compresi generazione greedy+sampling,confronto contesto/reset e cleanup. Pilot100k ancora in coda.

VideoV2 richiesto da utente: grafo mosca→LLM in animazione continua. Asset reale `artifacts/reel_graph_reference.json`:3000soma campionate da139662coordinate valide,1057archi reali fra questi nodi; visualizzazione solo,no modifica modello. Luna aggiorna MP4 con questo asset e SFXoriginali.

Reel V2: render30s720×1280/24fps con SFX originali consegnato da Luna; root controllati frame10/21/28s: grafo visibile, token in ingresso, interfaccia FLY-LM emergente, gag bzz. Richiesta correzione etichetta CLS→BOS prima consegna utente. Impulsi illustrativi,no telemetria reale; video promozionale concettuale,no prova capacità linguistica.

## 13settembre — confronto3.5+4 e preparazione notte

T5 L16d1/2000:DEV75,756%,CE0,67712;1382,41s training;cleanup0. L8d1/2000:68,75%,CE0,79938. L16 migliora a questo budget; precedente esclusione su200update non sostenibile. d8 lunghi rinviati per priorità utente, no passati.

Trainer `pretrain_resumable.py` preparato: CUDA Graph, storie intere, checkpoint pesi/Adam/stato/RNG/cursore. Smoke10tied e5H1, ripresa fresca entrambi, cinque sonde osservative su5H1, statistiche/null e fallimento isolato passati. Queste prove validano esecuzione, no qualità linguistica.

**Solo3.5+4 avviati:** `phase35_4_compare_queue.py`, stato `results/phase35_4_compare_live.json`. Tre2000update: tied10→H1-10→H1-5, stessi dati/semi/budget. Consiglio da CEDEV ripetuta; poi attesa utente. Root aveva preparato avvio auto del training lungo troppo presto; coda fermata su correzione utente, pilota preservato. `HOLD_PRETRAINING` impedisce avvio lungo. Nessun pretraining lungo partito.

Notte da autorizzare separato: checkpoint ogni2000, test dopo20000 e ripresa; eventuali run multiple sequenziali a budget uguale. Dettagli `PRETRAINING_NOTTE.md`. Ricerca Luna high `REPORT_LUNA_MOE_GRAFO_SUITE.md`; 44M MoEUT esempio piccolo verificato, no equivalenza anatomia=router.

Fan-out su grafo completo eseguitoCPU in23,53s:32 ascendenti scelti per out-degree su10 raggiungono146349nodi entro4archi; null di grado/lato151604. Su5 stessi seed160617/null161528. Espansione ampia esiste ma da sola no distingue esperti anatomici. Nodi/coordinate invariati fra soglie; cambiano archi, no forma anatomica. Disegno soma campionato no mostra tutti neuriti del cervical connective.

## Chiusura operativa3.5+4 e avvio autorizzato

Aggiornamento successivo sostituisce precedente attesa del via: utente ha autorizzato completare verifiche e lanciare. H1-10 scelta; confronto2000,replica23,audit64,14smoke GPU e7test CPU conclusi. Resoconto definitivo: `RESOCONTO_3_5_4_NOTTE.md`. Notte avviata e training oltre2000 verificato; stato `results/pretrain_actual_night_live.json`. Due profili8000 con sonde,migliore20000+sonde,poi continuo. Criteri memoria3.3 e qualità restano aperti.