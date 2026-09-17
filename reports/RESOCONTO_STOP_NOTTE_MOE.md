# Stop pretraining e bilancio completo degli esperimenti

13 settembre2026. Stop esplicito utente eseguito. **26.247 update,683.861 target,CEDEV finale3,634243.** `stop_reason=user_stop_file`, worker/coda finiti, cleanup allocator0MiB. Checkpoint completo `results/pretrain_continuous.latest.pt`; archivi ogni2000 e migliore ci sono. No riavvio autorizzato; `STOP_PRETRAINING` tenuto.

## 1. Cosa è stato effettivamente eseguito nella notte

Due continuazioni da2000 a8000: H1-10 e H1-5, seed17, ordine storie,208.271 target ciascuna. Vincitore H1-10 continuato a20000, sonde, poi ripreso fino a stop. Continuazioni dei piloti, non training nuovi dazero a8000.

| Profilo | Parametri | CE2000 | CE8000 | s/update nel tratto notturno |
|---|---:|---:|---:|---:|
| Head condivisa,soglia10 |4.410.893|5,761664|Non eseguito|—|
| Head separata,soglia10 |5.459.469|5,086994|3,958567|0,6255|
| Head separata,soglia5 |8.947.612|5,211444|4,052994|1,2209|

Tutti166.700nodi tenuti; archi10=2.753.975,5=6.242.118. H1 aggiunge1.048.576parametri. Soglia5 aggiunge cammini ma no miglior CE ai due budget misurati e costa~1,95×. Non provato che perda a ogni budget futuro o a più semi.

H1-10: CE8000≈3,959;20000≈3,7038;24000=3,6335;26000=3,6248;stop26247≈3,6342. Oscillazioni piccole: no convergenza definitiva. Accuracy finale≈32,9% prossimo token, non storie corrette.

683.861 target =~0,123% dei555.504.351token train. Generazione greedy: frasi semplici, ripetizioni, incoerenza soggetti. Campionamento0,8 spesso sgrammaticato. Esempio finale dopo `Once upon a time`: ", there was a little boy named Tim. Tim was very happy. He was very happy. The bird was very happy. He was very happy and happy."

DEV sempre16storie×128token. Baseline bigramma4,283: ora superata su DEV, ma bigramma allenato su soli52.673target iniziali mentre modello ha visto più dati. **Non benchmark equo finale.** Audit64 una volta su checkpoint2000 congelato:CE5,161; non ripetuto su modello finale. Test finale mai aperto.

## 2. MoE: sonde realizzate, router non realizzato

**No router appreso, top-k esperti, loss bilanciamento, mascheramento dinamico o risparmio calcolo MoE implementato/allenato.** Rete elabora grafo completo secondo config scelta.

Eseguite5modalità su ciascuno dei3checkpoint: H1-10/8000,H1-5/8000,H1-10/20000. Totale15worker GPU+3analisi CPU, tutti passati, checkpoint immutati, cleanup0.

1. Train osservazionale:16storie train9000–9015,128token.
2. DEV osservazionale:16storie fisse,128token.
3. Train con etichette gruppi casuali controllate.
4. DEV con stessa partizione casuale.
5. DEV resettando stato+cache+posizioni ogni token.

Misure: rate post4, frazione neuroni spiking, |potenziale|, |corrente token|, copertura nodi,CE/accuracy/entropia, proxy flusso tra gruppi.46gruppi superclass×lato, distinti dai77pool del readout. Null: stesse cardinalità, mescolamento entro lato e bin gradi entranti/uscenti. Una partizione nulla per checkpoint: non vasta ricerca su partizioni.

Analisi CPU: tabella train token→distribuzione attività, shrinkage10, valutazione DEV contro prior globale;32shuffle token entro storia train;bootstrap200repliche perstoria; concentrazione top-k, numero effettivo gruppi, similarità stesso token tra contesti contro token diversi abbinati grossolanamente per posizione/frequenza. Non classificatore neurale o router addestrato.

### Risultato del proxy token→attività

| Checkpoint | Guadagno anatomia,nat | Guadagno gruppi casuali,nat |
|---|---:|---:|
| H1-10,8000 |0,02832|0,02738|
| H1-5,8000 |0,03698|0,03218|
| H1-10,20000 |0,02976|0,02980|

Guadagno = calo CE distribuzione attività vs prior; **non CE linguistica**. Token porta segnale su attività e batte i32shuffle osservati. Ma anche gruppi casuali mostrano quasi stesso segnale: a20000 nessun vantaggio anatomico. H1-5/8000 ha differenza maggiore, indizio da verificare con altre partizioni/repliche e interventi, non prova di esperti biologici. Intervalli/bootstrap esplorativi su16storie DEV già usate per sviluppo.

H1-10/20000: ~19,76gruppi effettivi; top1≈8,36%,top2≈14,99%,top4≈26,22%,top8≈46,28% della distribuzione normalizzata delle medie rate pergruppo. **Non quote del totale spike**, né numero esperti attivati da router. Con questa misura attività non si concentra in soli1–2gruppi.

Similarità coseno: stesso token in contesti diversi0,9732; token diversi controllati0,9083. Anche gruppi casuali hanno0,9713/0,8980: identità token influenza risposte, ma non prova semantica anatomica.

23.645/166.700nodi (~14,2%) hanno emesso almeno uno spike nei4sottopassi post osservati su2048token DEV. A8000:23.676 su10 e24.388 su5. **Gli altri non provati inutili:** campione limitato, metà sottopassi, potenziali subthreshold e contributi diversi dagli spike non esauriti. Nessun nodo rimosso.

Maggiori proxy flusso: sensory→intrinsic nei circuiti ottici e connessioni interne cb. Proxy = rate medio sorgente×peso: non traiettoria causale, non instradamento significati. Alcuni gruppi ad alto rate hanno1–4nodi: classifica grezza non basta per scegliere esperti.

## 3. Memoria nel linguaggio

| Checkpoint | CE contesto normale | CE reset ogni token | Peggioramento |
|---|---:|---:|---:|
| H1-10,8000 |3,9596|4,8332|+0,8735|
| H1-5,8000 |4,0505|5,1629|+1,1124|
| H1-10,20000 |3,7064|4,5578|+0,8514|

Contesto funzionalmente utile. Reset rimuove insieme voltage/spike,KV e posizioni: **non isola attenzione né misura da solo memoria a lungo raggio.** Valori sonde differiscono leggermente dalle valutazioni trainer per riduzioni CUDA non deterministiche/harness.

## 4. Fan-out topologico: pochi nodi→molti

Grafo intero, seed32 ascendenti/descendenti/sensoriali/hub, ciascuno con controllo abbinato pergrado/lato; stessi seed tra10/5. Raggiungibilità a1/2/4/8/16archi.

Esempio ascendenti,4archi: soglia10 raggiunge146.349nodi, controllo151.604; soglia5 raggiunge160.617,controllo161.528. Espansione esiste, ma non esclusiva del gruppo anatomico scelto. Cammino strutturale non = trasmissione utile attraverso soglie/segni/dinamica LIF.

## 5. Esperimenti precedenti: cosa hanno dato

| Esperimento | Esito e limite |
|---|---|
| T0/API/causalità/reset/PAD/cache | Contratti verificati; non misura qualità linguistica |
| T2/T3,decodifica simboli | Head18,85%;probe uscite grezze92,29%,rappresentazione84,77%; budget diversi, non confronto equo di apprendimento |
| R0,gradienti E condivisa | Coseno medio−0,05simboli/+0,01testo; nessuna condanna generale del tying |
| R1,lettore originale calibrato |75,59%DEV su simboli;102.400presentazioni di feature congelate |
| R2,reinserimento nel circuito |74,61%; beneficio della lettura sopravvive senza rimuovere nodi |
| P1,canali classe/lato |18,55%vsbase18,07%,200update: nessun beneficio netto dimostrato |
| P2,pooling riorganizzato |29,00%,200update: indizio locale, non adottato |
| D1,surrogate scala2 invece4 |16,21%,200update: non adottato |
| D2,bias corrente0,5 invece0,6 |25,10%,200update: indizio locale, non adottato |
| H1,head separata |47,46%simboli/200update; poi vantaggio confermato sul testo2000; unica variante di questa serie adottata |
| T5,L8/L16,d1 a2000 |68,75%vs75,756%: L16 migliora al budget lungo; ancora sotto criterio80% |
| T5,d8 a200update |Circa12,6%entrambi; L16 ripristina gradiente verso sorgente ma breve training non recupera il task |
| GRU controllo,d4/d8 a2000 |99,55%/32,42%; d4 calibrato,d8 no |
| Adam/AdamW/Muon/ROOT |24run,96update ciascuna. Muon/ROOT circa1,3–1,7% meglio, sotto criterio2%in entrambi i semi; Adam mantenuto. ROOT fallback, non replica completa del paper |

Verdetti negativi suL16 erano limitati a200update: non escludono capacità successive. L16 non confrontato nel pretraining lungo; TBPTT8 fisso per isolare head/soglia.

## 6. Non eseguito / non va spacciato per completato

- Router MoE vero, top-k, esperti appresi, load balancing e training comparativi.
- Ablazioni causali gruppi/archi con controlli e rimescolamento topologico; null attuali rimescolano etichette, non connessioni.
- Classificatori lineari completi pergruppo/contesto, gradienti pergruppo e analisi SCC proposti da Luna: non tutti implementati nella suite notturna.
- L8/L16 d8 a2000; confronto TBPTT16 sul testo lungo; training lungo P1/P2/D1/D2.
- Dropout: solo proposto, mai inserito nella run.
- Nuova griglia lunga Muon/ROOT, ROOT completo, CSR/event-driven/GQA/quantizzazione e altre ottimizzazioni future.
- Viewer realtime vere attivazioni: ricerca e proposta ci sono, non implementato. Reel MP4 e grafico anatomico esistono, ma impulsi illustrativi, non telemetria live.
- Benchmark equi contro modelli storici, riaddestramento baseline bigramma sullo stesso budget e ablazioni contributo connettoma.

Quindi: chiuso confronto operativo3.5+4 necessario alla notte; **non esaurita l'intera lista di ricerca fase4**, né chiusi criteri memoria/qualità fase3. Nessuna prova che modello batta già modelli storici.

## 7. Prossime decisioni, nessuna esecuzione automatica

1. Misurare separatamente contributo cache/attenzione e stato LIF a checkpoint congelato.
2. Se perseguire MoE: più partizioni casuali e pochi interventi anatomici abbinati, senza pruning permanente; poi decidere se proporre router.
3. Dropout head0/5% e TBPTT8/16 confronti distinti; stesso checkpoint e budget, una modifica pervolta. Richiedono discussione prima della modifica architetturale.

Sorgenti: `results/pretrain_actual_night_live.json`, `pretrain_continuous.json`, `graph_probe_8000_h10_analysis.json`, `graph_probe_8000_h5_analysis.json`, `graph_probe_20000_analysis.json`, relativi JSON delle5modalità, `anatomy_fanout_thresholds.json`; report storici `RESOCONTO_ESTENSIONE_FASE_3.md`, `RESOCONTO_FASE_3_4.md`, ricerca `REPORT_LUNA_MOE_GRAFO_SUITE.md`.