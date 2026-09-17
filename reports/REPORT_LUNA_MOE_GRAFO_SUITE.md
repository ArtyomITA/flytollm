# Report Luna — MoE piccolo + grafo CNS mosca

Data: 2026-09-13. Ricerca web primaria; no GPU, training, pruning, cambio router.

## Verdetto operativo

**MoE sì come ipotesi testabile; regioni anatomiche no come esperti già validati.** Dopo checkpoint 20.000 update, prima sonde read-only su grafo completo. Solo se segnale mostra concentrazione stabile e null falliscono, fare ablation temporanee + controlli causali. No adozione automatica di regioni, soglie, maschere, router.

Reel = metafora visiva: "CNS sparso → cervello denso". Campionamento 3.000 soma/PCA concettuale; non telemetria biologica né prova traiettorie funzionali. Anatomia dà partizione/prior; da sola non assegna semantica, competenza linguistica, causalità.

## Evidenza MoE: limite inferiore credibile

| Fonte primaria | Scala/evidenza | Trasferimento prudente |
|---|---|---|
| **MoEUT**, Csordás et al., NeurIPS 2024 | Config pubblicate da **44M** a 1B; Universal Transformer, parametri condivisi, esperti FF, SwitchHead. Repo espone `MoEUT_44M` e router top-k. | Esempio piccolo riproducibile di MoE LM. Guadagno da routing appreso + Transformer; non prova che partizione anatomica funzioni in SNN. |
| **Scaling Laws for Upcycling MoE**, Liew et al., ICML 2025 | Dense 15M, 44M…; dense 15M con 8 esperti → **~92,2M parametri MoE totali** (tab. 3), top-k 1–4. Router init casuale; esperti init copiati dal dense. | Dense 15M = minimo suite, non MoE totale da 15M. Upcycling/routing richiedono budget/token e controllo sparsità; non estrapolare a 20k update. |
| Shazeer et al., 2017 | Gating appreso sceglie pochi FFN per input; condizionalità separa capacità totale da compute attivo. | Definizione MoE/router. No implicazione biologica. |

Conclusione: **no fonte primaria credibile con MoE LM end-to-end sotto 44M totali**. Repo didattici e post non contano. Meglio usare concetto MoE per misure (competizione, entropia, specializzazione), non importare subito architettura Transformer.

MoEUT e scaling paper: tre vincoli. (1) router appreso da token/stato; (2) specializzazione richiede dati+update, non da etichette umane; (3) sparsità cambia capacità attiva, carico, bilanciamento. In upcycling, duplicare dense conserva funzione iniziale, ma esperti troppo pre-addestrati specializzano più lento. A 20k update misurare traiettoria, non solo loss finale.

Fonti: [MoEUT arXiv](https://arxiv.org/abs/2405.16039), [repo ufficiale MoEUT](https://github.com/RobertCsordas/moeut), [Scaling Laws arXiv](https://arxiv.org/abs/2502.03009), [HTML con tabelle/config](https://arxiv.org/html/2502.03009), [Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538).

## Cosa dicono davvero i connectomi

FlyWire mostra rich-club, broadcaster/integrator, 78 neuropili anatomici; paper descrive topologia e possibili vie di flusso, non router per linguaggio. Modello LIF whole-brain di Shiu et al. usa connettività + neurotrasmettitori, predice circuiti sensorimotori; successo non dimostra LM, semantica, separazione esperti. FlyWire femmina-brain e MaleCNS qui (CNS maschile con VNC, 166.700 ID locali) non sono stesso dataset.

Fonti: [FlyWire wiring diagram, Nature 2024](https://doi.org/10.1038/s41586-024-07558-y), [network statistics/rich-club, Nature 2024](https://www.nature.com/articles/s41586-024-07968-y), [LIF connectome-constrained model, Nature 2024](https://doi.org/10.1038/s41586-024-07763-9), [MaleCNS dati/API](https://male-cns.janelia.org/download/).

Regola: `superclass × rootSide`, classe, neuromero, neuropilo = **etichette di partizione**. No chiamare gruppo "esperto semantico" finché non supera:

`osservazione su dati nuovi` + `null topologico/label` + `intervento matched-size` + `replica seed/checkpoint`.

## Immagini anatomiche verificate e lettura della PCA

Link primari per orientare vista:

- [MaleCNS Janelia: overview, rendering laterale, gallery e download](https://male-cns.janelia.org/). Dichiara brain + optic lobes + VNC.
- [Neuroglancer standalone MaleCNS v1.0](https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.jso): layer EM, segmentazione, sinapsi, compartimenti neuropil.
- [Pagina Janelia con immagini, video e link Neuroglancer](https://www.janelia.org/node/70079).
- [Paper MaleCNS, Fig. 1 e caption](https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/): volume, 3D rendering, brain/VNC, cervical connective.
- [Figure MANC: schema brain–VNC–cervical connective](https://elifesciences.org/articles/96084/figures).

Forma PCA dei 3.000 soma **può essere compatibile** con corpo allungato VNC + massa brain/optic-lobes, ma non identifica orientamento, collo, regione. PCA ruota/riflette assi; campionamento uniforme soma mostra occupazione geometrica, non densità sinaptica. Reel non dà etichette per validare orientamento.

Verifica somiglianza: colorare stessi punti con `somaNeuromere`/regione/neuropil e classi DN/AN; confrontare prima brain–neck–VNC, poi coordinate PCA. Separare sempre tre quantità: (a) densità punti/soma, (b) fan-in/fan-out e peso archi, (c) flusso dinamico. "Centro denso" in immagine non implica hub né router.

Candidato anatomico per versione futura: cervical connective/DN–AN come interfaccia routing, perché paper MaleCNS lo descrive come collo di bottiglia e integratore brain↔VNC. Prior strutturale, non esperto linguistico. Prima misurare su ID locali: copertura porte, grado, segno, ablation matched-size, null; no spostare router su base PCA.

## Suite concreta post-20k

### S0 — smoke e integrità, prima di unattended

1. CPU: verificare che checkpoint, config, fingerprint codice/dati e digest restino identici dopo load.
2. GPU singola, FP32/CUDA Graph: forward di 1–2 coppie, reset, replay identico; assert shape, finite, target validi. PAD mascherato; BOS input che predice primo token, EOS target finale. BOS non target delle storie locali.
3. Ripetere due volte stessa probe: logits/CE coincidono entro tolleranza documentata. Registrare peak VRAM, allocator finale, tempo.
4. `graph_specialization_probe.py` read-only: output su file nuovo, controllo digest checkpoint prima/dopo. Smoke piccolo (2 storie pari, 8–16 token) prima di 16×128.
5. Stop isolato se NaN/Inf, digest diverso, cleanup non nullo, cinque step lenti consecutivi, VRAM/RAM oltre tetto. No rilancio automatico stesso worker.

Costo target smoke: <60 s e <1 GB VRAM aggiuntiva. Un solo worker GPU. Statistiche/aggregazioni su CPU.

### S1 — osservazionale: il grafo mostra “competenze” stabili?

Checkpoint: 0k/5k/10k/20k se esistono; altrimenti 20k + un solo punto futuro. DEV fisso già noto solo per sviluppo; no chiamarlo audit nuovo. Riservare 64 storie per audit dopo scelta.

Per ogni gruppo anatomico e per controlli size-matched, raccogliere:

- media/quantili di potenziale, rate, spike fraction, corrente token;
- gradiente vs corrente sorgente e rapporto update/peso, senza confondere gradiente tied di `E` con quello del core;
- copertura input→uscita, SCC, segno, flusso pesato; separare numero nodi/archi da attività;
- probe lineare train-only: token 8-classi, token 4096-classi, contesto breve. Report train/DEV, CE/accuracy, gap probe→head;
- entropia assegnazione gruppo dato token/storia, mutual information, Gini. "Router" qui = **proxy osservazionale**, non router del modello.

Controlli/null obbligatori:

- permutare etichette gruppi mantenendo dimensioni;
- gruppi casuali stessa cardinalità e grado;
- permutare token/target nel probe;
- baseline costante, unigramma, bigramma, GRU d4/d8 con stesso stream/budget;
- regredire o stratificare per size, grado, reachability, lato: gruppo grande non vince solo perché raccoglie più segnale.

Segnale utile: effetto persistente su checkpoint e storie nuove, probe superiore ai null, non spiegato da size/grado. Un solo picco di rate o proxy signed non basta.

### S2 — osservazionale forte: “router anatomico” simulato offline

Senza cambiare pesi o forward, calcolare matrice token/storia × gruppo da energia normalizzata (`|current|`, `|voltage|`, rate). Fit train-only di classificatore/soft router offline; misurare:

- top-1/top-2 group accuracy e log-loss su DEV;
- entropy/load balance per gruppo;
- stabilità fra contesti dello stesso token e fra token diversi;
- generalizzazione a storie riservate.

Null: matrice con group labels permutate; matrice da grado/size soltanto; matrice con stato resettato; router uniforme. Se proxy non generalizza, no router anatomico in versione successiva.

### S3 — causale: interventi temporanei, non adozione

Solo dopo S1/S2 e su checkpoint congelato. Per ogni gruppo candidato, mascherare **temporaneamente in inferenza** output o corrente, preservando resto. Misurare ΔCE, Δlogits, accuracy, entropia, firing, gradiente; poi ripristinare checkpoint.

Controlli causali:

- gruppo casuale size/degree-matched;
- ablation gruppo periferico e gruppo rich-club;
- shuffle etichette senza cambiare mask reale;
- ablation archi stesso conteggio e grado, segno preservato;
- eventuale rewire degree-preserving come null topologico separato.

Interpretazione: Δ grande = dipendenza funzionale locale, non "competenza linguistica". Per specializzazione causale servono interazione gruppo×token, effetto replicato, null superati. Se maschera crea stato fuori distribuzione, segnalarlo e non concludere.

### S4 — test di routing appreso, solo prossima versione

Speculativo, fuori dalla suite autorizzata: aggiungere 2/4/8 esperti piccoli, top-1/top-2, load-balancing loss, router appreso dallo stato. Varianti separate: router libero, router init da partizione anatomica, router con gruppi permutati. Pari parametri attivi, token, update, seed; dense baseline e MoE random-partition obbligatori.

No eseguire S4 ora. Versione corrente ha già 166.700 nodi, 2,754M archi t10 / 6,242M t5, readout 154→256: prima chiudere collo di bottiglia informativo/readout e calibrare budget linguistico.

## Criteri di decisione

| Esito | Decisione |
|---|---|
| Attività concentrata ma probe≈null | Nessun esperto: concentrazione = topologia/size. |
| Probe alto, head bassa, ablation debole | Informazione accessibile; problema readout/ottimizzazione. Nessun router. |
| Probe alto, ablation token-specifica, null falliscono, replica passa | Candidato a prior di routing fase successiva; ancora non "regione linguistica". |
| Soglia5 aggiunge reachability ma non migliora probe/CE o raddoppia costo | Mantieni t10 per suite; no attribuire vantaggio a più archi. |
| Un gruppo domina tutti i token | Possibile hub/shortcut; test con shuffle e degree-matched prima di intervento. |

## Budget e tracciabilità

Per ogni run salvare config, seed, checkpoint hash, dataset/split, target validi, update, token nuovi/ripetuti, durata, peak VRAM/RAM, cleanup, curve per-storia, motivo di stop. Matrice minima: S0 → S1 su 2/16 storie → S2 → S3 su 4 gruppi + 4 null; poi audit64 una sola volta. No grid di optimizer o router.

Pretraining 20k non dimostra specializzazione: con esperienza locale, GRU d4 arriva ~99,55% DEV/2000 update mentre d8 resta ~32,42%; a pari 2000 update T5 L16 d1 arriva DEV75,756%, CE0,67712 vs L8 DEV68,75%, CE0,79938. Cautela sul budget, non prova di superiorità generale. Ogni risultato MoE va confrontato a controllo e budget equivalenti. Probe già registrati (token decodificabile post-grafo; head molto più bassa) suggeriscono priorità a interfaccia/readout, non pruning.

## Raccomandazione finale

**Ora:** grafo intero invariato; S0–S2 read-only su 20k; S3 solo se null cedono. **Dopo:** un MoE piccolo appreso con random-partition baseline e prior anatomico come init/feature, non verità. **Mai dedurre dal reel o dalla PCA che regione CNS sia esperto linguistico.**