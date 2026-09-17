# Report Luna high — soluzioni T2/T3 a grafo conservato

Data: 2026-09-13. Ricerca bounded, fonti primarie, nessun codice modificato, nessun GPU/test. Base locale: [RESOCONTO_T2_T3.md](RESOCONTO_T2_T3.md), [SUITE_DIAGNOSTICA_FASE_3.md](SUITE_DIAGNOSTICA_FASE_3.md), [RESOCONTO_T0_T1.md](RESOCONTO_T0_T1.md).

## Decisione fase3

T2/T3 spostano priorità su **lettura e ottimizzazione**. Token decodificabile dal grafo: probe DEV 92,29% su uscite grezze post, 88,28% dopo pooling, 84,77% su rappresentazione finale; head LM originale 18,85%. T3: firing post sensoriale 44,98%, uscite 0,366%, resto 0,211%, ma potenziale post informativo. Gradiente sensoriale ultimo token 1,55e−4, sei token prima 1,60e−5: attenuato, non nullo. Feedback/token RMS 0,249/0,701: presente, non prova utilità.

Numeri non provano head strutturalmente sbagliata: probe ha 8 classi, standardizzazione train-only, 102.400 target ripetuti, mentre mosca ha visto 2.560 target nuovi. Provano accessibilità segnale, non capacità head di apprenderlo in quel budget. Grafo, ID, segni e vie invariati. Nessun nodo/arco rimosso; nessun nodo dichiarato "linguistico".

T1 resta vincolo: GRU d0 DEV100%; d1 poi calibrato localmente con T1b: stessa GRU/LR/seed, 1.000 update, 2.000 stream nuovi (primi 400 identici al run precedente), train/DEV100%, DEV CE0,0024186, tempo training10,01s. d4/d8 non calibrati/non eseguiti. Bigramma DEV CE4,803 vs unigramma6,012 conferma segnale contestuale locale. Prima di leggere fallimento mosca come limite grafo, separare fattore budget del controllo.

## Cosa dicono le fonti, e cosa trasferiscono

| Tema | Prova primaria | Trasferimento prudente |
|---|---|---|
| Head tied | Press–Wolf e Inan et al. riportano vantaggi legame embedding input/output in LM convenzionali e descrivono aggiornamento condiviso. [Press–Wolf](https://arxiv.org/abs/1608.05859), [Inan et al.](https://arxiv.org/abs/1611.01462) | `z @ E.T` può essere utile, ma gradiente di E somma percorso lookup e output. Non segue che sia ottimo nel grafo mosca. SpikeGPT usa matrice trasposta nel readout; fonte non dimostra da sola condivisione parametro. |
| Probe | Hewitt–Liang mostrano che probe può imparare il compito invece di misurare informazione selettiva; servono controlli e task permutati. [ACL 2019](https://aclanthology.org/D19-1275/) | Probe T2 misura accessibilità codice token. Non sostituisce head, non è bypass nel modello, non dimostra LM. |
| Gradienti concorrenti | PCGrad/CAGrad mostrano conflitto quando più obiettivi condividono parametri. [PCGrad](https://proceedings.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf), [CAGrad](https://proceedings.neurips.cc/paper_files/paper/2021/file/9d27fdf2477ffbff837d73ef7ae23db9-Paper.pdf) | Qui loss è una, ma E, porte, core, attenzione e readout ricevono contributi con scale diverse; "conflitto" richiede coseni/norme misurati per percorso. Non importare subito PCGrad. |
| Readout biologicamente informato | Population decoding mostra che meccanismo readout e pooling cambiano informazione recuperabile. Conn2res separa input e readout per reti anatomiche; nel central complex trovati molti ingressi e pochi canali uscita. [Population decoding](https://pmc.ncbi.nlm.nih.gov/articles/PMC3879135/), [conn2res](https://www.nature.com/articles/s41467-024-44900-4), [central complex](https://elifesciences.org/articles/37017) | Raggruppare per ruolo anatomico è domanda testabile sul readout, non prova attitudine linguistica di regione. Attuali 77 gruppi `superclass × rootSide` sono convenzioni, non neuropili. |
| Credit assignment SNN | SuperSpike, Bellec et al. e NeurIPS 2018 mostrano surrogate/e-prop/BPTT o gradienti per dinamiche ricorrenti e compiti temporali; propagazione difficile ma possibile. [SuperSpike](https://pmc.ncbi.nlm.nih.gov/articles/PMC6118408/), [e-prop](https://www.nature.com/articles/s41467-020-17236-y), [Gradient Descent SNN](https://proceedings.neurips.cc/paper/2018/file/185e65bc40581880c4f2c82958de8cfe-Paper.pdf) | Attenuazione T3 motiva localizzazione gradiente, non cambio immediato algoritmo. e-prop sarebbe cambio training/contratto, non semplice ottimizzazione. |
| Surrogate e init | Ledinauskas et al. trovano vanishing/exploding al variare della surrogate; NeurIPS 2018 mostra dinamiche bilanciate e memoria ritardata in SNN addestrabili. [Ledinauskas et al.](https://arxiv.org/abs/2006.04436), [Zenke–Ganguli](https://pmc.ncbi.nlm.nih.gov/articles/PMC6118408/) | Misurare distanza da soglia, distribuzione potenziali e gradiente prima di scegliere una modifica. Nessun gamma universale trasferibile. |

## Soluzioni senza ridurre il grafo

### 1. Calibrare prima il readout originale, poi separare head tied/untied

Prima prova R0: congelare E e grafo, catturare offline feature del circuito invariato, calibrare soltanto blocchi readout originali (pooling/proiezione `154→256`) con head tied `logits = z256 @ E.T`. Stesso split, 2.560 target nuovi, 200 update T2; probe affine standardizzato resta **diagnostica**, con fit train-only e controllo etichette permutate. Calibrazione offline isola accesso alla rappresentazione.

Poi verificare R0 nel circuito completo: readout alimenta cache e feedback, quindi fit offline non basta a predire comportamento ricorrente. Solo se circuito completo resta limitato, confrontare H0 con H1 mantenendo E e grafo congelati: stesse porte, stesso `z256`, ma matrice output separata `Wout[4096,256]`, inizializzata dalla stessa E al punto zero per isolare vincolo di condivisione. H1 è **cambio architetturale di readout**, aggiunge1.048.576 parametri; costo e budget da esporre.

Per R0/H0/H1: stesso checkpoint iniziale, stream, seed e batch T2. Riportare CE/accuracy 4.096 e ristrette8 su train/DEV, norma logits, gradiente E separato per lookup/output, gap probe→head. Esito utile: calibrazione R0 migliora head nel circuito completo → strozzatura readout; H1 migliora ancora con probe invariato → possibile costo del tying; H0/H1 falliscono mentre probe resta alto → budget, scala o percorso da indagare. Nessun esito autorizza a chiamare i nodi "linguistici".

### 2. Misurare competizione dei gradienti prima di manipolarli

Sul checkpoint T2, senza update, scomporre CE in contributi: percorso output tied, lookup E, porte sensoriali, core, attenzione, pooling/readout. Per ogni blocco: norma e quantili; i **coseni** vanno calcolati soltanto fra contributi nello stesso spazio di parametri, in particolare `E_lookup` contro `E_output`. Non calcolare coseni fra blocchi di dimensioni/spazi diversi. Riportare anche gradiente rispetto a corrente token, non solo a E. Coseno negativo occasionale non basta per "gradient conflict"; serve persistenza su stream DEV fissi.

Se conflitto è solo nella E condivisa, prima variante training-only: una modifica di scala/LR del gruppo E, tutte le altre impostazioni fisse. Se è nelle porte/core, misurare path coverage e gradiente prima di cambiare optimizer. PCGrad resta analogia multi-obiettivo: applicarlo qui altererebbe gradiente CE, esperimento separato, non correzione automatica.

### 3. Porte sensoriali stratificate, grafo invariato

Iniettore attuale usa porte sparse fisse su `sensory`. Candidato S1: stessa dimensionalità e numero connessioni per porta, ma assegnazione deterministica stratificata per `superclass`, `class/subclass`, `entryNerve`, `somaNeuromere` e lato, quando campo esiste; nessun neurone/arco/segno eliminato. S1 cambia **interfaccia architetturale**, non nucleo. Fallback esplicito per annotazioni mancanti; seme e tabella ID→porta salvati.

Misurare copertura componenti BPE, quota ingressi raggiungibili per categoria, separabilità corrente→uscite, gradiente verso sorgenti, CE/accuracy. Se migliora solo probe corrente ma non head, porta trasporta informazione ma lettura resta strozzatura. Stratificazione è scelta di routing da verificare, non mappa di significato regioni.

### 4. Pooling per ruolo disponibile, poi neuropilo quando mappato

Pooling P0 attuale: 2.241 uscite selezionate, 77 gruppi artificiali, potenziale+rate →154. P1: gruppi fissi per ruolo locale (`superclass`, `class/subclass`, `somaNeuromere`, `entryNerve/exitNerve`, lato), con potenziale e rate separati e proiezione compatta finale di dimensione invariata. P1 cambia **readout/pooling**, non grafo. Non usare `type` ad alta cardinalità senza tetto predefinito: altrimenti head guadagna memoria parametrica e confronto perde significato.

Mapping neuropilo esplicito richiede syn-partners/ROI MaleCNS; fonte primaria descrive classi, neuromeri, nervi e circuiti sensoriali→motori, mentre annotazioni locali ispezionate non hanno colonna neuropilo. Prima validare ID e copertura con tabella/API ufficiale, poi un solo confronto P0/P1. [MaleCNS primario](https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/), [dati/API Janelia](https://male-cns.janelia.org/download/)

Lettura deve conservare segno, direzione e doppia osservazione potenziale+rate. Che central complex o altri circuiti abbiano bottleneck pubblicati non implica idoneità al testo: serve solo a motivare moduli di readout separati e misurabili.

### 5. Percorsi completi e credit assignment

Con grafo fisso, per ogni categoria sensoriale e gruppo output: reachability diretta, distribuzione lunghezze cammino, SCC/WCC, contributo eccitatorio/inibitorio, gradiente CE lungo percorso. Audit del routing, non pruning. Output senza cammino o senza gradiente può indicare porta/pooling non raggiunto; cammino con gradiente attenuato indica problema dinamica/credit assignment.

NeurIPS 2018 dimostra in SNN piccoli che dinamica può conservare memoria ritardata e che training può operare su scale temporali diverse; e-prop propone tracce di eleggibilità come alternativa a BPTT. Prove su altri task e architetture. Per questa fase: mantenere autograd/surrogate attuali, localizzare prima; e-prop, feedback alignment o bypass sarebbero nuovi cambi training/architettura.

### 6. Due soglie che non vanno confuse

`threshold10`/`threshold5` nel dataset sono **soglie strutturali sul conteggio sinapsi per arco**: cambiano quali connessioni entrano nel grafo e la normalizzazione. Soglia LIF `v_th` è **soglia dinamica di attivazione**: decide spike/reset durante forward. Non sono la stessa variabile, non vanno sintonizzate nello stesso esperimento.

In questa fase tenere grafo di riferimento, nessun pruning/riduzione. Se un giorno si testa `v_th`, è variante dinamica/training con topologia fissa; se si cambia soglia conteggio, è variante del grafo e richiede nuova inizializzazione/controllo. Firing basso uscite T3 non giustifica né t5 né abbassamento automatico di `v_th`, perché potenziale resta decodificabile.

### 7. Surrogate e inizializzazione LIF

T3 mostra potenziali finiti, distribuzione disomogenea, gradiente sensoriale attenuato. Candidato training-only D1: mantenere tutto fisso e cambiare una surrogate (forma o larghezza); no griglia. Candidato training-only D2: mantenere surrogate e usare inizializzazione del potenziale/corrente calibrata sulla distribuzione iniziale osservata, senza target di firing inventati.

Budget per D1 o D2: 200 update, 2.560 target nuovi, stesso seed/stream T2. Misurare quantili `v−v_th`, frazione dentro zona surrogate, gradiente per gruppo, rapporto update/peso, firing valido, CE, probe. Cambiare potenziale/corrente iniziale o surrogate richiede confronto esplicito con l'utente prima di qualunque run, anche con topologia invariata. Se surrogate migliora gradiente ma peggiora probe/CE, non adottare; se D2 cambia solo firing senza aumentare informazione leggibile, non adottare. Letteratura mostra sensibilità del training, non valore corretto per questo LIF.

## Addendum T4/T5 — interpretazione prudente

ID autorevoli e dettagli: [SUITE_SOLUZIONI_FASE_3.md](SUITE_SOLUZIONI_FASE_3.md). Probe K/V T4: DEV75,98%/74,02%, permutato11,72%/13,67%; K/V portano informazione token sopra controllo. Sul checkpoint TinyStories storico a16 prefissi DEV, CE reale5,711880, cache vuota5,712096, contenuto reverse5,711734: differenze minime, reverse persino leggermente migliore, quindi nessuna prova di uso causale della cache. Perturbazione congiunta K/V/posizioni: errore massimo2,38e−6, zero cambi argmax. T5 d1 a update uguali: L8 DEV27,82%, CE2,01442; L16 DEV26,26%, CE2,01879: segnali discordanti, nessuna promozione L16. d8 ancora in corso. T1 d1 chiuso localmente a100% con1k update.

## T1, T4, T5: cenni minimi

**T1b, calibrazione d1 già eseguita:** stessa GRU/LR/seed, 1.000 update e 2.000 stream nuovi, con primi 400 identici al run precedente; train/DEV100%, DEV CE0,0024186, 10,01s. Poiché è cambiato anche il pool stream, non attribuire il delta ai soli update; calibrazione operativa chiusa, causalità del budget resta separata. Vecchia proposta 200→512 non eseguita, superata. d4/d8 fuori.

**T4:** stesso checkpoint e prompt; cache reale, cache temporalmente permutata, cache vuota. Nessun update. Misurare Δlogits/CE e probe su K/V; effetto grande significa dipendenza dalla cache, non utilità garantita.

**T5:** L8 e L16 con update uguali, stessi 16 token, pesi iniziali, reset, LR e stream; cambia solo il detach interno. Riportare CE/accuracy e gradiente verso token sorgente. Nessun confronto con più update o topologia diversa.

## Proposte uno-fattore e criteri

| ID | Classe | Unico fattore | Budget | Criterio osservabile |
|---|---|---|---|---|
| G0 | Diagnostica | nessun update: decomposizione gradienti per percorso | checkpoint T2, stream DEV fissati | norme/coseni, gradiente sorgente, path coverage; decide se serve training o interfaccia |
| C1 | Training | T1b già eseguito: d1 a 1.000 update/2.000 stream | stessa GRU/LR/seed | train/DEV100%, DEV CE0,0024186; d4/d8 non calibrati |
| R0 | Training/readout | calibrare blocchi readout originali con E+grafo congelati, poi verifica full circuit | 200 update, 2.560 target | offline vs circuito completo; decide se strozzatura è readout |
| H1 | Architettura readout | solo dopo R0: tied H0→output untied H1, E+grafo congelati | 200 update, 2.560 target | head vs probe, +1.048.576 parametri, gradiente E; decide vincolo tied vs ottimizzazione |
| S1 | Architettura interfaccia | porte random→stratificate | 200 update, grafo fisso | copertura/gradienti per categoria e CE; nessuna rimozione nodi |
| P1 | Architettura readout | pooling artificiale→ruoli locali | 200 update, dim output fissata | probe pre/post pooling, CE, segni, coverage; neuropili solo dopo mapping verificato |
| D1/D2 | Training/dinamica; confronto utente richiesto | surrogate **oppure** init, mai entrambi | 200 update, 2.560 target | zona soglia, gradienti, firing, CE; non scegliere da firing isolato |

## Priorità e limite di conoscenza

1. **G0 + T1b:** misurare competizione/path; d1 calibrato localmente, d4/d8 no.
2. **R0:** calibrare readout originale offline con E/grafo congelati, poi verificare nel circuito completo.
3. **H1, quindi S1 oppure P1:** tied/untied e una modifica di interfaccia scelta dai risultati G0; mantenere tutti nodi e connessioni.

T4/T5 restano prove brevi per memoria; non riaprire matrice di esperimenti. Nessuna fonte dimostra che MaleCNS impari linguaggio, nessuna assegna attitudine linguistica innata a una regione, nessun risultato qui promette vittoria benchmark. Domanda chiudibile in fase3 è più stretta: **il segnale token arriva e resta allenabile attraverso porte, dinamica e readout su grafo conservato?**

## Fonti primarie usate

- [Press & Wolf, output embedding e weight tying](https://arxiv.org/abs/1608.05859)
- [Inan, Khosravi & Socher, tying word vectors/classifiers](https://arxiv.org/abs/1611.01462)
- [Hewitt & Liang, probes con control tasks](https://aclanthology.org/D19-1275/)
- [Yu et al., PCGrad](https://proceedings.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)
- [Liu et al., CAGrad](https://proceedings.neurips.cc/paper_files/paper/2021/file/9d27fdf2477ffbff837d73ef7ae23db9-Paper.pdf)
- [Zenke & Ganguli, SuperSpike](https://pmc.ncbi.nlm.nih.gov/articles/PMC6118408/)
- [Bellec et al., e-prop](https://www.nature.com/articles/s41467-020-17236-y)
- [Huh & Sejnowski, Gradient Descent for SNN](https://proceedings.neurips.cc/paper/2018/file/185e65bc40581880c4f2c82958de8cfe-Paper.pdf)
- [Ledinauskas et al., surrogate/gradient stability](https://arxiv.org/abs/2006.04436)
- [Population decoding, readout mechanism/pooling](https://pmc.ncbi.nlm.nih.gov/articles/PMC3879135/)
- [Connectome-based reservoir computing](https://www.nature.com/articles/s41467-024-44900-4)
- [Franconville et al., Drosophila central-complex functional connectome](https://elifesciences.org/articles/37017)
- [Berg et al., complete male CNS connectome](https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/)
- [MaleCNS v1.0 data/API](https://male-cns.janelia.org/download/)

Classificazione: **prova** = risultato riportato dal paper; **analogia** = principio trasferito con limiti; **ipotesi locale** = variante da testare sul nostro grafo/contratto.