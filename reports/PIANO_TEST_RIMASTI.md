# Piano dei test — 17 settembre 2026, 21:45 (tabella con colonna Risultato; coda 8b ferma su richiesta utente, riprendibile)

Fonte di verità per "cosa manca": numerazione originale della lista concordata (15 settembre, ripresa in `SUITE_TEST_FASE_6B.md` con le lettere A-K) più le idee della ricerca (`RICERCA_CONNETTOMA_LM.md` §1(c), §1(d), §4) e le leve della ricerca sull'attività (`RICERCA_ATTIVITA_CERVELLO_MOSCA.md`). Ogni voce: stato, dove sta il risultato, costo. "Corto" = ≤ 2000 update (≤ 45 min GPU) oppure CPU / sola inferenza; "lungo" = 8000 update (≥ 55 min) oppure implementazione da mezza giornata in su.

## 0. Migliore configurazione attuale

A 2000 update: baseline standard + 8+8 sottopassi + sinapsi a conduttanza + tau per tipo (lr 1e-2) + Muon 1e-3 sulle matrici dense = **4,152** (K5, `phase7_lever_best_2000`), contro 4,580 della baseline standard. Non provata: a pari tempo GPU (1277 ms/update, 3,5 volte Muon da solo) e a 8000 update. Baseline standard a 8000 (K0) = 3,652, uguale al configuration model (3,648): 8+8 a 8000 vale +0,061, conduttanza +0,029. Muon a 8000: 3e-4 3,571 (+0,081), 1e-3 **3,541** (+0,112, K8: prima leva sopra soglia a 8000, sotto il Transformer piccolo 3,563 che però è allenato con Adam); Muon 3e-3 a 2000 4,250 (plateau con 1e-3, non adottato). Candidati confermati per il default: Muon 1e-3 (a 8000), 8+8, conduttanza, tau 1e-2 (a 2000, da confermare a 8000 con L1).

## 1. Numerazione originale, stato voce per voce

I numeri 1.1, 1.4, 2.3.7, 2.3.8, 2.4.1, 2.4.2, 2.5.2, 2.5.3, 2.7.2 non hanno testo in nessun file: ricostruiti dalla struttura della lista (ogni blocco 2.x segue l'ordine delle idee di `RICERCA_CONNETTOMA_LM.md`: 2.4.x = §1(c) "parametrizzazione", 2.5.x = §1(d) e idee H/I/J, 2.3.x = idea K "anatomia nativa") e marcati **[ricostruito]**.

| N. | Voce | Stato | Dove |
|---|---|---|---|
| 1.1 | Grafo rimescolato a gradi conservati, seed 41 **[ricostruito]** | fatto 14/9 | RESOCONTO_FASE_6_CONTROLLI §1 (3,814 contro 3,959 a 8000) |
| 1.2 | Secondo seed di rimescolamento | fatto | B1 |
| 1.3 | Sonda lineare sullo stato | fatto | C1 |
| 1.4 | GRU e Transformer a pari token **[ricostruito]** | fatto 14/9 | RESOCONTO §2 (3,429 / 3,563) |
| 1.5 | Configuration model | fatto | B2 (4,533; 3,648 a 8000) |
| 2.3.1 | Copertura degli emilineaggi | fatto (analisi) | D1 |
| 2.3.2 | Raggiungibilità porte → readout | fatto (analisi) | D2, F8 |
| 2.3.3 | Firing rate per superclasse e classe | fatto (analisi) | C2 |
| 2.3.4 | Null entro compartimento | fatto | B6 (5,005) |
| 2.3.5 | Readout per raggruppamento anatomico (pool motorio) | fatto | D3 (+0,24 a 2000; falso positivo a 8000) |
| 2.3.6 | Porte sensoriali anatomiche + controllo a pari capacità | fatto | B7, F1, F6, G2 |
| 2.3.7 | Condivisione parametri per emilineaggio (idea K1) **[ricostruito]** | NON fatto | costo: etichette emilineaggio (copertura nota solo per il VNC) + gather; run corto |
| 2.3.8 | Null "emilineaggio rimescolato" (idea §1(d) 6) **[ricostruito]** | NON fatto | ha senso solo insieme a 2.3.7; run corto |
| 2.3.9 | Lesione del rich club | fatto (analisi) | C3 |
| 2.3.10 | Lesione della sottorete dimorfica | fatto (analisi) | C4 |
| 2.4.1 | Condivisione parametri per tipo cellulare (stile flyvis, 734 parametri) **[ricostruito]** | parziale | tau e bias per tipo fatti (H1, H2, J1, K2); i fattori di scala per coppia tipo-tipo al posto del peso per arco NO: cambia il core, decisione utente |
| 2.4.2 | Condivisione parametri per emilineaggio **[ricostruito]** | NON fatto | come 2.3.7 (stessa voce vista dal lato parametri) |
| 2.4.3 | Riordino dei nodi per comunità | fatto | E1, S8 (non equivalente: porte e gruppi sull'indice) |
| 2.4.4 | Sparsità APL | fatto | D4 (neutro) |
| 2.4.5 | Unità graduate nel lobo ottico | fatto | D5 (neutro) |
| 2.4.6 | Costanti di tempo per tipo | fatto | D6 → H2 → J1 → K2 (+0,171 con lr 1e-2) |
| 2.4.7 | Potenziali di inversione (conduttanza) | fatto | D7 → H3 (+0,121) |
| 2.4.8 | Soglia sinaptica relativa | fatto | D8 (+0,13), G3 (6,24 M archi +0,05) |
| 2.4.9 | Segni morbidi 6% | fatto | D9 (+0,11; 3,897 a 8000) |
| 2.4.10 | FP16 / INT8 | fatto (analisi) | E2 (non conviene) |
| 2.5.1 | Profilo di dipendenza dal contesto (H) | fatto | A1, G0i, J3 |
| 2.5.2 | Fattoriale {reale, rimescolato} × {attention ON, OFF} in allenamento (idea I) **[ricostruito]** | NON fatto | solo a inferenza (kv_0 in A1); serve allenare senza attention: 2 run corte |
| 2.5.3 | Nucleo sostituito a pari parametri (idea J: MLP denso, identità senza archi, config model) **[ricostruito]** | parziale | config model fatto (B2); MLP denso e identità NO |
| 2.5.4 | Pesi permutati sulla topologia reale | fatto | B3 (5,103: nullo) |
| 2.5.5 | Segni di Dale permutati | fatto | B4 (4,645) |
| 2.5.6 | Mappa di reclutamento token → popolazione | fatto (analisi) | C6 (per superclasse; per tipo cellulare da rifare) |
| 2.5.7 | Lesioni per popolazione in nat | fatto (analisi) | C5, G0i, J3 |
| 2.5.8 | Localizzazione dell'attività | fatto (analisi) | C7 |
| 2.5.9 | Gap train/dev e rumore in ingresso | fatto (analisi) | C8 (corretto in G0i) |
| 2.5.10 | Ritardi sinaptici anatomici | NON fatto | preprocessing di una giornata (`syn-partners` 6,8 GB); con 8 sottopassi risoluzione 2-3 bin |
| 2.6.1 | Scala di null: Erdős-Rényi | fatto | B5 (4,676; 3,697 a 8000) |
| 2.6.2 | Compito affine alla mosca | NON fatto | task sensorimotorio da disegnare: lungo |
| 2.6.3 | Sonda Kenyon cell | fatto (analisi) | C9 |
| 2.6.4 | Sonda ring attractor / posizione | fatto, fallita | C10 (r² negativo; da rifare) |
| 2.6.5 | Raggio spettrale e persistenza | fatto (analisi) | C11, G0i |
| 2.6.6 | e-prop pilota | NON fatto | tracce di eleggibilità dentro `Propagate`: lungo |
| 2.7.1 | Distribuzione dei gate e correnti di feedback | fatto (analisi) | A2 |
| 2.7.2 | Guadagno allenabile per ROI + CE per ROI (idea D) **[ricostruito]** | NON fatto | ~100 parametri, logging; run corto |
| 2.7.3 | PAPA (attention media, input-indipendente) | fatto | dentro A1 |
| 2.8.1 | Curve CE contro token in log-log | fatto (analisi) | A4 |
| 2.8.2 | Metrica prequential per update | NON fatto | logging per update nel trainer (impronte checkpoint): medio |
| 2.8.3 | Loss per posizione nella finestra | fatto (analisi) | A5 (da rifare con offset random) |
| — | Idea B: regime reservoir (nucleo congelato, solo interfacce) | NON fatto | run corto (2000, ~10 min: niente backward nel core) |
| — | Idea K5: weight tying per omologia seriale (968 serial set) | NON fatto | medio: etichette serial set + tying dei parametri per tipo |
| — | Idea K6: tau LTC (dipendente dall'input) per emilineaggio | NON fatto | medio-lungo; rischio sul gradiente surrogato |
| — | Idea K7: T dall'anatomia | fatto come 8+8 (G1, +0,181); 12+12 NO | run corto (~35 min con Muon) |
| — | Leva ricerca 4: sinapsi con dinamica temporale (α-synapse + ritardo 1 sottopasso) | NON fatto | stato in più nel core (variante): medio; run corto |
| — | Leva ricerca 7b: rumore in ingresso calibrato | NON fatto | RNG fuori dal CUDA Graph (rumore per batch dalla CPU): medio |
| — | Proposta utente 17/9 17:30: memoria a pesi veloci nel corpo fungiforme (KDA anatomica: chiavi = Kenyon cell, valori = MBON, gate = DAN per compartimento) | NON fatto | corti M1, M2, M4; medio M3 (vedi §2 M/N) |
| — | Proposta utente 17/9 17:30: giri multipli per token con attention dentro il giro (looped), sottopassi sequenziali per zona, layer anatomici per zona | NON fatto | corti N1, N2; medio N3 (vedi §2 M/N) |
| — | Idee M della ricerca (ring attractor al posto di RoPE, terzo fattore DAN, distillazione) | rimandate | lunghe |

## 2. Tutti i test, per categoria, corti e lunghi, con risultato

Regola dell'utente (17/9): ogni test corto che dà un risultato entra fra i candidati per il default e va provato dentro la combinazione; l'adozione nello standard la decide l'utente. "Corto" = run ≤ 2000 update oppure CPU / sola inferenza; "lungo" = 8000 update o implementazione da mezza giornata in su. Durate MISURATE il 17/9 sulla GTX 1080 col kernel fuso: 300 update ≈ 5 min; 2000 update a 4+4 15 min; a 8+8 27 min; a 12+12 35 min; configurazione K5 46 min; 8000 update a 4+4 56 min; a 8+8 100 min; suite di inferenza 27 min. Base dei corti dal 17/9 sera: candidato sicuro per il default = standard + Muon 1e-3 (4,275 a 2000; 3,541 a 8000; baseline standard 4,580 / 3,652). Soglia di effetto 0,10 nat; rumore fra run e fra semi 0,02. Stato: FATTO = eseguito e registrato nella suite; IN CODA = coda 8b (riprendibile); DA FARE = non ancora implementato o non in coda.

### A. Canale esterno e budget

Corti

| N. | Test | Tecnica | Concetto | Tempo | Risultato |
|---|---|---|---|---|---|
| A5b | Loss per posizione con offset random | Finestre da 16 token a offset casuale, indice loss(pos 16) − loss(pos 1) sullo stesso checkpoint. | L'uso del contesto misurato è reale o è l'allineamento delle finestre? | 10 min | DA FARE |
| A2.7.2 | Guadagno allenabile per superclasse | Uno scalare per superclasse (init 1, lr 1e-2) sulla corrente esterna; flag core `gain_group`. | Quali regioni il testo alza o spegne. | 15 min | IN CODA |

Lunghi

| N. | Test | Tecnica | Concetto | Tempo | Risultato |
|---|---|---|---|---|---|
| 2.8.2 | Prequential per update | Loss per update sullo stesso stream per reale, rimescolato, GRU, Transformer; bit totali. Serve logging nel trainer. | Confronto onesto "quanti bit in tutto". | mezza giornata | DA FARE |

### B. Null e controlli sul grafo

Corti

| N. | Test | Tecnica | Concetto | Tempo | Risultato |
|---|---|---|---|---|---|
| B8 | Replica con seed delle porte 23 | Nello standard le porte sono random: altro seed. | Quanta varianza viene dalla scelta casuale delle porte. | 15 min | IN CODA |
| B9 | Porte random a pari capacità: 2.639 (olfatto) e 1.733 (udito) | Stesso default con meno porte random. | Separare "anatomia" da "numero di porte" nei test F. | 2 × 15 min | IN CODA |
| B2.5.3a | Nucleo identità | Pesi sinaptici a 1e-6 e congelati: neuroni isolati, interfacce uguali. | Il pavimento "senza grafo". | 15 min | IN CODA |
| B10 | Reservoir | Pesi sinaptici congelati all'init anatomica. | La topologia fissa porta informazione senza allenare gli archi? Attesa ≈ default (C13w). | 15 min | IN CODA |
| B2.5.2 | Fattoriale grafo × attention | Reale / rimescolato a soglia 10 × attention accesa / spenta (4 run). | Interazione: il reale dipende dall'attention più del rimescolato? | 4 × 15 min | IN CODA |
| 2.5.3b | MLP denso a pari parametri | MLP da 2,75 M parametri al posto del core. | Il connettoma vale più di un blocco denso qualunque? | 1-2 h codice + 10 min | DA FARE |
| 2.3.8 | Null "emilineaggio rimescolato" | Etichette permutate a pari dimensione; solo con 2.3.7. | Serve la partizione biologica? | 15 min | DA FARE |

Lunghi

| N. | Test | Tecnica | Concetto | Tempo | Risultato |
|---|---|---|---|---|---|
| B11 | Promozioni a 8000 dei controlli sopra soglia | Regola della suite, contro 3,652. | Regge con l'allenamento? | 56 min l'una | DA FARE |
| B12 | Null del grafo a pari architettura looped | Rimescolato dentro la variante N vincente. | Con attention nel giro, chi lavora? | 56 min | DA FARE |
| B13 | Scala dei null con sinapsi che imparano | Reale, rimescolato, configuration model rifatti col lr sinaptico scelto da M0. | Finora si confrontavano reservoir (C13w). | 3 × 15 min + 3 × 56 min | DA FARE (dopo M0) |

### C. Sonde, mappe e lesioni

Corti

| N. | Test | Tecnica | Concetto | Tempo | Risultato |
|---|---|---|---|---|---|
| C12 | Raggio spettrale all'init + errata | Iterazione di potenza su CPU, init e checkpoint. | L'allenamento cambia il guadagno ricorrente? | 5 min | FATTO: 0,500 all'init per costruzione, identico sui checkpoint; 0,405 sui neuroni attivi. ERRATA: i valori di C11/G0i/J3 (0,53 → 0,73 → 0,35) erano fasi di un'oscillazione, nulli. |
| C13w | Movimento dei pesi sinaptici | ‖w − w₀‖/‖w₀‖ sui checkpoint. | Il cervello impara o è una scatola fissa? | 5 min | FATTO: 0,11% a 1000, 0,30% a 8000; 2,9% degli archi mossi. Core ≈ reservoir. |
| D2b | Raggiungibilità pesata per segno | Propagazione lineare con segno da porte a lettura, all'init. | Perché certi ingressi funzionano. | 1 min | FATTO: porte random raggiungono l'85% della lettura in 1 salto; occhi in 2-3 salti con segni che si cancellano (coerenza 0,07). |
| C15 | Sinapsi rimesse all'init | CE del checkpoint K8 con i pesi sinaptici riportati all'init; copertura per superclasse. | Quanto lavoro fa l'apprendimento sinaptico. | 5 min | IN CODA |
| C6b | Reclutamento per tipo cellulare | Tassi per 11.752 tipi per classe di token. | Mappa funzionale a risoluzione di tipo. | 10 min | DA FARE |
| C10b | Sonda posizionale rifatta | Ridge con cross-fit su EB/PB/FB contro gruppi random. | La mosca ha un codice di posizione? | 10 min | DA FARE |
| C13 | Inferenza sui checkpoint con varianti | Estendere `phase6c_inference.py` alle varianti; suite su K5. | Cosa fa dentro il modello migliore. | 30 min codice + 27 min | DA FARE |
| D3b | Lettura sul potenziale (hub) | Readout da v invece che dal tasso. | Gli hub muti portano informazione nel potenziale? | 15 min | DA FARE |

Lunghi: nessuno.

### C-bis. Quanto cervello impara (scoperta del 17/9; richiesta utente: non lavorare su un campione di pochi neuroni)

Corti

| N. | Test | Tecnica | Concetto | Tempo | Risultato |
|---|---|---|---|---|---|
| M0 | Micro-sweep del passo sinaptico | 9 run da 300 update: default, lr dedicato dei pesi 1e-3 / 3e-3 / 1e-2 / 3e-2, con gradiente binario e allargato; si guarda il movimento dei pesi, non la CE finale. | Trovare il passo con run da 5 minuti prima di spendere run da 2000. | 9 × 5 min | IN CODA |
| C14 | lr dedicato per le sinapsi | Gruppo Adam separato per `core.raw`, 1e-3 e 1e-2. | Le sinapsi non imparano perché il passo è troppo piccolo? | 2 × 15 min | IN CODA |
| C16 | Gradiente allargato (`soft_gw`) | Nel backward il gradiente del peso usa l'attività presinaptica morbida; da solo e con lr 1e-3 / 1e-2. | Far imparare anche le sinapsi dei neuroni che non sparano. | 3 × 15 min | IN CODA |
| H1o | Omeostasi di risveglio per tipo | Scarto di soglia senza gradiente per ognuno degli 11.752 tipi; sotto il 2% di attività la soglia scende piano, mai sotto il normale. A 4+4 e a 12+12. | Il cervello regola da solo chi svegliare. | 15 + 35 min | IN CODA |
| H2a | Arousal a impulsi e a onda | Scarto di soglia globale schedulato: impulso (1,0 → 0,7 per 100 update ogni 500) o onda sin² senza gradini. | Reclutamento periodico a costo di 4+4; l'onda evita il cambio brusco. | 2 × 15 min | IN CODA |

Lunghi

| N. | Test | Tecnica | Concetto | Tempo | Risultato |
|---|---|---|---|---|---|
| C17 | Conferma a 8000 del regime in cui le sinapsi imparano | lr sinaptico (e/o `soft_gw`, omeostasi) scelto dopo M0, a 8000. | Allenare davvero il cablaggio alza il tetto? | 56 min | DA FARE (dopo M0) |

### D. Natura del connettoma come parametrizzazione

Corti

| N. | Test | Tecnica | Concetto | Tempo | Risultato |
|---|---|---|---|---|---|
| L5 | Default + 12+12 sottopassi | 24 passaggi per token. | La profondità oltre il diametro del grafo (13) aiuta? | 35 min | FATTO: 4,162 (+0,113 sul default, sopra soglia; +0,025 su 8+8: satura). Costo ×2,5: a pari tempo perde. |
| L7 | K5 con scala pesi ×0,5 | Peso iniziale dimezzato nella configurazione completa. | Leva minore, si somma? | 46 min | FATTO: 4,132 (+0,020 su K5: dentro il rumore). Non candidata. |
| M4 | Porte a conduttanza | Corrente esterna × (E − v)/E; flag `cond_ports`. | Chi è vicino a soglia riceve di più. | 15 min | IN CODA |

Medi (ore di codice, run corto)

| N. | Test | Tecnica | Concetto | Tempo | Risultato |
|---|---|---|---|---|---|
| 2.3.7 | Parametri per emilineaggio | tau, bias e guadagno per 35-200 gruppi (etichette solo VNC). | Granularità sopra il tipo. | 3-4 h + 15 min | DA FARE |
| D10 | Sinapsi con dinamica temporale | α-synapse 1-2 sottopassi + ritardo 1. | Sommazione temporale. | 3-4 h + 20 min | DA FARE |
| D11 | Rumore in ingresso calibrato | σ 0,05 / 0,1 / 0,2 da un banco di rumore su GPU. | Regolarizzazione biologica. | 2 h + 3 × 15 min | DA FARE |
| D12 | Weight tying seriale | 5.830 neuroni, 968 serial set. | Sfrutta la ripetizione segmentale? | 3 h + 15 min | DA FARE |

Lunghi

| N. | Test | Tecnica | Concetto | Tempo | Risultato |
|---|---|---|---|---|---|
| 2.4.1 | Pesi per coppia tipo-tipo | ~10⁵ fattori al posto di 2,75 M pesi per arco. | Quanto si perde a 10⁵ parametri. | 1 giorno + 56 min | DA FARE |
| 2.5.10 | Ritardi sinaptici anatomici | Da morfologia (`syn-partners` 6,8 GB). | Il tempo di propagazione conta? | 1-2 giorni | DA FARE |
| K6r | tau LTC per emilineaggio | Costanti dipendenti dall'input. | Dinamica adattiva. | 1-2 giorni | DA FARE |
| 2.6.2 | Compito affine alla mosca | Task sensorimotorio da disegnare. | Impara meglio ciò per cui è cablata? | giorni | DA FARE |
| 2.6.6 | e-prop pilota | Tracce di eleggibilità in `Propagate`. | Apprendimento locale. | giorni | DA FARE |

### E. Sistemi

| N. | Test | Tecnica | Concetto | Tempo | Risultato |
|---|---|---|---|---|---|
| S9 (lungo) | Backward sparso | Salta gli archi con presinaptico muto. | Velocità ×1,5 potenziale. | 1 giorno | DA FARE |
| E3 (decisione) | Kernel fuso sul modello principale | `FusedCore` in `fly_core`, ×1,73 identico. | Velocità gratis. | 0 | decisione utente |

### F. Regioni specializzate

| N. | Test | Tecnica | Concetto | Tempo | Risultato |
|---|---|---|---|---|---|
| F9 (corto) | Porte anatomiche corte + 8+8 + Muon 1e-3 | Udito/tatto + proiezione visiva come ingressi, nella configurazione migliore. | Gli ingressi veri smettono di costare? | 28 min | FATTO: 4,332 (−0,145 su L4 con porte random; era −0,279 con Adam). Costano ancora, la metà. |
| F3L (lungo) | Occhi + fru/dsx a 8000 | F3 col kernel fuso. | L'unica coppia anatomica regge? | 56 min | DA FARE |

### K / L. Follow-up delle vincitrici

Corti

| N. | Test | Tecnica | Concetto | Tempo | Risultato |
|---|---|---|---|---|---|
| K7 | Muon 3e-3 | lr di Muon ancora più alto. | lr ottimo più su? | 15 min | FATTO: 4,250 (+0,026 su 1e-3: plateau). |
| L4 | Default + 8+8 | Profondità sopra Muon 1e-3. | Quanto resta alla profondità. | 27 min | FATTO: 4,187 (+0,088, sotto soglia; con Adam era +0,181). |
| L3 | Replica K5 seed 23 | Stessa configurazione, altro seed dei dati. | 4,152 è solido? | 46 min | FATTO: 4,173 (differenza 0,021 = rumore). |
| L2 | Pari tempo GPU | Dai dati di K8. | Le leve pesanti pagano al secondo? | 0 | FATTO: nel tempo di K5 a 2000 (4,152) il default fa 7000 update = 3,615. |

Lunghi

| N. | Test | Tecnica | Concetto | Tempo | Risultato |
|---|---|---|---|---|---|
| K0 | Baseline standard a 8000 | Riferimento per tutte le promozioni. | — | 56 min | FATTO: 3,652 (= configuration model 3,648). |
| K6 | Muon 3e-4 a 8000 | Promozione. | Muon accelera o alza il tetto? | 56 min | FATTO: 3,571 (+0,081). |
| K8 | Muon 1e-3 a 8000 | Promozione. | Idem, col lr giusto. | 56 min | FATTO: 3,541 (+0,112: unica leva sopra soglia a 8000; sotto il Transformer piccolo 3,563 allenato con Adam). |
| L1 | K5 a 8000 | Configurazione completa da zero. | Le leve di dinamica aggiungono sopra Muon a 8000? | ~3 h (stima) | DA FARE |

### M. Attention anatomica (corpo fungiforme come memoria a pesi veloci)

| N. | Test | Tecnica | Concetto | Tempo | Risultato |
|---|---|---|---|---|---|
| M1 (corto) | Memoria MB + MHA | Pesi veloci KC→MBON con regola delta, gate DAN. | Il modello usa una memoria anatomica? | mezza giornata + 20 min | DA FARE |
| M2 (corto) | Memoria MB al posto della MHA | Idem, attention spenta. | Il contesto può stare nell'anatomia? | 20 min | DA FARE |
| M3 (lungo) | Variante fedele | Solo archi KC→MBON reali, gate per 15 compartimenti. | Le sinapsi vere bastano come memoria? | 1-2 giorni | DA FARE |

### N. Looped, giri e profondità

Corti

| N. | Test | Tecnica | Concetto | Tempo | Risultato |
|---|---|---|---|---|---|
| N0 | Diagnostica fra le letture | Stato di lettura e attention ipotetica dopo ogni sottopasso, sul checkpoint K8. | Se lo stato non cambia, rileggere non serve. | 10 min | IN CODA |
| N6c | Trasferimento di profondità | Checkpoint allenati a 4+4 / 8+8 / 12+12 valutati alle altre profondità. | Quanto salta la CE a ogni cambio di profondità. | 3 × 3 min | IN CODA |
| N6d | Shock di profondità (idea utente) | 12 → 8 → 4 → shock a 12, fasi da 500 update, un solo grafo con sottopassi congelati. | Lo shock accende neuroni e sinapsi nuove? | 35 min | IN CODA |
| N1 | Due letture per token | Attention dopo i sottopassi 2 e 6, cache per lettura, identificativo di passo. | Giro ottimo in pretraining = 2 (letteratura). | 15-20 min | IN CODA |
| N2 | Attention a ogni sottopasso | 7 letture, cache per lettura, injection per concatenazione. | Attention come parte della dinamica (Huginn/Ouro alla lettera). | 20-25 min | IN CODA |
| N3 | Letture a ogni sottopasso, cache della prima lettura | Recursive KV sharing (MoR). | Servono stati distinti per sottopasso? | 20-25 min | IN CODA |
| N4 | Due letture senza identificativo di passo | Ablazione. | −0,107 in letteratura: anche qui? | 15-20 min | IN CODA |
| N5 | Token solo al primo sottopasso | Ablazione dell'input injection. | L'informazione viaggia nello stato o nell'iniezione? | 15 min | IN CODA |
| N6b | Profondità ciclica per token | 12-8-4 per token, valutazione a 4 / 8 / 12. | Un modello a tre velocità. | 27 min + 3 inferenze | DA FARE (proposto) |
| N7 | Sottopassi sequenziali per zona | Aggiornamento sensoriali → centrali → MB/CX → discendenti. | Più zone per sottopasso a costo 4+4. | mezza giornata + 15 min | DA FARE |
| N12 | Prelude e coda con pesi propri | Prima e ultima lettura separate. | Le letture di bordo lavorano diverso. | 2 h + 25 min | DA FARE |

Medi

| N. | Test | Tecnica | Concetto | Tempo | Risultato |
|---|---|---|---|---|---|
| N8 | Stato di lettura più ricco | Discendenti + KC + MBON + CX → 512, una lettura. | Collo di bottiglia: letture o capacità dello stato? | 2 h + 20 min | DA FARE |
| N9a | Fattoriale grafo × letture a 2000 | {reale, null gradi+pesi ×3} × {1 lettura, N2}, 2 semi. | Interazione grafo × attention ripetuta. | ~4 h | DA FARE |
| N11 | Attention fra popolazioni | Self-attention fra 77 gruppi con maschera ROI-ROI. | Neuromodulazione sulle vie esistenti. | 1 giorno + 20 min | DA FARE |

Lunghi

| N. | Test | Tecnica | Concetto | Tempo | Risultato |
|---|---|---|---|---|---|
| N6e | Shock di profondità a 8000 con passo sinaptico giusto | Fasi da 2000 (12 → 8 → 4 → 12) con il lr sinaptico di M0. | La versione lunga dell'idea dell'utente. | ~2 h 10 | DA FARE (dopo N6d e M0) |
| N9b | Fattoriale a 8000 | Ensemble ≥ 5, 3 semi, + Transformer a pari FLOP. | Il controllo che decide. | ~12 h | DA FARE |
| N10 | Layer anatomici = zone | Attention per zona, connettoma come cablaggio fra layer. | Solo se N1/N2 aiutano. | 1-2 giorni | DA FARE |
| N13 | Promozione a 8000 della variante N vincente | Regola ≥ 0,10. | Regge? | 56-90 min | DA FARE |

## 3. Ordine proposto

1. Finire la coda 7g/7h (K0 fatta: 3,652; K6 in corso; poi K7, K8).
1b. N0 diagnostica (10 min CPU): dice se le letture ripetute hanno qualcosa di nuovo da leggere; decide se N1/N2 valgono la pena.
2. Corti a zero GPU (categoria C + A5 + raggio spettrale + raggiungibilità pesata): un pomeriggio di CPU, nessuna decisione.
3. Corti GPU che chiudono domande aperte: L2 (pari tempo), L4, L5, null "solo porte", pari capacità olfatto/udito, 2.5.3 identità, idea B reservoir: circa 3 h.
3b. Corti looped: N1 (2 letture), N2 (8 letture), N4/N5 ablazioni, N7 zone, M4 porte a conduttanza (~2 h 30 in tutto); poi N3, N6, N12; M1 e M2 dopo mezza giornata di codice.
4. L1 promozione di K5 a 8000 (2 h 50) e L3 replica: decidono il nuovo standard.
5. Medi: tooling inferenza per varianti, 2.5.2 attention OFF, 2.7.2 guadagno per ROI, sinapsi temporali, rumore in ingresso.
6. Lunghi da decidere insieme: 2.4.1, 2.5.10, 2.6.2, 2.6.6, K6 LTC, F3 lungo.

## 4. Decisioni dell'utente pendenti

1. Nuovo standard = K5 (dopo L1 a 8000)?
2. Sostituire K6 con la coda 7h (K7 + K8)?
3. Kernel fuso sul modello principale (matematica identica, ×1,73).
4. F3 run lungo.
5. 2.4.1 pesi per coppia tipo-tipo (cambio del core).
