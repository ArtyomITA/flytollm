# Piano dei test FlyToLLM — aggiornato il 18 settembre 2026

## Indice

0. [In breve: miglior valore e conclusioni](#0-in-breve-miglior-valore-e-conclusioni)
1. [Numerazione originale, stato voce per voce](#1-numerazione-originale-stato-voce-per-voce)
2. [Tutti i test per categoria, corti e lunghi, con risultato e osservazioni tecniche](#2-tutti-i-test-per-categoria-corti-e-lunghi-con-risultato)
   A canale esterno · B null e controlli · C sonde · C-bis quanto cervello impara · D parametrizzazione · E sistemi · F regioni · K/L vincitrici · M attention anatomica · N looped e profondità
3. [Micro-sweep del passo sinaptico (M0), tabella](#3-micro-sweep-del-passo-sinaptico-m0)
4. [Fila dei prossimi test](#4-fila-dei-prossimi-test)
5. [Decisioni dell'utente pendenti](#5-decisioni-dellutente-pendenti)

## 0. In breve: miglior valore e conclusioni

**Miglior valore.** A 8000 update: configurazione standard + Muon 1e-3 sulle matrici dense = **3,541** (baseline standard 3,652; configuration model 3,648; Transformer piccolo 3,563 con Adam; GRU 3,429): è il candidato sicuro per il default. A 2000 update: tutte le leve insieme (8+8 + conduttanza + tau 1e-2 + Muon 1e-3) = **4,152** (replica con altro seed 4,173), 3,5 volte più lenta; 12+12 da solo 4,162; default candidato 4,275.

**Conclusioni degne di nota.**
1. **Le sinapsi quasi non imparano**: 0,30% di variazione in 8000 update, 2,9% degli archi mossi. Il nucleo è stato finora una scatola quasi fissa; l'apprendimento sta nelle interfacce. I confronti reale contro null fatti finora confrontavano reservoir.
2. **Il problema non è il passo ma quanti neuroni sparano** (M0): con lr sinaptico fino a 300 volte più grande i pesi si muovono fino al 15,8%, tutto stabile, ma la CE a 300 update non cambia e gli archi che ricevono gradiente salgono solo da 1,2% a 6,8%. Col gradiente allargato gli archi mossi passano da 3,3% a 14,6% a pari lr. Le leve giuste sono quelle che accendono neuroni: gradiente allargato, omeostasi, arousal, shock di profondità (in coda).
3. **Muon è l'unica leva che resta sopra soglia a 8000** (+0,112, costo zero). Le leve di dinamica sono acceleratori: 8+8 da +0,16 a +0,06, conduttanza da +0,10 a +0,03.
4. **A pari tempo GPU vince il default semplice**: nel tempo in cui K5 fa 2000 update (4,152) il default ne fa 7000 (3,615).
5. **Profondità**: satura a 8+8 (12+12 +0,025); fra 8 e 12 si può cambiare profondità quasi gratis (+0,04 / +0,07), tutto ciò che tocca 4+4 costa (shock 4 → 12: +0,34 subito).
6. **Attention**: nel blocco pre rileggere non aggiunge nulla; dopo il feedback lo stato cambia e a fine token una rilettura guarderebbe altrove nel 39% dei casi; la dinamica degli spike è un ciclo di periodo 2.
7. **Ingressi anatomici**: costano ancora (−0,145) ma la metà di prima; da porte random il segnale arriva all'85% della lettura in un salto, dagli occhi in 2-3 salti con segni che si cancellano.
8. **Rumore**: 0,02 fra run e fra semi; sotto 0,04 due configurazioni non si distinguono con una run.
9. **Errata**: il raggio spettrale riportato in C11/G0i/J3 era un artefatto (iterazione che oscilla); vale 0,500 per costruzione e non cambia con l'allenamento.
10. **Bug evitati**: la testa separata del trainer avrebbe scartato in silenzio le varianti looped (corretto prima di ogni run); regola di scelta automatica nelle code = proposta dell'assistente, da approvare; trappola del recorder ("run `nome`" nelle note).

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

| N. | Test | Tecnica | Concetto | Tempo | Risultato | Osservazioni tecniche (caveman ultra, per l'assistente) |
|---|---|---|---|---|---|---|
| A5b | Loss per posizione con offset random | Finestre da 16 token a offset casuale, indice loss(pos 16) − loss(pos 1) sullo stesso checkpoint. | L'uso del contesto misurato è reale o è l'allineamento delle finestre? | 10 min | DA FARE | — |
| A2.7.2 | Guadagno allenabile per superclasse | Uno scalare per superclasse (init 1, lr 1e-2) sulla corrente esterna; flag core `gain_group`. | Quali regioni il testo alza o spegne. | 15 min | IN CODA | — |

Lunghi

| N. | Test | Tecnica | Concetto | Tempo | Risultato | Osservazioni tecniche (caveman ultra, per l'assistente) |
|---|---|---|---|---|---|---|
| 2.8.2 | Prequential per update | Loss per update sullo stesso stream per reale, rimescolato, GRU, Transformer; bit totali. Serve logging nel trainer. | Confronto onesto "quanti bit in tutto". | mezza giornata | DA FARE | — |

### B. Null e controlli sul grafo

Corti

| N. | Test | Tecnica | Concetto | Tempo | Risultato | Osservazioni tecniche (caveman ultra, per l'assistente) |
|---|---|---|---|---|---|---|
| B8 | Replica con seed delle porte 23 | Nello standard le porte sono random: altro seed. | Quanta varianza viene dalla scelta casuale delle porte. | 15 min | IN CODA | — |
| B9 | Porte random a pari capacità: 2.639 (olfatto) e 1.733 (udito) | Stesso default con meno porte random. | Separare "anatomia" da "numero di porte" nei test F. | 2 × 15 min | IN CODA | — |
| B2.5.3a | Nucleo identità | Pesi sinaptici a 1e-6 e congelati: neuroni isolati, interfacce uguali. | Il pavimento "senza grafo". | 15 min | IN CODA | — |
| B10 | Reservoir | Pesi sinaptici congelati all'init anatomica. | La topologia fissa porta informazione senza allenare gli archi? Attesa ≈ default (C13w). | 15 min | IN CODA | — |
| B2.5.2 | Fattoriale grafo × attention | Reale / rimescolato a soglia 10 × attention accesa / spenta (4 run). | Interazione: il reale dipende dall'attention più del rimescolato? | 4 × 15 min | IN CODA | — |
| 2.5.3b | MLP denso a pari parametri | MLP da 2,75 M parametri al posto del core. | Il connettoma vale più di un blocco denso qualunque? | 1-2 h codice + 10 min | DA FARE | — |
| 2.3.8 | Null "emilineaggio rimescolato" | Etichette permutate a pari dimensione; solo con 2.3.7. | Serve la partizione biologica? | 15 min | DA FARE | — |

Lunghi

| N. | Test | Tecnica | Concetto | Tempo | Risultato | Osservazioni tecniche (caveman ultra, per l'assistente) |
|---|---|---|---|---|---|---|
| B11 | Promozioni a 8000 dei controlli sopra soglia | Regola della suite, contro 3,652. | Regge con l'allenamento? | 56 min l'una | DA FARE | — |
| B12 | Null del grafo a pari architettura looped | Rimescolato dentro la variante N vincente. | Con attention nel giro, chi lavora? | 56 min | DA FARE | — |
| B13 | Scala dei null con sinapsi che imparano | Reale, rimescolato, configuration model rifatti col lr sinaptico scelto da M0. | Finora si confrontavano reservoir (C13w). | 3 × 15 min + 3 × 56 min | DA FARE (dopo M0) | — |

### C. Sonde, mappe e lesioni

Corti

| N. | Test | Tecnica | Concetto | Tempo | Risultato | Osservazioni tecniche (caveman ultra, per l'assistente) |
|---|---|---|---|---|---|---|
| C12 | Raggio spettrale all'init + errata | Iterazione di potenza su CPU, init e checkpoint. | L'allenamento cambia il guadagno ricorrente? | 5 min | FATTO: 0,500 all'init per costruzione, identico sui checkpoint; 0,405 sui neuroni attivi. ERRATA: i valori di C11/G0i/J3 (0,53 → 0,73 → 0,35) erano fasi di un'oscillazione, nulli. | power iter oscilla periodo 2 (0,7194/0,3475, prod 0,25). rho=0,500 da norm Σ|w_in|=0,5. trained = init a 4 cifre. attivi-only 0,405 (T8 0,412). null signed 0,198/0,234, abs 0,493/0,464. fix: geo-mean 2 iter + seed fisso in phase6c_inference.spectral. C11/G0i/J3 void. |
| C13w | Movimento dei pesi sinaptici | ‖w − w₀‖/‖w₀‖ sui checkpoint. | Il cervello impara o è una scatola fissa? | 5 min | FATTO: 0,11% a 1000, 0,30% a 8000; 2,9% degli archi mossi. Core ≈ reservoir. | dw/w0: 1k 0,11% · 4k 0,21% · 8k 0,30% · Muon8k 0,26 · T8-8k 0,32. moved>1e-4 2,9% (from-active 14,9%), max 0,016. causa: gw = g_dst·s_src, 81% muti, 19% archi da sorgenti attive; softplus raw≈−3,5 → step_w ≈ lr·σ(raw) ≈ lr·0,03. |
| D2b | Raggiungibilità pesata per segno | Propagazione lineare con segno da porte a lettura, all'init. | Perché certi ingressi funzionano. | 1 min | FATTO: porte random raggiungono l'85% della lettura in 1 salto; occhi in 2-3 salti con segni che si cancellano (coerenza 0,07). | reach lineare con segno, init thr10: random→chunks hop1 reach 0,85 coh 0,26; visual→chunks hop3 reach 0,37 coh 0,07; visual→fru hop2 reach 0,03; visual→hub coh 0,96 net<0; anatomical hop1 coh≈1 reach 0,41. |
| C15 | Sinapsi rimesse all'init | CE del checkpoint K8 con i pesi sinaptici riportati all'init; copertura per superclasse. | Quanto lavoro fa l'apprendimento sinaptico. | 5 min | FATTO: sinapsi rimesse all'init su K8: 3,5407 → 3,5401 (0,0006). Il nucleo è un reservoir fisso; l'apprendimento sta nelle interfacce. | K8 4+4 8k: trained 3,5407 · all-init 3,5401 · moved-init 3,5395 · unmoved-init 3,5494 (Δ 0,009: i pesi NON mossi contano più dei mossi = rumore di init). ol_intrinsic 89k neuroni 17,6% attivi, 0,24% archi mossi. bug closure fixato (record.add_). |
| C6b | Reclutamento per tipo cellulare | Tassi per 11.752 tipi per classe di token. | Mappa funzionale a risoluzione di tipo. | 10 min | DA FARE | — |
| C10b | Sonda posizionale rifatta | Ridge con cross-fit su EB/PB/FB contro gruppi random. | La mosca ha un codice di posizione? | 10 min | DA FARE | — |
| C13 | Inferenza sui checkpoint con varianti | Estendere `phase6c_inference.py` alle varianti; suite su K5. | Cosa fa dentro il modello migliore. | 30 min codice + 27 min | FATTO: caricatore universale delle varianti (phase8_load_variant.py): K5 ricostruito, CE DEV 4,1529 = 4,152. Suite di inferenza su ogni variante. | VariantSuite = Suite con base.load_payload patchato; build_control_cns fast_mode off + control da worker.json; src32/dst32 droppati; strict load. 0,9 min. |
| D3b | Lettura sul potenziale (hub) | Readout da v invece che dal tasso. | Gli hub muti portano informazione nel potenziale? | 15 min | DA FARE | — |

Lunghi: nessuno.

### C-bis. Quanto cervello impara (scoperta del 17/9; richiesta utente: non lavorare su un campione di pochi neuroni)

Corti

| N. | Test | Tecnica | Concetto | Tempo | Risultato | Osservazioni tecniche (caveman ultra, per l'assistente) |
|---|---|---|---|---|---|---|
| M0 | Micro-sweep del passo sinaptico | 9 run da 300 update: default, lr dedicato dei pesi 1e-3 / 3e-3 / 1e-2 / 3e-2, con gradiente binario e allargato; si guarda il movimento dei pesi, non la CE finale. | Trovare il passo con run da 5 minuti prima di spendere run da 2000. | 9 × 5 min | IN CODA | — |
| C14 | lr dedicato per le sinapsi | Gruppo Adam separato per `core.raw`, 1e-3 e 1e-2. | Le sinapsi non imparano perché il passo è troppo piccolo? | 2 × 15 min | FATTO: passo sinaptico 1e-3 / 1e-2 a 4+4: 4,275 / 4,276 (= 4,275); a 8+8 1e-2: 4,202 vs L4 4,187 (rumore). Pesi mossi 1,4% / 14,6%: si muovono, la CE non cambia. | 4+4: 1e-3 4,275 · 1e-2 4,276 (dw 1,4% / 14,6%). T8 1e-2: 4,202 vs 4,187 (+0,015 rumore). movimento ∝ lr, CE piatta: loss piatta lungo direzioni sinaptiche. |
| C16 | Gradiente allargato (`soft_gw`) | Nel backward il gradiente del peso usa l'attività presinaptica morbida; da solo e con lr 1e-3 / 1e-2. | Far imparare anche le sinapsi dei neuroni che non sparano. | 3 × 15 min | IN CORSO: gradiente allargato da solo a 8+8: 4,183 vs L4 4,187 (rumore) con archi mossi 10,5% vs 2,5%; con passo 1e-3: 4,185, pesi mossi 3,2% (×18), archi 20,6%; con 1e-2 in coda. | T8 soft_gw: 4,183 vs 4,187. dw 0,38% vs 0,18% (×2,2), moved 10,5% vs 2,5% (×4), grad 6,4/6,3, spike 0,0542 uguale. 708 ms vs 680 (+4%). copertura ≠ collo di bottiglia (3ª conferma: M0, C14, C16). |
| H1o | Omeostasi di risveglio per tipo | Scarto di soglia senza gradiente per ognuno degli 11.752 tipi; sotto il 2% di attività la soglia scende piano, mai sotto il normale. A 4+4 e a 12+12. | Il cervello regola da solo chi svegliare. | 15 + 35 min | IN CODA | — |
| H2a | Arousal a impulsi e a onda | Scarto di soglia globale schedulato: impulso (1,0 → 0,7 per 100 update ogni 500) o onda sin² senza gradini. | Reclutamento periodico a costo di 4+4; l'onda evita il cambio brusco. | 2 × 15 min | IN CODA | — |

Lunghi

| N. | Test | Tecnica | Concetto | Tempo | Risultato | Osservazioni tecniche (caveman ultra, per l'assistente) |
|---|---|---|---|---|---|---|
| C17 | Conferma a 8000 del regime in cui le sinapsi imparano | lr sinaptico (e/o `soft_gw`, omeostasi) scelto dopo M0, a 8000. | Allenare davvero il cablaggio alza il tetto? | 56 min | DA FARE (dopo M0) | — |

### D. Natura del connettoma come parametrizzazione

Corti

| N. | Test | Tecnica | Concetto | Tempo | Risultato | Osservazioni tecniche (caveman ultra, per l'assistente) |
|---|---|---|---|---|---|---|
| L5 | Default + 12+12 sottopassi | 24 passaggi per token. | La profondità oltre il diametro del grafo (13) aiuta? | 35 min | FATTO: 4,162 (+0,113 sul default, sopra soglia; +0,025 su 8+8: satura). Costo ×2,5: a pari tempo perde. | 952 ms/upd, 34,6 min. gap vs L4 0,064→0,025 lungo run. = K5 (4,152) senza cond/tau. scala profondità 4,275/4,187/4,162. pari tempo: default ~5k upd = 3,79. |
| L7 | K5 con scala pesi ×0,5 | Peso iniziale dimezzato nella configurazione completa. | Leva minore, si somma? | 46 min | FATTO: 4,132 (+0,020 su K5: dentro il rumore). Non candidata. | curva −0,02/−0,04 costante vs K5. dw 0,09% moved 0,5% (K5 0,12/1,9): pesi init piccoli → Δ assoluto minore. 1 seed non basta (rumore 0,021). |
| M4 | Porte a conduttanza | Corrente esterna × (E − v)/E; flag `cond_ports`. | Chi è vicino a soglia riceve di più. | 15 min | IN CODA | — |

Medi (ore di codice, run corto)

| N. | Test | Tecnica | Concetto | Tempo | Risultato | Osservazioni tecniche (caveman ultra, per l'assistente) |
|---|---|---|---|---|---|---|
| 2.3.7 | Parametri per emilineaggio | tau, bias e guadagno per 35-200 gruppi (etichette solo VNC). | Granularità sopra il tipo. | 3-4 h + 15 min | DA FARE | — |
| D10 | Sinapsi con dinamica temporale | α-synapse 1-2 sottopassi + ritardo 1. | Sommazione temporale. | 3-4 h + 20 min | DA FARE | — |
| D11 | Rumore in ingresso calibrato | σ 0,05 / 0,1 / 0,2 da un banco di rumore su GPU. | Regolarizzazione biologica. | 2 h + 3 × 15 min | DA FARE | — |
| D12 | Weight tying seriale | 5.830 neuroni, 968 serial set. | Sfrutta la ripetizione segmentale? | 3 h + 15 min | DA FARE | — |

Lunghi

| N. | Test | Tecnica | Concetto | Tempo | Risultato | Osservazioni tecniche (caveman ultra, per l'assistente) |
|---|---|---|---|---|---|---|
| 2.4.1 | Pesi per coppia tipo-tipo | ~10⁵ fattori al posto di 2,75 M pesi per arco. | Quanto si perde a 10⁵ parametri. | 1 giorno + 56 min | DA FARE | — |
| 2.5.10 | Ritardi sinaptici anatomici | Da morfologia (`syn-partners` 6,8 GB). | Il tempo di propagazione conta? | 1-2 giorni | DA FARE | — |
| K6r | tau LTC per emilineaggio | Costanti dipendenti dall'input. | Dinamica adattiva. | 1-2 giorni | DA FARE | — |
| 2.6.2 | Compito affine alla mosca | Task sensorimotorio da disegnare. | Impara meglio ciò per cui è cablata? | giorni | DA FARE | — |
| 2.6.6 | e-prop pilota | Tracce di eleggibilità in `Propagate`. | Apprendimento locale. | giorni | DA FARE | — |

### E. Sistemi

| N. | Test | Tecnica | Concetto | Tempo | Risultato | Osservazioni tecniche (caveman ultra, per l'assistente) |
|---|---|---|---|---|---|---|
| S9 (lungo) | Backward sparso | Salta gli archi con presinaptico muto. | Velocità ×1,5 potenziale. | 1 giorno | DA FARE | — |
| E3 (decisione) | Kernel fuso sul modello principale | `FusedCore` in `fly_core`, ×1,73 identico. | Velocità gratis. | 0 | decisione utente | — |

### F. Regioni specializzate

| N. | Test | Tecnica | Concetto | Tempo | Risultato | Osservazioni tecniche (caveman ultra, per l'assistente) |
|---|---|---|---|---|---|---|
| F9 (corto) | Porte anatomiche corte + 8+8 + Muon 1e-3 | Udito/tatto + proiezione visiva come ingressi, nella configurazione migliore. | Gli ingressi veri smettono di costare? | 28 min | FATTO: 4,332 (−0,145 su L4 con porte random; era −0,279 con Adam). Costano ancora, la metà. | shortpath 14.957 porte (5.756 mecc + 9.201 vis proj). gap vs L4 0,29→0,15 lungo run (può chiudersi a 8k). 641 ms. porte meno → B9 separa capacità. |
| F3L (lungo) | Occhi + fru/dsx a 8000 | F3 col kernel fuso. | L'unica coppia anatomica regge? | 56 min | DA FARE | — |

### K / L. Follow-up delle vincitrici

Corti

| N. | Test | Tecnica | Concetto | Tempo | Risultato | Osservazioni tecniche (caveman ultra, per l'assistente) |
|---|---|---|---|---|---|---|
| K7 | Muon 3e-3 | lr di Muon ancora più alto. | lr ottimo più su? | 15 min | FATTO: 4,250 (+0,026 su 1e-3: plateau). | Muon lr: 1e-4 4,648 · 3e-4 4,411 · 1e-3 4,276 · 3e-3 4,250. gap 3e-3−1e-3: 0,225→0,078→0,048→0,026. grad 5,07. 3e-3 non adottato; parziale 8k a 3e-3 interrotta a 2000 (5,190/4,786/4,387). |
| L4 | Default + 8+8 | Profondità sopra Muon 1e-3. | Quanto resta alla profondità. | 27 min | FATTO: 4,187 (+0,088, sotto soglia; con Adam era +0,181). | gain 8+8 vs ottimizzatore: Adam +0,181 · Muon3e-4 +0,139 · Muon1e-3 +0,088. 680 ms (×1,8). pari tempo: default ~3,6k upd = 3,95. |
| L3 | Replica K5 seed 23 | Stessa configurazione, altro seed dei dati. | 4,152 è solido? | 46 min | FATTO: 4,173 (differenza 0,021 = rumore). | seed 23 vs 17: Δ 0,021, curve sovrapposte (500 5,141/5,158). dw 0,120/0,125%, leak 0,40-0,996 entrambi. soglia pratica famiglia K5: <0,04 indistinguibile. |
| L2 | Pari tempo GPU | Dai dati di K8. | Le leve pesanti pagano al secondo? | 0 | FATTO: nel tempo di K5 a 2000 (4,152) il default fa 7000 update = 3,615. | K8@7000 = 3,615 vs K5@2000 = 4,152, stesso wall (46 min). confonde dati visti (7k vs 2k upd). |

Lunghi

| N. | Test | Tecnica | Concetto | Tempo | Risultato | Osservazioni tecniche (caveman ultra, per l'assistente) |
|---|---|---|---|---|---|---|
| K0 | Baseline standard a 8000 | Riferimento per tutte le promozioni. | — | 56 min | FATTO: 3,652 (= configuration model 3,648). | = config model 3,648; replica G0@4000 4,000 vs 3,999; 365 ms. riferimento di tutte le promozioni. |
| K6 | Muon 3e-4 a 8000 | Promozione. | Muon accelera o alza il tetto? | 56 min | FATTO: 3,571 (+0,081). | gap vs base: 2000 +0,138 · 4000 +0,096 · 6000 +0,070 · 8000 +0,081. 379 ms. |
| K8 | Muon 1e-3 a 8000 | Promozione. | Idem, col lr giusto. | 56 min | FATTO: 3,541 (+0,112: unica leva sopra soglia a 8000; sotto il Transformer piccolo 3,563 allenato con Adam). | gap vs base: 2000 +0,288 · 4000 +0,155 · 6000 +0,114 · 8000 +0,112 (si stabilizza). vs 3e-4: +0,149→+0,031. grad med 5,9 max 25,8. replica K1 4,274. < Transformer 3,563 (Adam, non pari ottimizzatore). |
| L1 | K5 a 8000 | Configurazione completa da zero. | Le leve di dinamica aggiungono sopra Muon a 8000? | ~3 h (stima) | DA FARE | — |

### M. Attention anatomica (corpo fungiforme come memoria a pesi veloci)

| N. | Test | Tecnica | Concetto | Tempo | Risultato | Osservazioni tecniche (caveman ultra, per l'assistente) |
|---|---|---|---|---|---|---|
| M1 (corto) | Memoria MB + MHA | Pesi veloci KC→MBON con regola delta, gate DAN. | Il modello usa una memoria anatomica? | mezza giornata + 20 min | DA FARE | — |
| M2 (corto) | Memoria MB al posto della MHA | Idem, attention spenta. | Il contesto può stare nell'anatomia? | 20 min | DA FARE | — |
| M3 (lungo) | Variante fedele | Solo archi KC→MBON reali, gate per 15 compartimenti. | Le sinapsi vere bastano come memoria? | 1-2 giorni | DA FARE | — |

### N. Looped, giri e profondità

Corti

| N. | Test | Tecnica | Concetto | Tempo | Risultato | Osservazioni tecniche (caveman ultra, per l'assistente) |
|---|---|---|---|---|---|---|
| N0 | Diagnostica fra le letture | Stato di lettura e attention ipotetica dopo ogni sottopasso, sul checkpoint K8. | Se lo stato non cambia, rileggere non serve. | 10 min | FATTO: nel blocco pre rileggere dà lo stesso richiamo (coseno ≥ 0,965); dopo il feedback lo stato si allontana: al sottopasso 8 una rilettura guarderebbe altrove nel 39% dei token. Dinamica = ciclo di periodo 2. | cos rep vs real s1-8: .869 .934 .977 1 .978 .929 .854 .760. recall cos: .965 .980 .992 1 .993 .967 .902 .799. argmax agree: .851 .886 .936 1 .925 .849 .740 .614. JS max .096. entropia 1,10→1,25. spike Jaccard lag1 .308 lag2 .656 lag3 .354 lag4 .537. CE nel JSON (7,36) invalida: logit con embedding tied invece di model.head; fix fatto, rerun in 8c. |
| N6c | Trasferimento di profondità | Checkpoint allenati a 4+4 / 8+8 / 12+12 valutati alle altre profondità. | Quanto salta la CE a ogni cambio di profondità. | 3 × 3 min | FATTO: fra 8 e 12 il cambio costa +0,04 / +0,07; tutto ciò che tocca 4+4 costa 0,12-0,50; shock 4 → 12 = +0,34. | matrice train×eval (4/8/12): 4: 4,276 4,396 4,620 · 8: 4,442 4,185 4,223 · 12: 4,666 4,235 4,162. diagonale = eval di fine run. no test-time scaling (8→12 peggiora 0,038). |
| N6d | Shock di profondità (idea utente) | 12 → 8 → 4 → shock a 12, fasi da 500 update, un solo grafo con sottopassi congelati. | Lo shock accende neuroni e sinapsi nuove? | 35 min | IN CODA | — |
| N1 | Due letture per token | Attention dopo i sottopassi 2 e 6, cache per lettura, identificativo di passo. | Giro ottimo in pretraining = 2 (letteratura). | 15-20 min | IN CODA | — |
| N2 | Attention a ogni sottopasso | 7 letture, cache per lettura, injection per concatenazione. | Attention come parte della dinamica (Huginn/Ouro alla lettera). | 20-25 min | IN CODA | — |
| N3 | Letture a ogni sottopasso, cache della prima lettura | Recursive KV sharing (MoR). | Servono stati distinti per sottopasso? | 20-25 min | IN CODA | — |
| N4 | Due letture senza identificativo di passo | Ablazione. | −0,107 in letteratura: anche qui? | 15-20 min | IN CODA | — |
| N5 | Token solo al primo sottopasso | Ablazione dell'input injection. | L'informazione viaggia nello stato o nell'iniezione? | 15 min | IN CODA | — |
| N6b | Profondità ciclica per token | 12-8-4 per token, valutazione a 4 / 8 / 12. | Un modello a tre velocità. | 27 min + 3 inferenze | DA FARE (proposto) | — |
| N7 | Sottopassi sequenziali per zona | Aggiornamento sensoriali → centrali → MB/CX → discendenti. | Più zone per sottopasso a costo 4+4. | mezza giornata + 15 min | DA FARE | — |
| N12 | Prelude e coda con pesi propri | Prima e ultima lettura separate. | Le letture di bordo lavorano diverso. | 2 h + 25 min | DA FARE | — |

Medi

| N. | Test | Tecnica | Concetto | Tempo | Risultato | Osservazioni tecniche (caveman ultra, per l'assistente) |
|---|---|---|---|---|---|---|
| N8 | Stato di lettura più ricco | Discendenti + KC + MBON + CX → 512, una lettura. | Collo di bottiglia: letture o capacità dello stato? | 2 h + 20 min | DA FARE | — |
| N9a | Fattoriale grafo × letture a 2000 | {reale, null gradi+pesi ×3} × {1 lettura, N2}, 2 semi. | Interazione grafo × attention ripetuta. | ~4 h | DA FARE | — |
| N11 | Attention fra popolazioni | Self-attention fra 77 gruppi con maschera ROI-ROI. | Neuromodulazione sulle vie esistenti. | 1 giorno + 20 min | DA FARE | — |

Lunghi

| N. | Test | Tecnica | Concetto | Tempo | Risultato | Osservazioni tecniche (caveman ultra, per l'assistente) |
|---|---|---|---|---|---|---|
| N6e | Shock di profondità a 8000 con passo sinaptico giusto | Fasi da 2000 (12 → 8 → 4 → 12) con il lr sinaptico di M0. | La versione lunga dell'idea dell'utente. | ~2 h 10 | DA FARE (dopo N6d e M0) | — |
| N9b | Fattoriale a 8000 | Ensemble ≥ 5, 3 semi, + Transformer a pari FLOP. | Il controllo che decide. | ~12 h | DA FARE | — |
| N10 | Layer anatomici = zone | Attention per zona, connettoma come cablaggio fra layer. | Solo se N1/N2 aiutano. | 1-2 giorni | DA FARE | — |
| N13 | Promozione a 8000 della variante N vincente | Regola ≥ 0,10. | Regge? | 56-90 min | DA FARE | — |


## 3. Micro-sweep del passo sinaptico (M0)

Run da 300 update sul default candidato (9 di 9 finite). "Pesi mossi" = ‖w − w₀‖/‖w₀‖; "archi mossi" = frazione con |Δw| > 1e-4.

| lr sinapsi | CE a 100 / 200 / 300 | pesi mossi | archi mossi > 1e-4 | Δ massimo | norma grad | spike |
|---|---|---|---|---|---|---|
| default (1e-4) | 6,575 / 5,984 / 5,857 | 0,08% | 1,2% | 0,005 | 3,84 | 0,050 |
| 1e-3 | 6,554 / 5,980 / 5,849 | 0,75% | 3,3% | 0,039 | 3,88 | 0,050 |
| 3e-3 | 6,524 / 5,972 / 5,832 | 1,98% | 4,1% | 0,101 | 3,94 | 0,050 |
| 1e-2 | 6,488 / 5,974 / 5,839 | 4,98% | 5,3% | 0,262 | 3,96 | 0,051 |
| 3e-2 | 6,502 / 5,978 / 5,843 | 15,80% | 6,8% | 1,127 | 3,97 | 0,050 |
| 1e-3 + gradiente allargato | 6,553 / 5,980 / 5,847 | 1,74% | 14,6% | 0,059 | 3,88 | 0,050 |
| 3e-3 + gradiente allargato | 6,523 / 5,975 / 5,840 | 4,57% | 17,7% | 0,172 | 3,94 | 0,051 |
| 1e-2 + gradiente allargato | 6,489 / 5,974 / 5,849 | 10,49% | 20,2% | 0,391 | 3,95 | 0,051 |
| 3e-2 + gradiente allargato | 6,503 / 5,995 / 5,895 | 27,91% | 23,2% | 1,177 | 3,92 | 0,049 |

## 4. Fila dei prossimi test

In esecuzione, coda 8b (riprendibile, log `results/phase7_queue.console.log`, stato `results/phase8b_live.json`), nell'ordine:
1. M0 micro-sweep del passo sinaptico (9 run da 3,5 min).
2. C14 lr sinaptico 1e-3 e 1e-2 a 2000; C16 gradiente allargato da solo e con 1e-3 / 1e-2 (5 × 15 min).
3. N6d shock di profondità 12 → 8 → 4 → 12 (35 min).
4. H1o omeostasi per tipo (15 min); H2a arousal a impulsi e a onda (2 × 15 min); omeostasi a 12+12 (35 min).
5. N1, N2, N5 attention nel giro; M4 porte a conduttanza; A2.7.2 guadagno per superclasse.
6. B2.5.3a identità, B10 reservoir, N4, N3, B8 seed porte, B9 pari capacità (2), B2.5.2 fattoriale (4).

Coda 8c (da lanciare dopo la 8b): C15 rifatta, N0 con la CE corretta, e i seguiti decisi sui risultati: conferma a 2000 e a 8000 del regime in cui più sinapsi imparano (C17), shock + passo sinaptico (N6e), scala dei null rifatta con sinapsi che imparano (B13), L1 (K5 a 8000).

Da implementare: C6b, C10b, C13, A5b, D3b, N6b, N7, N12, 2.5.3b, M1/M2, medi e lunghi delle categorie D, M, N.

