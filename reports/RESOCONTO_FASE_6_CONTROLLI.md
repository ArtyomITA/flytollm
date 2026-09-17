# Resoconto fase 6 — controlli (14 settembre 2026)

Protocollo fissato prima dei risultati: `PROTOCOLLO_FASE_6_CONTROLLI.md`. Coda: `phase6_controls_queue.py`; stato vivo `results/phase6_controls_live.json`; tabella `results/phase6_controls_summary.json`; log per job `results/phase6_<job>.console.log` e `.jsonl`.

Stato alle 21:05: sei job su sei completati (`ok`, allocator finale 0, nessun abort del supervisore, picco VRAM 458 MiB per la mosca). Il sesto (`phase6_rewired_h10_8000`, ripresa del rimescolato da 2000 a 8000) è caduto al primo tentativo sul controllo di equivalenza eager/CUDA-graph (vedi §5) ed è passato al secondo tentativo con lo stesso comando, lanciato da `phase6_rewired_8000_runner.py`; ripresa esatta (modello, ottimizzatore e stato ricorrente uguali al checkpoint), equivalenza eager/graph alla ripresa: loss 0, gradiente 3,1e-8, pesi 4,8e-7.

## 1. Tabella CE DEV16 a pari target

DEV16: 16 storie riservate, prefisso 128, 2.048 target; CE in nat; acc = accuratezza top-1 sul token successivo. Target cumulativi identici per tutti i modelli: 52.673 a 2000 update, 208.271 a 8000 (stesso generatore di coppie, stesso ordine di storie, seed 17).

| Modello | Parametri | 2000 update: CE / acc | 8000 update: CE / acc | s/update |
|---|---|---|---|---|
| Mosca reale H1-10, seed 17 (`pretrain_pilot_h10`, `pretrain_night8000_h10`) | 5.459.469 | 5,087 / 0,199 | 3,959 / 0,300 | 0,667 / 0,626 |
| Mosca reale H1-10, seed 23 (replica, `phase35_4_final_review`) | 5.459.469 | 5,239 / — | — | — |
| Mosca rimescolata a gradi conservati, seed cablaggio 41 (`phase6_rewired_h10_*`) | 5.459.469 | 4,931 / 0,220 | 3,814 / 0,315 | 0,683 / 0,686 |
| GRU 2×256, LR 1e-4 | 1.903.872 | 5,385 / 0,110 | 4,175 / 0,272 | 0,016 |
| GRU 2×256, LR 1e-3 | 1.903.872 | 4,213 / 0,278 | 3,429 / 0,354 | 0,019 |
| Transformer d256 2 blocchi, LR 1e-4 | 2.626.560 | 4,410 / 0,274 | 3,563 / 0,347 | 0,032 |
| Transformer d256 2 blocchi, LR 1e-3 | 2.626.560 | 4,755 / 0,201 | 4,323 / 0,223 | 0,035 |
| Unigramma (conteggi sui 52.673 target di training) | — | 5,843 | 5,843 | — |
| Bigramma (idem) | — | 4,283 | 4,283 | — |
| Mosca reale a 26.247 update (riferimento, non a pari dati) | 5.459.469 | — | 3,634 | — |

Verifiche di parità eseguite: hash dell'init artificiale del rimescolato uguale al pilota (`76746edb…`, quindi stessi pesi di partenza per embedding, porte, attention, readout: cambia solo il cablaggio); CE DEV prima del training: mosca reale 8,376, rimescolata 8,394, GRU 8,322, Transformer 8,359; equivalenza eager/graph del rimescolato a 2000: loss 0, gradiente 1,5e-8, pesi 1,4e-6; picco VRAM: mosca 458 MiB, GRU 75 MiB, Transformer 78 MiB; rimescolamento: 2.753.975 archi, gradi in/out identici, 0 self-loop, 0 duplicati, archi originali sopravvissuti 4,6e-5, reciprocità 0,117 → 0,0019.

## 2. Curve DEV ogni 500 update

CE DEV16 (accuratezza tra parentesi). La mosca reale ha valutazioni solo a 2000/4000/6000/8000.

| Update | Target | Rimescolata | GRU 1e-4 | GRU 1e-3 | Transf. 1e-4 | Transf. 1e-3 | Mosca reale |
|---|---|---|---|---|---|---|---|
| 500 | 13.341 | 5,888 (0,073) | 5,966 (0,073) | 5,169 (0,173) | 5,391 (0,171) | 5,362 (0,124) | — |
| 1000 | 25.758 | 5,744 (0,080) | 5,940 (0,073) | 4,730 (0,213) | 4,999 (0,214) | 5,175 (0,139) | — |
| 1500 | 39.883 | 5,235 (0,171) | 5,650 (0,085) | 4,331 (0,272) | 4,574 (0,266) | 4,756 (0,204) | — |
| 2000 | 52.673 | 4,931 (0,220) | 5,385 (0,110) | 4,213 (0,278) | 4,410 (0,274) | 4,755 (0,201) | 5,087 (0,199) |
| 3000 | 78.584 | 4,556 (0,258) | 4,937 (0,174) | 3,914 (0,300) | 4,152 (0,289) | 4,663 (0,207) | — |
| 4000 | 104.702 | 4,261 (0,279) | 4,727 (0,203) | 3,770 (0,319) | 3,986 (0,298) | 4,593 (0,213) | 4,463 (0,264) |
| 5000 | 129.578 | 4,160 (0,288) | 4,650 (0,211) | 3,757 (0,320) | 3,895 (0,318) | 4,557 (0,217) | — |
| 6000 | 155.559 | 4,006 (0,298) | 4,453 (0,258) | 3,614 (0,334) | 3,772 (0,337) | 4,528 (0,213) | 4,171 (0,288) |
| 7000 | 182.508 | 3,915 (0,304) | 4,288 (0,254) | 3,503 (0,345) | 3,646 (0,336) | 4,379 (0,236) | — |
| 8000 | 208.271 | 3,814 (0,315) | 4,175 (0,272) | 3,429 (0,354) | 3,563 (0,347) | 4,323 (0,223) | 3,959 (0,300) |

Valori intermedi del rimescolato a 2500/3500/4500/5500/6500/7500: 4,738 / 4,460 / 4,223 / 4,106 / 3,947 / 3,864. Doppia valutazione finale del rimescolato: 3,8147 e 3,8143 (mosca reale: 3,9592 e 3,9579).

## 2b. Baseline rifatte sullo stesso stream (aggiunta 14 settembre, 21:30)

`phase6_baselines_plus.py` (solo CPU, `results/phase6_baselines_plus.json`) ricostruisce esattamente lo stream di training visto da tutti i modelli (stesse coppie, stesso ordine) e rifà le baseline di conteggio con smoothing dichiarato, sui target cumulativi a 2000 (52.673) e a 8000 (208.271). CE su DEV16 (2.048 target):

| Baseline | Stimata su 52.673 target | Stimata su 208.271 target |
|---|---|---|
| Unigramma add-1 | 5,843 | 5,785 |
| Bigramma interpolato del progetto (`(c+10p)/(n+10)`) | 4,283 | 3,832 |
| Bigramma Kneser-Ney interpolato (sconto 0,75) | 4,084 | 3,691 |
| Trigramma Kneser-Ney interpolato | 3,917 | 3,410 |

Le due prime righe a 52.673 coincidono con i valori riportati finora (5,843 / 4,283), a conferma della ricostruzione. La tabella a 208.271 target è quella corretta per i confronti a 8000 update: finora il bigramma congelato a 2000 aveva visto quattro volte meno dati dei modelli. A pari dati un trigramma Kneser-Ney (3,410) batte la mosca reale a 8000 (3,959), la mosca a 26.247 update (3,634), il Transformer 1e-4 (3,563) e quasi la GRU 1e-3 (3,429). Lettura: a questa scala di dati (208 mila target) i modelli di conteggio sono la baseline forte, e qualunque confronto tra architetture va letto in regime di dati scarsi.

Prequential (proxy): media della loss di training lungo lo stream, pesata per numero di target, campionata ogni 16 update dalla telemetria (125 campioni a 2000, 500 a 8000). Premia chi impara prima, non solo chi finisce meglio.

| Modello | Fino a 2000 | Fino a 8000 |
|---|---|---|
| Mosca reale H1-10 | 5,855 | 4,945 |
| Mosca rimescolata | 5,772 | 4,800 |
| GRU 1e-4 | 5,966 | 5,144 |
| GRU 1e-3 | 5,056 | 4,246 |
| Transformer 1e-4 | 5,334 | 4,490 |
| Transformer 1e-3 | 5,349 | 4,901 |

Stesso ordine della CE finale: il rimescolato impara prima della mosca reale lungo tutto lo stream, non solo al punto finale.

## 3. Lettura secondo il protocollo

**Controllo 1, il cablaggio reale conta?** A 2000 update il rimescolato fa 4,931 contro 5,087 della mosca reale con lo stesso seed (differenza 0,157 nat a favore del rimescolato) e contro 5,239 della replica seed 23 (0,309). La soglia di variabilità fissata nel protocollo era circa 0,15 nat (distanza tra i due seed della mosca reale): la differenza è al limite di quella soglia, ma nel verso opposto a quello di un vantaggio anatomico. A 8000 update il rimescolato fa 3,814 contro 3,959 della mosca reale (0,145 nat a favore del rimescolato; accuracy 0,315 contro 0,300), con lo stesso vantaggio lungo tutta la curva: 0,20 nat a 4000, 0,17 a 6000. Esito: nessuna evidenza che la topologia reale aiuti in questa pipeline; il rimescolato è costantemente migliore, di poco oltre la variabilità da seed, quindi il cablaggio anatomico agisce da vincolo sfavorevole per questo compito (terzo caso del protocollo). Un solo seed di rimescolamento e un solo seed di init per condizione: risultato da replicare prima di dichiararlo robusto; raggio spettrale delle due matrici non misurato. Le generazioni greedy del rimescolato a 2000 sono degeneri quanto quelle del reale a 2000 ("He was a big. He was a big." contro ", a a big. The bird was a big."); a 8000 il rimescolato produce ", there was a little girl named Tim. The dog was very happy. The dog was very happy.", stesso livello del reale.

**Controllo 2, la mosca vale qualcosa in assoluto?** No, in questa configurazione. A 8000 update e pari dati: GRU LR 1e-3 3,429 e Transformer LR 1e-4 3,563 contro 3,959 della mosca (0,53 e 0,40 nat meglio), con 2,9 e 2,1 volte meno parametri e 33 e 19 volte meno tempo per update. Anche la mosca dopo 26.247 update (3,634) resta sopra la GRU a 8000. A 2000 update la mosca (5,087) è sotto il bigramma (4,283); GRU 1e-3 e Transformer 1e-4 sono già sotto il bigramma a 1500. Le baseline con l'altro LR sono peggiori (GRU 1e-4 4,175; Transformer 1e-3 4,323, instabile a metà training), come da protocollo entrambe riportate senza ricerca ulteriore. Il Transformer ha il vantaggio strutturale dichiarato del gradiente su tutto il prefisso 128, la GRU no: la GRU con TBPTT 8 batte comunque la mosca.

**Generazioni greedy a 8000 (prompt "Once upon a time").** Mosca reale: ", there was a little girl named Tim. The girl was a big dog named. The girl was a big dog. The dog was very happy." GRU 1e-3: ", there was a little girl named Tim. Tim loved to play with his mom. One day, Tim saw a big box of his friends." Transformer 1e-4: ", there was a little girl named Lily. She loved to play with her mom. One day, she saw a big tree with her mom." Le baseline formano frasi con soggetto coerente; la mosca ripete schemi.

**Cosa questo non dice.** Non chiude la fase 3 e non autorizza modifiche architetturali. Non dice che il connettoma non possa contribuire con altra lettura, altre porte o altro schema di training; dice che, nella pipeline attuale, il cablaggio reale non produce vantaggio misurabile e che l'insieme "grafo + interfacce" impara più lentamente di modelli standard a pari dati. È coerente con lo scopo dichiarato dall'utente (usare la natura del connettoma per farlo parlare, non vincere una gara di efficienza).

## 4. Limiti dichiarati

Un seed per configurazione (rimescolamento compreso); DEV16 già usato durante lo sviluppo; test finale e audit64 non consultati; LR delle baseline limitato a due valori; Transformer con orizzonte di gradiente più lungo della mosca; tempi per update non confrontabili tra architetture (mosca con CUDA Graph FP32, baseline eager); confronto a pari dati, non a pari tempo né a pari parametri.

## 5. Incidente tecnico: ripresa del rimescolato

Il primo tentativo di ripresa da `phase6_rewired_h10_2000.latest.pt` è fallito dopo 19 s in `phase3_t45.FairCapture` (`torch.testing.assert_close`, `rtol=1e-3, atol=1e-5`): 2 elementi su 65.536 di un parametro 256×256 differivano di 5,2e-5 tra passo eager e passo catturato. Causa: il controllo confronta un passo Adam eseguito con momenti azzerati; per elementi con gradiente dell'ordine di 1e-8 l'aggiornamento vale circa `lr·g/(|g|+eps)`, funzione discontinua del rumore degli atomics nello scatter sparso, quindi il confronto può fallire per caso su un checkpoint allenato (tutte le riprese precedenti erano passate con errore massimo ≤1,9e-6). Il secondo tentativo con lo stesso comando ha superato il controllo (pesi 4,8e-7) ed è arrivato a 8000 update (`phase6_rewired_8000_runner.py`, che prevedeva in alternativa un run da zero a 8000; non è servito). Il file del primo fallimento è conservato come `results/phase6_rewired_h10_8000.failed_resume_1.*`. Nessun file elencato in `pretrain_resumable.SOURCES` è stato modificato: i checkpoint esistenti restano riprendibili. Nota per il futuro: la stessa fragilità vale per ogni ripresa, compreso il checkpoint a 26.247 update; un fix richiederebbe di toccare `phase3_t45.py` (che è nelle impronte dei checkpoint) e va deciso con l'utente.

## 6b. Fase 6b, punto 1: test corti pre-registrati (14 settembre, 22:45)

Pre-registrazione: `PREREGISTRAZIONE_FASE_6B.md` (soglie fissate prima dei run). Coda `phase6b_queue.py`, stato `results/phase6b_live.json`. Stessa pipeline, stesso init artificiale (hash `76746edb…`), stessi 52.673 target a 2000 update.

| Condizione a 2000 update | CE DEV16 | Accuracy | s/update |
|---|---|---|---|
| Mosca reale seed 17 / seed 23 | 5,087 / 5,239 | 0,199 / — | 0,667 |
| Rimescolata a gradi conservati, seed 41 | 4,931 | 0,220 | 0,683 |
| Rimescolata a gradi conservati, seed 43 (`phase6b_rewired_s43_2000`) | 4,896 | 0,224 | 0,667 |
| Configuration model, seed 41 (`phase6b_configmodel_s41_2000`) | 4,533 | 0,260 | 0,611 |

Curva del seed 43 ogni 500: 5,887 / 5,760 / 5,148 / 4,896. Curva del configuration model: 5,732 / 5,228 / 4,752 / 4,533. Equivalenza eager/graph per entrambi: gradiente 1,5e-8, pesi 4,8e-7.

Letture pre-registrate. Test 1.2: Δ = 5,087 − 4,896 = +0,191 ≥ +0,10: il vantaggio del rimescolato è replicato con segno concorde su due seed di cablaggio (0,157 e 0,191); resta un solo seed di init. Test 1.5: 4,533 ≤ 4,931 − 0,10: anche l'assegnazione anatomica dei gradi ai nodi è un vincolo sfavorevole; la scala è monotona nel verso "più casuale, meglio": reale 5,087 → gradi conservati 4,90-4,93 → configuration model 4,533. Differenze strutturali del configuration model che possono spiegarlo, misurate sul grafo: archi uscenti dai neuroni sensoriali 167.131 → 291.847 (il testo entra nel grafo con più fan-out), hub riassegnati a nodi casuali, frazione eccitatoria 0,633 → 0,642. Quale di queste conta non è separabile con questo run. Test 1.3, sonda lineare sullo stato (`phase6_state_probe.py`, `results/phase6b_state_probe.json`, 15 settembre 00:05). Caratteristiche: potenziale di 10.336 neuroni (8.192 casuali seed 5 uniti ai 2.241 di readout) più spike di readout, 12.577 in tutto; ridge duale con λ = righe di training (6.087 righe da 48 storie di training 0-47), valutazione su DEV16 (2.048 righe). Accuratezza top-1 nel decodificare dallo stato il token entrato k posizioni prima; controllo = stessa sonda con etichette permutate; classe maggioritaria 7,3%.

| Lag k | Reale 8000 | Rimescolata 8000 | Reale 26.247 | Controllo permutato |
|---|---|---|---|---|
| 0 | 78,4% | 81,5% | 77,7% | 3-4% |
| 1 | 63,8% | 70,6% | 63,8% | 3% |
| 2 | 45,7% | 51,4% | 46,1% | 3% |
| 4 | 23,9% | 23,5% | 26,0% | 3% |
| 8 | 10,1% | 7,7% | 10,3% | 3% |

Letture pre-registrate: il nucleo "tiene" il token a ogni distanza fino a 8 (sopra il controllo di più di 5 punti) in tutte le condizioni, tranne la rimescolata a distanza 8 (+4,4 punti, sotto soglia). Reale e rimescolata differiscono (≥5 punti) a distanza 1 (+6,8 per la rimescolata) e 2 (+5,7 per la rimescolata); uguali a 0, 4 e 8. Da 8000 a 26.247 update il contenuto decodificabile dello stato non cambia. Limite dichiarato: lo stato include il feedback dell'attention, quindi la sonda non separa nucleo e canale esterno.

## 6c. Fase 6b completa: suite corta (15 settembre) e promozioni a 8000 (notte 15-16 settembre)

Tracciamento completo, con pre-registrazioni, curve ogni 500 update e considerazioni di ogni test: `SUITE_TEST_FASE_6B.md` (categorie A canale esterno, B null sul grafo, C sonde e lesioni, D natura come parametrizzazione, E sistemi, S velocità, F regioni specializzate). Pipeline identica al pilota (seed 17, init artificiale identica), riferimento a 2000 update 5,087 (rifatto: 5,080; pavimento di rumore 0,01 nat), a 8000 update 3,959; rimescolata a gradi conservati 3,814 a 8000. I run di controllo da 6c in poi usano il kernel fuso cupy (`fly_core_fast.FusedCore`, ×1,73), verificato equivalente al core PyTorch a 100 e a 2000 update (S6, S8b); il modello principale (`fly_core.py`) è invariato.

Scala dei null a 2000 update (tutti battono il reale): reale 5,087 > rimescolata entro superclasse 5,005 > gradi conservati 4,931 / 4,896 > Erdős-Rényi 4,676 > segni permutati 4,645 ≈ porte random 4,639 > configuration model 4,533. Pesi permutati 5,103 (nessun effetto: i pesi sono allenabili). Scelte biologiche a 2000: lettura per gruppi anatomici 4,848 (+0,24), soglia sinaptica relativa 4,961 (+0,13), 6% segni capovolti 4,982 (+0,11), potenziali di inversione 5,044 (+0,04); APL, lobo ottico graduato, costanti di tempo per tipo: nessun effetto (regioni fuori percorso o parametri che Adam 1e-4 non muove). Regioni specializzate (F): occhi come porte 5,906 (unigramma), lettura fru/dsx 5,285, occhi + fru/dsx 5,127 (interazione −0,98 nat: l'unica coppia anatomica che funziona, via visiva del corteggiamento), lettura dagli hub 5,781 (hub inibiti, muti), complesso centrale 5,529, udito/tatto come porte 5,182, occhi + hub 5,904. Analisi F8: la CE a 2000 è monotona nella distanza media in sinapsi porte→lettura (random 1,14 → 4,639; anatomiche 1,62 → 5,087; udito 2,19 → 5,182; olfatto 3,39 → 5,339; occhi 3,66 → 5,906).

Promozioni a 8000 update (regola: chi batte il reale di ≥ 0,10 a 2000 viene rifatto da zero a 8000, kernel fuso, stessi 208.271 target):

| Variante (promossa da 2000) | CE 2000 | CE 4000 | CE 6000 | CE 8000 | Δ vs reale 8000 (3,959) | Verdetto a 8000 |
|---|---|---|---|---|---|---|
| Configuration model (seed 41) | 4,533 | 3,986 | 3,810 | 3,648 | +0,311 | vantaggio confermato |
| Porte random a pari capacità (17.937), grafo intero | 4,639 | 4,070 | 3,863 | 3,689 | +0,270 | vantaggio confermato |
| Segni di Dale permutati | 4,645 | 4,078 | 3,872 | 3,723 | +0,236 | vantaggio confermato |
| Erdős-Rényi | 4,676 | 4,102 | 3,854 | 3,697 | +0,262 | vantaggio confermato |
| Lettura per gruppi anatomici (57) | 4,848 | 4,279 | 4,032 | 3,944 | +0,015 | non confermato (dentro il rumore) |
| Soglia sinaptica relativa a pari archi | 4,961 | 4,385 | 4,117 | 3,903 | +0,056 | tendenza positiva, sotto soglia |
| 6% segni capovolti (segni morbidi) | 4,982 | 4,353 | 4,097 | 3,897 | +0,062 | tendenza positiva, sotto soglia |

Letture. (1) I quattro null a topologia o segni alterati convergono a 8000 in 0,08 nat (3,648-3,723), tutti 0,24-0,31 sopra il reale: il tetto di questa pipeline con 166.700 LIF è ~3,65-3,70 e non dipende dalla struttura fine del grafo. (2) Il grafo anatomico intero con solo le porte spostate (porte random, 3,689) recupera l'87% del divario reale→tetto: il freno principale del cablaggio reale è dove entra il testo (periferia sensoriale, dietro l'inibizione locale), non il cablaggio in sé. (3) La lettura per gruppi anatomici è un falso positivo della regola (+0,24 a 2000, +0,015 a 8000): vantaggio di ottimizzazione, non di capacità. (4) La soglia sinaptica relativa (3,903) e i segni morbidi (3,897) restano sopra il reale lungo tutta la curva (+0,06 a 8000, sopra il rumore 0,01, sotto la soglia 0,10): le sole scelte con grafo e porte anatomici intatti che migliorano anche a lungo, e in misura piccola. Confronto con le baseline a pari dati (GRU 3,429, Transformer 3,563, trigramma KN 3,410): il tetto dei null resta sotto di 0,22-0,29 nat.

Decisioni aperte (utente): kernel fuso sul modello principale; riordino dei nodi solo con permutazione dei buffer a gruppi/porte invariati; run lungo di occhi + fru/dsx; combinazione porte random + soglia relativa (+ segni morbidi, potenziali di inversione) come base dei test successivi; D6 con learning rate dedicato per le costanti di tempo.

## 6. Decisioni

Nessuna. I risultati sono presentati all'utente; le opzioni successive (secondo seed di rimescolamento, ottimizzazioni elencate in `RESOCONTO_STOP_NOTTE_MOE.md` §7 e `PIANO_FASI.md` fase 4, esiti della ricerca bibliografica del 14 settembre) vanno proposte e scelte insieme.
