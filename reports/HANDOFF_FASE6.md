# Handoff fase 6/7 — 17 settembre 2026, 13:30 (catena notturna 7c-7f conclusa; coda 7g proposta, non lanciata)

Leggere questo prima di tutto dopo un compact. Stato autorevole del progetto: `PIANO_FASI.md`, `MEMORIA_PROGETTO.md`; controlli fase 6: `PROTOCOLLO_FASE_6_CONTROLLI.md`, `RESOCONTO_FASE_6_CONTROLLI.md`; suite test corti con pre-registrazioni, risultati e considerazioni di ogni test: `SUITE_TEST_FASE_6B.md` (file di tracciamento richiesto dall'utente: "per ogni test devi tracciare tutto"); ricerca bibliografica: `RICERCA_CONNETTOMA_LM.md` (§0 = correzioni dai fatti del progetto).

## Stato al 17 settembre 2026, 13:30 (catena notturna 7c → 7d → 7e → 7f FINITA alle 11:36, niente in esecuzione)

Tutte le run delle sezioni G, H, I, J sono eseguite e registrate con considerazioni in `SUITE_TEST_FASE_6B.md`; nessuna voce "in coda" resta. GPU libera; `results/phase7f_live.json` status completed; monitor `tail` orfani chiusi.

Risultati chiave a 2000 update sulla baseline standard (4,580; soglia 0,10, rumore 0,01):
- Sopra soglia: 8+8 sottopassi 4,399 (+0,181); Muon 3e-4 sulle matrici dense 4,411 (+0,169, costo zero); conduttanza 4,459 (+0,121); tau per tipo lr 3e-3 4,462 (+0,118; leak imparati 0,80-0,98).
- Combinazioni: 8+8 + conduttanza 4,350 (miglior valore, +0,230; additività 76%); conduttanza + tau 1e-3 4,419 (+0,161, additività 88%). Regola di promozione di G5 su G1 (≥ 0,10) non superata (+0,049).
- Sotto soglia / scartate: scala pesi ×0,25 4,503 ≈ ×0,5 4,508 (plateau, +0,08); bias per tipo riparametrizzato 4,547 (+0,033, bias imparati ≤ 0,04: leva scartata; il −0,457 di H1 era fame di gradiente dal clip); Muon 1e-4 4,648 (−0,068, lr troppo basso); Muon anche sulla testa 4,909 (−0,329: testa resta su Adam); 8+8 + Muon 1e-4 4,420 (da rifare con 3e-4).
- Promozioni a 8000: 8+8 3,592 (prima run fedele al grafo sotto TUTTI i null: config 3,648, porte random 3,689); conduttanza 3,624. ATTENZIONE: manca la baseline standard a 8000 (solo 4000: 3,999); a 4000 il vantaggio di 8+8 è +0,068 e quello della conduttanza +0,033 (si restringono con l'allenamento). Prima cosa da fare: `phase7_baseline_8000` (~55 min).
- Inferenza sul checkpoint 8+8 (J3): mai attivi 77,8% contro 81,1%, KC attive 14,6% contro 11,7%, senza attention 4,543 contro 4,770, lesione rich club 0,90 contro 1,24: più lavoro nel core e più distribuito; lobi ottici ancora a effetto zero.

Stato 17/9 20:40: l'utente usa il PC da circa le 22:12: un waiter in background crea `STOP_PHASE7` alle 22:10, la coda in corso (8a o 8b) si ferma al confine del job successivo (il job in corso finisce). NON rilanciare nulla finché l'utente non dice di ripartire. Per ripartire: cancellare `STOP_PHASE7`, poi `./.venv/Scripts/python.exe phase8b_queue.py >> results/phase7_queue.console.log 2>&1` in background (la coda 8b è riprendibile: salta le run con risultato ok e gli smoke già passati; lo stato vecchio viene rinominato). Coda 8b aggiornata (35 run): inferenza C15 / N0 / N6c, micro-sweep M0 del passo sinaptico (9 run da 300 update, la tabella va portata all'utente che sceglie il lr), C14, C16, N6d shock di profondità, H1o omeostasi per tipo, H2a arousal a impulsi e a onda, omeostasi a 12+12, poi attention nel giro e controlli. Codice nuovo: flag core `soft_gw`, `homeo`, `arousal`; `--core-lr`, `--depth-schedule`, `--homeo`, `--arousal`; `phase8_coverage.py`, `phase8_c15_revert_weights.py`, `phase8_depth_transfer.py`, `phase8_cpu_analyses.py`. Scoperte del 17/9 sera: pesi sinaptici mossi dello 0,3% in 8000 update (C13w), raggio spettrale di C11/G0i/J3 = artefatto (C12).

Stato 17/9 18:50 (via utente 18:30: "fai per ogni categoria i corti, segnati il nuovo candidato sicuro per default e parti con quello; dove servono modifiche al core falle comunque"): candidato sicuro per il default = standard + Muon 1e-3 (scritto in `STANDARD_MOSCA_6B.md`; 4,275 a 2000, 3,541 a 8000). IN ESECUZIONE: coda 8a `phase8a_queue.py` (sezione L della suite: L4 8+8, F9 porte corte + 8+8, L5 12+12, L3 replica K5 seed 23, L7 K5 scala ×0,5; avvio 18:37, circa 3 h 10) e, incatenata da un waiter in background, coda 8b `phase8b_queue.py` (sezione M: N0 diagnostica, N1/N2/N3/N4/N5 attention nel giro, M4 porte a conduttanza, 2.7.2 guadagno per superclasse, identità, reservoir, seed porte, pari capacità 2.639/1.733, fattoriale grafo × attention; 16 run da ~15 min + smoke, circa 5 h). Log unico `results/phase7_queue.console.log`, stati `results/phase8a_live.json`, `results/phase8b_live.json`; stop: creare `STOP_PHASE7`; pausa: `pause_runs.py`. Codice nuovo: `fly_lm_variants.py` (LoopedFlyLM), flag `cond_ports`/`gain_group`, opzioni `--attn-* --token-injection --freeze-core --ports-seed --ports-count`, `phase8_n0_diagnostic.py`, `phase8_selftest.py` (CPU, superato). Dopo ogni run: `phase6_record.py`, considerazioni sulla riga, stampa all'utente; i test con risultato diventano candidati per il default. Corti ancora da implementare: C6b, C10b, C12, D2b, C13 (inferenza sulle varianti), A5b, D3b, N6, N7, N12, 2.5.3b, M1/M2.

Stato 17/9 18:15: code 7g, 7h (interrotta su richiesta utente: K8 era partita a 3e-3 per una regola "CE più bassa" della coda in contrasto con la pre-registrazione) e 7i COMPLETE, niente in esecuzione. K7 Muon 3e-3 a 2000 = 4,250 (plateau con 1e-3 4,276); K6 Muon 3e-4 a 8000 = 3,571; K8 Muon 1e-3 a 8000 = 3,541 (+0,112 sulla baseline standard 8000 3,652: prima leva sopra soglia a 8000). Regola di scelta automatica nelle code (margine ≥ 0,10 in `phase7g_queue.best`) = proposta mia NON approvata: l'utente deve scegliere fra margine 0,10 e "sopra il rumore 0,02"; scrivere sempre la regola nella proposta della coda prima del lancio. Prossimi per il piano: N0 diagnostica (CPU), corti looped N1/N2/N4/N5/N7, L2/L4/L3, L1 (K5 a 8000).

Ricerca looped transformer (17/9, 2 agenti Opus): `RICERCA_LOOPED_A.md` (meccanica, 17 schede 2018-2026) e `RICERCA_LOOPED_B.md` (design: KV cache per iterazione, injection, tre varianti A/B/C, controllo fattoriale 2×2); sezione N del piano. K0 baseline standard 8000 = 3,652 (= configuration model): 8+8 a 8000 +0,061, conduttanza +0,029.

Lista completa di cosa resta (numerazione originale 1.x-2.8.x voce per voce, corti/lunghi per categoria, proposte utente M/N su attention anatomica e giri): `PIANO_TEST_RIMASTI.md` (17/9 17:30).

Coda 7g proposta (script `phase7g_queue.py`, pronto, NON lanciato: decisione dell'utente): Muon 1e-3; tau lr 1e-2; 8+8 + Muon 3e-4; conduttanza + tau 3e-3; tutte le vincenti insieme (8+8 + conduttanza + tau + Muon con i lr migliori scelti a regola); baseline standard 8000; promozione Muon 3e-4 a 8000. Circa 3 h 50 di GPU totali, tutto pausabile (`pause_runs.py`), riposo 3 min ogni ~1,5 h.

Regole della notte ancora valide: 3 minuti di riposo ogni ~1,5 h di attività GPU (accumulatore `results/.gpu_activity.json` letto da `pretrain_night_queue.run_job`); run sempre pausabili (`pause_runs.py status|pause|resume`); timeout generosi (14400/21600/28800) perché il timeout del supervisore corre anche in pausa.

Trappola del recorder: qualsiasi testo "run `nome`" in una sezione fa saltare quella run a `phase6_record.py` (successo tre volte: G4, bias reruns, bias_scaled). Nelle note scrivere il nome senza la parola "run" davanti.

Codice nuovo (modello principale intatto): `fly_core_variants.py` varianti combinabili con '+' e `bias_type` (BIAS_SCALE 0,02); `pretrain_control.py --type-param-lr --weight-scale --optimizer muon --muon-lr --muon-head --pre-steps --post-steps --ports shortpath`; `fly_rewire.py relthr5 / relthr_mono`; `optimizer_variants.HybridMuon` (Muon FP32 sulle nn.Linear, Adam sul resto, catturabile in CUDA Graph; passare `adam_cls=original_adam` dal monkeypatch); `phase6c_inference.py --checkpoint-path` (scarta core.src32/dst32; NON valido per checkpoint con varianti di dinamica).
## Stato della suite (tutto in `SUITE_TEST_FASE_6B.md`)

Tutte le run corte sono eseguite e registrate: A1-A5, B1-B7, C1-C11, D2-D9, E1-E2, S0-S8b, F1-F8. Nessuna voce "in coda/in corso" resta nel file. Pavimento di rumore a 2000 update: 0,01 nat (S0c 5,080 contro pilota 5,087; fuso S8b 5,083).

Risultati chiave a 2000 update (riferimento reale 5,087, soglia 0,10):
- Null: tutto batte il reale. Scala: reale 5,09 > entro superclasse 5,005 > gradi conservati 4,93/4,90 > ER 4,676 > segni permutati 4,645 ≈ porte random 4,639 > configuration model 4,533. Pesi permutati 5,103 (nulla: i pesi sono allenabili).
- Scelte biologiche che aiutano: lettura per gruppi anatomici D3 4,848 (+0,24); soglia sinaptica relativa D8 4,961 (+0,13); 6% segni capovolti D9 4,982 (+0,11); potenziali di inversione D7 5,044 (+0,04, sopra il rumore, sotto soglia).
- Neutre sul reale: APL D4, lobo ottico graduato D5, tau per tipo D6 (leak fermi: Adam 1e-4 non li muove; da rifare con lr dedicato).
- Categoria F (regioni specializzate, core PyTorch di riferimento perché `phase6d_queue.py` non aveva `--fast-mode fused`; risultati validi): occhi F1 5,906 (unigramma), fru F2 5,285, occhi+fru F3 5,127 (interazione −0,98 nat, unica coppia anatomica che funziona: via visiva del corteggiamento), hub F4 5,781 (hub inibiti, muti), CX F5 5,529, udito F6 5,182, occhi+hub F7 5,904. F8 (`phase6d_hops.py`, `results/phase6d_hops.json`): la CE segue la distanza in sinapsi porte→lettura; eccezioni hub (vicini, inibiti) e F3 (stessa distanza di F1, via specifica).
- Velocità: kernel fuso cupy (`fly_core_fast.FusedCore`/`FusedVariantCore`) ×1,73, equivalente a 100 e a 2000 update; riordino nodi (S8 5,121) NON equivalente perché gruppi di lettura e porte sono definiti sull'indice dei nodi; FP16/INT8/CSR/Adam fused: niente.

## Standard fissato dall'utente (16 settembre, 18:30)

`RICERCA_ATTIVITA_CERVELLO_MOSCA.md` (16/9, 3 agenti): perché l'86% è muto (nei LIF whole-brain pubblicati è attivo lo 0,03-0,4%), leve riviste in ordine di evidenza (1 eccitabilità per tipo + conduttanza, 2 più sottopassi, 3 porte anatomiche corte: ocelli/JO/LC, 4 sinapsi con tau, 5 soglia relativa a 6,24 M, 6 segno monoammine, 7 scala calibrata + rumore). Nessuna scelta presa: opzioni per l'utente.

`STANDARD_MOSCA_6B.md`: grafo intero con archi a soglia relativa (relthr s41), segni dal neurotrasmettitore, porte random 17.937, lettura discendenti a blocchi, LIF, kernel fuso. Combinazione mai eseguita insieme: primo run a 8000 col comando nel file, solo col via dell'utente.

## Decisioni che spettano all'utente (non prendere da soli)

1. Adottare il kernel fuso sul modello principale (matematica identica, ×1,73). Il riordino solo con permutazione dei buffer a gruppi/porte invariati (da implementare, verificare con un S8c a 2000).
2. Run lungo di F3 (occhi + fru/dsx): pendenza migliore fino a 1500 con un terzo dell'attività.
3. Combinazione delle vincitrici (porte random + segni permutati + lettura anatomica, eventualmente + reversal + soglia relativa) come base per i test successivi (A1, C1, C2, C5, C7 sulle promosse).
4. D6 con lr dedicato per `leak_logit` (es. 1e-2) o init per superclasse.

## Da rifare / lista lunga (documentata nel file suite)

A5 con offset random delle finestre; C10 sonda posizionale (fallita); C6 per tipo cellulare; raggio spettrale all'init; null "solo porte riassegnate"; 2.639 e 1.733 porte random a pari capacità (olfatto/udito); raggiungibilità pesata per segno (F1 contro F3); lettura sul potenziale invece del tasso (hub); S9 backward sparso; 2.6.2 compito affine, 2.6.6 e-prop, 2.5.10 ritardi anatomici, 2.8.2 prequenziale per update (non eseguiti, da proporre).

## Documentazione

Chiusa il 16 settembre 02:55: `RESOCONTO_FASE_6_CONTROLLI.md` §6c, `PIANO_FASI.md` (riga fase 6 + Evidenza6b), `MEMORIA_PROGETTO.md`, questo handoff, memoria `progetto-flytollm.md`. Prossimo passo: le decisioni dell'utente elencate sopra; nessun run da lanciare senza il suo via.

## Strumenti scritti oggi

`phase6_record.py` (recorder, idempotente; sezioni chiuse al primo tra `###`, `##`, `---`; generazione su una riga), `phase6d_hops.py` (distanze porte→lettura), `phase6f_promote.py --list` (candidate), `phase6s_speed.py --compare-only` (tabella velocità), `phase6c_reachability.py`, `phase6c_summary.py`, code `phase6c/6d/6e_queue.py`.

## Regole dell'utente

Nuove il 17/9: (1) ogni test corto con risultato diventa candidato per il default, proporre la combinazione senza aspettare; (2) orari solo da `date` e dai timestamp dei file di stato, durate solo misurate (2000@4+4 15 min, 8+8 27 min, K5 46 min, 8000@4+4 56 min, 8000@8+8 100 min).

Caveman ultra ovunque; agenti Opus 5 effort high con snippet caveman; subagent meccanici haiku. Grafo intero, mai pruning. Ogni modifica architetturale al modello principale: chiedere. Non decidere scala/budget: opzioni con numeri. Controllare sempre i worker in esecuzione (processi, log, coda, GPU) prima di rispondere o aspettare; mai output dei worker in pipe, sempre su file di log. Non lanciare nulla mentre l'utente gioca o dorme senza il suo via esplicito (oggi il via è stato dato per tutta la suite e per le promozioni). Riportare "eseguito" contro "criterio superato". Stampare il risultato di ogni run all'utente appena finisce e scriverlo nel file suite.

## Repo pubbliche

- https://github.com/ArtyomITA/flytollm + https://artyomita.github.io/flytollm/ (vetrina, sorgente `E:\claudecode pesante\flytollm-showcase`).
- https://artyomita.github.io/ (profilo, sorgente `E:\claudecode pesante\artyomita-site`). Per pushare: `gh auth switch --user ArtyomITA`, poi tornare a `adrian-iancu_ktcsrl`.
