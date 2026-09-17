# Handoff fase 6/7 — 17 settembre 2026, 03:00 (notte autonoma: code 7c → 7d → 7e → 7f incatenate)

Leggere questo prima di tutto dopo un compact. Stato autorevole del progetto: `PIANO_FASI.md`, `MEMORIA_PROGETTO.md`; controlli fase 6: `PROTOCOLLO_FASE_6_CONTROLLI.md`, `RESOCONTO_FASE_6_CONTROLLI.md`; suite test corti con pre-registrazioni, risultati e considerazioni di ogni test: `SUITE_TEST_FASE_6B.md` (file di tracciamento richiesto dall'utente: "per ogni test devi tracciare tutto"); ricerca bibliografica: `RICERCA_CONNETTOMA_LM.md` (§0 = correzioni dai fatti del progetto).

## Cosa sta girando adesso (notte 17 settembre, utente a dormire dalle 02:55)

Catena automatica, un worker GPU, log unico `results/phase7_queue.console.log`, stati `results/phase7{c,d,e,f}_live.json`:
- **7c** (sezione H, in corso): H1 bias per tipo lr 1e-3 = 5,037 (instabile, −0,46); H2 tau per tipo lr 1e-3 = 4,519 (+0,06, stabile); H3 conduttanza sulla baseline = 4,459 (+0,12, sopra soglia); poi H4 bias+tau+conduttanza, H5 monoammine −1, H6 scala pesi ×0,5 / ×2.
- **7d**: bias per tipo con lr 1e-4 e 3e-4 (via utente).
- **7e** (sezione I): Muon sulle matrici dense (base 1e-4, 3e-4, con 8+8, anche sulla testa) + G5 = 8+8 + conduttanza (via utente).
- **7f** (sezione J, test consigliati da me con via generale dell'utente): tau lr 3e-3; tau + conduttanza; inferenza sul checkpoint 8+8; promozioni a 8000 di 8+8 e conduttanza (e di 8+8 + conduttanza se G5 batte G1 di ≥ 0,10).
- Altri test decisi in base ai risultati (richiesta utente: "anche test futuri di rilevanza dei prossimi test"): aggiungere code 7g+ senza chiedere, purché brevi (≤ 2000 update) o promozioni da regola.

Regole della notte: 3 minuti di riposo ogni ~1,5 h di attività GPU (accumulatore `results/.gpu_activity.json` letto da `pretrain_night_queue.run_job`; la coda 7c gira col codice vecchio, per lei c'è `PAUSE_BETWEEN_RUNS`=180 rimosso da un watcher dopo l'avvio di H5); `pause_runs.py pause|resume` per la pausa vera; `STOP_PHASE7` ferma le catene tra un run e l'altro. A ogni fine run: `phase6_record.py` + considerazioni nel file suite + stampa in chat.

Fase 7 (sezione G) chiusa alle 00:50: baseline standard 4,580 / 3,999 (2000 / 4000); 8+8 sottopassi +0,181; porte anatomiche corte −0,279; 6,24 M archi +0,047; tre insieme −0,222; inferenza sui checkpoint 1000/2000/4000 (G0i).

Codice nuovo (modello principale intatto): `fly_core_variants.py` varianti combinabili con '+' e `bias_type`; `pretrain_control.py --type-param-lr --weight-scale --optimizer muon --muon-lr --muon-head --pre-steps --post-steps --ports shortpath`; `fly_rewire.py` kind `relthr5`, `relthr_mono`; `optimizer_variants.HybridMuon`; `phase6c_inference.py --checkpoint-path` (scarta `src32/dst32`; NON adatto a checkpoint con varianti di dinamica: ricostruisce un LIF semplice); `pause_runs.py`; code `phase7*_queue.py`.

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

Caveman ultra ovunque; agenti Opus 5 effort high con snippet caveman; subagent meccanici haiku. Grafo intero, mai pruning. Ogni modifica architetturale al modello principale: chiedere. Non decidere scala/budget: opzioni con numeri. Controllare sempre i worker in esecuzione (processi, log, coda, GPU) prima di rispondere o aspettare; mai output dei worker in pipe, sempre su file di log. Non lanciare nulla mentre l'utente gioca o dorme senza il suo via esplicito (oggi il via è stato dato per tutta la suite e per le promozioni). Riportare "eseguito" contro "criterio superato". Stampare il risultato di ogni run all'utente appena finisce e scriverlo nel file suite.

## Repo pubbliche

- https://github.com/ArtyomITA/flytollm + https://artyomita.github.io/flytollm/ (vetrina, sorgente `E:\claudecode pesante\flytollm-showcase`).
- https://artyomita.github.io/ (profilo, sorgente `E:\claudecode pesante\artyomita-site`). Per pushare: `gh auth switch --user ArtyomITA`, poi tornare a `adrian-iancu_ktcsrl`.
