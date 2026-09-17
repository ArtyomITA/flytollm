# Fasi3.5+4 e notte — 13 settembre2026

**Verifiche3.5+4 finite; H1-10 scelta, notte avviata.** Ripresa2000 e update2016 verificati nel worker20680. Risultato macchina: `results/phase35_4_final_review.json`; stato vivo: `results/pretrain_actual_night_live.json`.

## Confronto controllato

Stessi seed17,52.673 target a2000update, storie complete, ordine e init interfacce uguali. CUDA GraphFP32, batch2×16,TBPTT8,Adam1e-4,clip1,4pre+4post,MHA4/KV128. Tutti166.700nodi conservati. Soglia sinaptica ≠ soglia LIF1.

| Profilo | Parametri | CEDEV media finale | s/update | VRAM picco allocata |
|---|---:|---:|---:|---:|
| Tied10 | 4.410.893 | 5,761664 | 0,6829 | 430,17MiB |
| H1-10 | 5.459.469 | 5,086994 | 0,6671 | 457,71MiB |
| H1-5 | 8.947.612 | 5,211444 | 1,3023 | 656,95MiB |

CE minore = meglio. H1-10 migliora0,67467nat: beneficio netto locale, stesso budget e costo. Ripetizioni DEV5,086737/5,087250: variabilità CUDA piccola, ben sotto differenza tra head. Non due semi indipendenti.

H1-5 ripetizioni5,211335/5,211554: peggio diH1-10 a questo budget, quasi2× costo. Non adottata ora; confronto8000 lascia recupero con più training. Tutti tre worker: cleanup0MiB.

Baseline unigramma5,842783; bigramma4,283000, conteggi sui52.673 target train. H1 batte unigramma; resta sotto bigramma. DEV16×128token, non intero validation set.

Generazioni H1-10: frammenti grammaticali ("bird was a big"), ripetizioni, incoerenza. Qualità linguistica aperta. Test finale escluso.

## Portata della fase4

Confronto operativo10→5 e head separata, risorse misurate, backend custom CUDA Graph mantenuto. Nodi originali intatti; archi2.753.975→6.242.118. Raggiungibilità uscite2204→2232 su2241; guadagno strutturale non garantisce qualità.

Nessun nuovo router, sparsificazione nodi, CSR, quantizzazione o cambio dinamica. Restano candidati futuri, da motivare con profilo e verifiche. Passaggio notturno vuole profilo stabile misurato, non ogni backend possibile.

## Verifiche prima del lancio

- Trainer10tied e5H1: salvataggio/ripresa in nuovo processo, pesi/Adam/stato esatti; cleanup0.
- Cinque modalità suite grafo su5H1: tutte passate; anatomia/null stessa CE8,168309 sullo smoke, stesso digest checkpoint.
- Statistiche CPU: controlli sintetici, dati reali smoke, shuffle entro storia, bootstrap e concentrazione attività.
- Coda: sette test CPU; errori sonde e analisi non bloccanti, stop esplicito rispettato, sorgente ripresa20k identica.
- Ripresa dopo sonde8→9 passata: pesi/Adam/stato esatti, sorgente immutata, cleanup0. Smoke harness audit passato: CE concorda con sonda indipendente entro5,96e-8; audit non consumato nello smoke.
- Replica H1-10 seed23: CE8,332785→5,239103, stessi52.673 target,2000update. Sensibilità al seme presente; non replica intero confronto tre profili.
- Audit64, scelto prima dei risultati e consultato una volta sulla candidata17 congelata:8090target,CE5,161087,accuracy19,37%; unigramma5,860740,bigramma4,359582. Nessun tuning o selezione notturna sull'audit.
- Ripresa tre checkpoint allenati2000→2001: tutte passate; pesi/Adam/stato esatti, checkpoint immutati, cleanup0MiB. Totale14smoke GPU passati; sette test CPU coda passati.

Protocollo T6 originale100k target resta storico. Priorità utente aggiornata: confronto2000update,52.673 target; non equivalente a100k. Estensione8000/20000 dà poi nuovi budget osservabili.

## Notte autorizzata

Due profili fino8000update ciascuno, sequenziali; sonde grafo a ciascuno8000. Migliore per regola→20000→sonde→ripresa continua. Checkpoint ogni2000. Confronti usano DEV; audit64 escluso da selezione notturna.

H1 adottata se batte tied10; soglia5 se CE non peggiore di H1-10. Due semi descrivono sensibilità init, non garantiscono convergenza. Fase3.3 aperta: L16d1 migliora a2000 ma d8 lungo non completato; nessuna promessa che pretraining risolva la memoria.

Piano dettagliato: `PRETRAINING_NOTTE.md`. Arresto morbido: creare `STOP_PRETRAINING` nella radice. Ripresa vuole checkpoint e config coerenti. Nessun avvio doppio o sovrascrittura risultati.