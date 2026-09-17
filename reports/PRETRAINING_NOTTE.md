# Pretraining continuo — piano operativo

**Avviato13settembre:** H1-10 ripresa da2000, update2016 effettivo verificato. `HOLD_PRETRAINING` rimosso dopo verifiche; launcher27100, primo worker20680. Stato autorevole: `results/pretrain_actual_night_live.json`; avanzamento nei JSONL job corrente. PID = foto lancio, non identificatori permanenti.

**Via utente:** completa3.5+4, aggiorna piano, verifica robustezza, poi lancia senza altra conferma. `HOLD_PRETRAINING` resta fino fine verifiche; rimosso solo al lancio autorizzato. Vecchia coda ferma, pilota preservato. `phase35_4_compare_queue.py` termina dopo confronto; avvio lungo separato, risultati verificati.

Richieste13settembre: unisci3.5+4, scegli head/soglia dai risultati; continua fino stop, checkpoint ogni2000update; pausa20k per suiteMoE osservazionale poi riprendi. Luna high: `REPORT_LUNA_MOE_GRAFO_SUITE.md`.

## Sequenza

1. Smoke trainer10/tied e5/separate; ripresa in processo nuovo, verifica bitwise pesi, Adam, stato ricorrente. Correttezza replay CUDA verificata vs eager. Bitwise traiettoria successiva non garantito: riduzioni atomiche CUDA.
2. Smoke ogni modalità suite post20k, incluso reset; smoke CPU statistiche/null e gestione errori suite.
3. Confronto TinyStories, **2000update per candidata**, stessi seed17, storie complete, ordine, batch2,16posizioni/update, TBPTT8, Adam1e-4/clip1. Tied10→separate10→separate5. Max64000target/candidata, effettivi registrati; PAD escluso. BOS input predice primo token. Nessun attraversamento storie.
4. Scelta autorizzata: H1 solo se CE media due valutazioni finali <tied10; soglia5 solo se H1 scelta e CE5≤CE10. Verifica stesso numero target e hash iniziale interfacce. Tutte finite/entro guardie. Scelta provvisoria locale, non prova multi-seed né convergenza. Se nessuna migliora da init, stop e report.
5. Replica candidata a2000update con seed23; audit64 una volta su candidata seed17 congelata. Audit non seleziona profilo né modifica iperparametri. Smoke harness audit su soli DEV; verifica extra ripresa8→9 dopo cinque sonde su checkpoint H5 smoke.
6. **Due pretraining sequenziali fino8000update ciascuno**, continuando rispettivi checkpoint2000. Vince tied10: coppia tied10/H1-10; vince H1: coppia H1-10/H1-5. Parte prima candidata2000. Stessi target effettivi, ordine storie e seed; tempi diversi registrati. Dopo ogni8000: stessa suite osservazionale già verificata, fallimenti non bloccanti. Scegli a8000 solo con medesima regola CE e modifiche autorizzate; sonde non cambiano scelta/pesi.
7. Continua vincitore8000 fino **20000update totali**. Adam/stato/cursore preservati; non ripartono da zero.
8. Salva, chiudi worker/GPU; suite osservazionale in processi isolati. Riparti da stesso checkpoint fino stop utente. Nessun router/maschera/peso adottato dopo suite.

Questa run = **pretraining esplorativo**. Non marca superati criteri memoria sintetica o qualità linguistica non ancora dimostrati. L16d1/2000:75,76% vsL8:68,75%; test d8 lunghi rinviati per priorità utente. TBPTT8 invariato nel confronto linguistico per non mescolare altri cambiamenti.

## Checkpoint / stop

- Ogni2000update: `.latest.pt` atomico e `.stepXXXXXXXX.pt` conservato; `.best.pt` per CEDEV più bassa. Generazioni fisse, CE, gradienti, telemetria ogni intervallo. Stop/evaluation finale salva anche fuori multipli2000.
- Payload: pesi+buffer grafo, config, Adam(momenti/step), voltage/spike/cacheKV/posizioni, RNG CPU/CUDA/Python, cursore storia/offset/epoca, metriche e hash codice/tokenizer/manifest.
- Caricamento Adam via copia negli stessi tensori catturati; nessuna sostituzione indirizzi dopo cattura CUDA.
- Continuo per epoche, coppie sequenziali da storia130; cache/stato resettati a ogni coppia. Storielle intere, non solo prefisso128. Validation resta DEV16/prefisso128 fisso; audit64 una volta prima della notte su scelta congelata; test finale chiuso.
- Baseline unigramma/bigramma conta solo primi2000update poi resta congelata; evita crescita RAM illimitata. Non confondere conteggio baseline con tutti i token del pretraining.
- Stop morbido: crea `STOP_PRETRAINING` nella radice progetto, oppure chiedi all’assistente di fermare. Controllato a ogni update/fra test. Per ripartire rimuovi consapevolmente il marker e usa checkpoint/config coerenti.
- Guardie paging autorizzate: RAM/commit/VRAM, heartbeat90s, cinque update consecutivi>2s, nonfinite, disco<2GiB. Regressione DEV>1nat dal migliore interrompe e preserva checkpoint. Nessun auto-retry di training fallito.
- Nessuno stop a ore richiesto; timeout tecnico max1anno per worker continuo, oltre normale vita di questa run.

## Suite a20k, modello congelato

| Prova | Misura/controllo |
|---|---|
| G1 train osservazionale | Storie train9000..9015,128token; attività/corrente/potenziale per superclass×lato; campioni pertoken |
| G2 DEV osservazionale | DEV16 fissa; stesse misure, CE/entropia, copertura167k nodi; proxy flusso fra gruppi |
| G3/G4 gruppi null train/DEV | Permuta etichette entro lato e bin log2 dei gradi entranti/uscenti; cardinalità gruppi esatta, gradi approssimati |
| G5 contesto reset | DEV identica; reset stato+cache+posizione pertoken; ΔCE non isolabile alla sola attenzione |
| G6 statistiche held-out | Tabella train token→distribuzione attività gruppi; shrinkage10 fisso, confronto DEV con prior globale;32shuffle etichette token entro storia train |

Estensione già implementata e smoke-testata: entropia/load pergruppo; massa top1/2/4/8; numero effettivo gruppi; stabilità stesso token fra storie contro token diversi con posizione/frequenza approssimativamente abbinate; intervallo bootstrap200 repliche perstoria sul guadagno held-out. Statistiche derivate dalle stesse osservazioni, non nuove run training.

Test strutturali `anatomy_fanout_tests.py` eseguiti ora su10/5: seed32 ascendenti, discendenti, sensoriali, hub globali; ciascuno con controllo grado/lato; espansione diretta a1/2/4/8/16 archi, copertura readout e gruppi. Seed fissati su10 e riusati su5. Geometria non usata come criterio. Risultati `results/anatomy_fanout_thresholds.json`, inclusi automaticamente nell’analisi20k. Topologia fissa: non serve ricalcolarla ogni checkpoint; confrontala con attività/pesi appresi.

G6 = stima descrittiva su istogrammi, non router inserito nel modello. Cross-entropy proxy diversa da CE linguistica. Firing post4 medio, non segnale completo di tutti gli8sottopassi. Flusso proxy=`rate_medio_sorgente × peso`; non misura traiettorie causali. Gruppi superclass/lato non neuropili. Nessun batch×archi materializzato: aggregazione archi CPU a blocchi262144 per sola statistica.

Ogni worker max600s, risorse liberate prima del successivo. Errori sonde registrati; checkpoint sorgente hash invariato; ripresa training prevista comunque. Stop esplicito o risorse ancora insufficienti prevalgono. Analisi CPU timeout120s. Sonde causali e nuovi router = fase successiva da proporre all’utente.

## Quanto TinyStories?

Manifest locale:555504351token train,2717493storie. Con max32target/update,20000update≤640000target: circa0,115% dei token del corpus, **non un’epoca completa**. Nessun numero magico di update. Confronta target effettivi/CE/qualità, non passi di Transformer con batch molto maggiori.

[Paper TinyStories](https://arxiv.org/html/2305.07759v2): piccoli GPT-Neo, tokenizer top10k, contesto512/window256, fino30oreV100 per i modelli descritti. Evidenza che piccoli Transformer possono generare storie; nessuna garanzia trasferibile alla SNN corrente. [MoEUT ufficiale](https://github.com/RobertCsordas/moeut): profilo44M disponibile, spunto per routing futuro.

## File

`pretrain_resumable.py`: trainer. `pretrain_night_queue.py --from-comparison`: selezione e orchestrazione. `graph_specialization_probe.py`: sondeCUDA. `analyze_graph_specialization.py`: statistiche/nullCPU. Stato confronto: `results/phase35_4_compare_live.json`; stato nuova notte: `results/pretrain_actual_night_live.json`. Vecchio `pretrain_night_live.json` resta annullato. Risultati/job: JSON+JSONL+console; checkpoint separati.

Coda impedisce sospensione automatica sistema mentre lavora; schermo libero di spegnersi. Ripristino richiesta energetica alla fine. Sette test CPU verificano selezione, budget uguali, sonde fallite, errore avvio, output preesistente preservato, stop esplicito e sequenza8000→20000→ripresa da sorgente identica.
