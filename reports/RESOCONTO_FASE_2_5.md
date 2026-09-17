# Fase 2.5 — verifica integrata

Fatto su CNS soglia10: **166.700 neuroni, 2.753.975 archi, 4.410.893 parametri**. Architettura uguale: BPE4096, embedding256 condiviso, 4 sottopassi prima + 4 dopo attenzione; corrente post = token + feedback. Fase verifica esecuzione e gradienti; non prova capacità linguistiche.

## Cosa è stato verificato

- **134 token, cache128, batch2:** replay CUDA Graph = riferimento GPU eager; errore max logits 4,77e-7. Eviction ok, sequenze indipendenti, padding e reset anche su slot inattivo.
- **Causalità:** cambiare token futuri non cambia prefisso. BOS azzera memoria e stato slot, taglia gradienti verso storia precedente; altri slot vanno avanti.
- **Gradienti:** finiti e non nulli per tutti i tensori parametri nel probe: sinapsi, embedding, porte, gate, Q/K/V/O. Non vuol dire ogni sinapsi riceve gradiente utile.
- **Training con memoria:** sei finestre, B2×L4, TinyStories train storie0/1, 48 posizioni. Forward, backward, Adam e copia stato catturati. Parametri e momenti = riferimento eager; errore max parametri 4,77e-7.
- **TBPTT:** dopo backward/update teniamo valori stato/cache ma tagliamo grafo autograd. Finestra dopo ricorda passato senza retropropagare per tutte le finestre. Stato prodotto da pesi precedenti: approssimazione esplicita, da valutare in fase3.

Generazione greedy/campionata, EOS e salvataggio/caricamento già visti in 2.4. Non nuove misure qualità.

## Dove costa

Probe training Adam, media 6 replay B2×L4: **165 ms/finestra**; GPU forward+loss 53,1 ms, backward 109,7 ms, optimizer 2,26 ms. Circa 48 posizioni/s nel microtest, esclusi caricamento dati e logging: non throughput promesso del pretraining.

Profilo forward isolato B2: nucleo pre 5,84 ms + post 5,73 ms; attenzione 0,177 ms. **Priorità ottimizzazione: propagazione grafo e backward.** Sostituire subito attenzione dà poco margine nel profilo attuale. Tempi isolati non sommano al training completo.

Picco allocato nel probe carry 387,6 MiB, uso GPU campionato circa 1.351 MiB. Include finestre piccole: non estrapolare a L128/full graph. Ogni worker completo rilascia i tensori: allocatore 0 MiB; nessun fallback modello su CPU.

## Cosa racconta il cervello iniziale

- Circa 90% neuroni non spara nei quattro sottopassi visti; firing medio circa 5%. Non uguale a neuroni morti: potenziale e ingressi cambiano con sequenza/training.
- Solo 1,45% nodi letti spara almeno uno spike nella fase post del campione; leggiamo anche il **potenziale**, quindi lettura non per forza nulla.
- Feedback RMS 0,0329 contro corrente token 0,699 sulle porte: circa 4,7%, gate 0,1. Ramo attenzione riceve gradienti; utilità da provare.
- Raggiungibilità strutturale diretta dalle porte: 162.034/166.700 nodi; readout 2.204/2.241. **37 nodi letti irraggiungibili a soglia10.** Nessuna rimozione; controllare recupero archi in fase4. Raggiungibilità ignora segni e soglia firing: non prova propagazione reale.
- Potenziale RMS 0,407, max assoluto 7,69; inibizione dà potenziali negativi sotto −1. Da monitorare nel tempo, da solo non prova instabilità.

## Checkpoint delle attivazioni

Helper opzionale `lm_checkpointing.checkpoint_forward`: ricalcola segmenti nel backward per tenere meno attivazioni. Stesse equazioni; non adottato nella baseline.

| Segmento token | Picco incrementale | Forward+backward |
|---|---:|---:|
| Nessuno | 195,76 MiB | 0,794 s |
| 2 | 113,13 MiB | 1,061 s |
| 4 | 144,96 MiB | 0,889 s |
| 8 | 200,99 MiB | 1,021 s |

B2×L8, singola misura dopo warmup **GPU eager**, non benchmark CUDA Graph. Segmento2: −42% memoria incrementale, +34% tempo nel probe. Errore relativo L2 max per tensore gradiente <2,9e-6. Segmento8 peggiora memoria: segmentazione va misurata. Compatibilità training con checkpoint dentro CUDA Graph resta valutazione fase4.

## Ottimizzatori e prossima fase

Confronto richiesto AdamW/Muon/ROOT separato in [CONFRONTO_OTTIMIZZATORI.md](CONFRONTO_OTTIMIZZATORI.md). Varianti diagnostiche; baseline Adam invariata.

**Fase3:** provare overfit controllato, apprendimento su task copia/ritardo e miglioramento validation; seguire attività, gradienti, stato dopo update e contributo attenzione. Poi decidere LR/gate e ottimizzatore su prove comparabili. Fase4: accelerare costo dominante, provare checkpoint catturato e recuperare connessioni. Modifiche architetturali vogliono discussione prima.

Config riferimento: `configs/diagnostic_v1.json`. Evidenze: `results/phase25_{toy,cns}_*.worker.json` e relativi log/supervisori. Hash del runner possono differire fra prove per miglioramenti della verifica; i risultati registrano gli hash usati davvero. Nessuna retrodatazione hash.