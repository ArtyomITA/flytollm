# Suite soluzioni — grafo intero, lettura e memoria

Obiettivo: usare massimo male-CNS per linguaggio; instradare/leggere meglio, non costruire grafo artificiale piccolo. Numero nodi, ID, direzioni, segni preservati. Soglia sinaptica10 riferimento; no pruning. Soglia firing LIF1 distinta da soglia conteggio sinapsi10/5. MHA4 e KV128 già fatti; no nuova attenzione ora.

## Cosa significa calibrare

**Controllo T1:** dimostrare che task/dati/budget permettono a modello di riferimento di imparare. Non scegliere struttura mosca. T1b risolto localmente: GRU d1 train/DEV100%, CE0,00242,1000update,62.000 target,10,01s; T1 a200update insufficiente. d4/d8 non coperti da questo successo.

**Lettura mosca:** ridurre divario tra segnale disponibile e predizione effettiva. T2: rappresentazione finale probe84,77%, head18,85%. Non cambiare subito head: prima pareggiare budget/obiettivo e verificare blocchi esistenti. Probe separato vedeva102.400 target ripetuti contro2.560 nuovi mosca.

**Dinamica/routing:** verificare che ingressi, cammini, uscite portino segnale utile sotto training, non massimizzare firing. Nodi bassa attività possono conservare potenziale o servire altri contesti; no eliminazione automatica.

## Ordine esperimenti

| ID | Domanda / fattore isolato | Prova concreta e budget massimo proposto | Decisione | Stato |
|---|---|---|---|---|
| C1 = T1b | Bastava più budget al controllo d1? | Stessa GRU/LR/seed,1000update,stream nuovi | d1 calibrato per controllo; no verdetto mosca | **Eseguito:100%DEV** |
| M1 = T4 | Cache contiene/usa informazione? | Pesi fissi; reale/vuota/contenuto invertito/riordino congiunto; simboli+TinyStories | Separare contenuto da uso e utilità | **Eseguito** |
| M2 = T5 | L16 aiuta se update uguali? | d1/d8,200update/config,un update/16token; solo detach diverso | No vantaggio utile a200update; capacità a budget maggiore non esclusa | **Eseguito**; [risultati](RESOCONTO_T1B_T4_T5.md) |
| R0 | Gradiente input/output E compete? |8 coppie DEV simboli +8 coppie train TinyStories,ultimo target; no update. Scomposizione E lookup/output, norme e coseni SOLO stesso spazio parametri | Conflitto persistente ipotesi supportata; no PCGrad automatico | Proposto |
| R1 | Lettura originale impara a budget adeguato? | E/porte/core/attenzione congelati; allenare solo norm/proiezione/output-norm già esistenti su pooled154 congelate. Stessa head4096 tied, no parametro nuovo;102.400 presentazioni target,Adam0,001,clip1; train/DEV separati | Se offline migliora, verificare circuito completo prima di adozione | **Eseguito**:75,59%DEV offline; R2:74,61% nel circuito |
| R2 | Miglioramento R1 sopravvive nel circuito? | Reinserire soli pesi readout R1; inferenza CUDA Graph su stessi DEV e contesti nuovi; confrontare head/CE, correnti, cache, firing | Readout modifica cache/feedback: offline successo non basta | **Eseguito**:74,61%DEV,CE1,0565 |
| P0 | Dove conviene instradare sul grafo intero? | Audit topologico per classi/nervo/lato e uscite: raggiungibilità, distanze dirette, quota porte raggiunte; gradienti e sensitività su train TinyStories, massimo16 coppie×16token | Classifica candidati da verificare DEV; nessun nodo migliore a priori | Proposto, osservazionale |
| P1 | Assegnare canali alle porte per ruolo aiuta? | Stessi17937 ingressi,fanin8,256 canali,numero parametri; solo mappa canale→nodo stratificata per annotazioni disponibili. Un confronto col riferimento,200update | Deve migliorare CE/lettura DEV, non solo copertura | **Cambio interfaccia: domanda prima** |
| P2 | Raggruppare uscite per ruolo aiuta? | Stessi2241 nodi,77 gruppi e154 feature; sola partizione anatomica deterministica, pool potenziale/rate separati. Confronto200update | Guadagno senza più parametri; misura gap grezzo→pooled | **Cambio readout: domanda prima** |
| D1 | Scala surrogate è limite? | Solo se R0/T3 lo indicano: una larghezza alternativa esplicita,200update,stessa topologia/soglia forward | Gradiente migliore deve tradursi in CE/lettura; firing isolato non basta | **Cambio backward: proposta precisa prima** |
| D2 | Inizializzazione/corrente o soglia LIF limita? | Una sola variabile da definire dopo T3; budget200update,stessi dati. Non insieme a D1/P1/P2 | Migliorare apprendimento senza distruggere segni/stabilità | **Cambio dinamica: domanda prima** |
| H1 | Weight tying è davvero limite? | Solo dopo R1/R2/R0: output separato inizializzato da E,stessi dati/budget; aggiunge1.048.576 parametri | Confronto capacità/costo esplicito; non prima scelta | **Cambio architettura: domanda prima** |
| L1 = T6 | Funziona su linguaggio nuovo? | Pilot100k target nuovi,DEV ai punti prefissati,unigram/bigram,stabilità/generazione; secondo seme solo se promettente | Apprendimento generalizzato e uso contesto; poi audit64 una volta | Non avviato; budget temporale da concordare |

Ordine minimo: **C1/T4/T5 → R0/R1/R2 → P0**, poi scegliere UN intervento solo se necessario → pilot. Non eseguire tabella come griglia combinatoria. P1/P2 alternative guidate da diagnosi, non obbligatorie.

## Mapping senza ridurre nodi

- `class/subclass`, nervo ingresso/uscita, lato, neuromero disponibili localmente: gruppi routing/readout candidati.
- Mappare neuropili richiede annotazioni ROI/syn-partners ufficiali; verificare copertura ID prima. Non rinominare blocchi numerici come regioni biologiche.
- Topologia misura cammini POSSIBILI. Gradiente è sensibilità locale surrogate, non prova di percorso causale. Combinare topologia, variazioni segnale, interventi temporanei, validazione su testo distinto.
- Ranking per TinyStories stimato solo su train; DEV per verifica candidata fissata, audit riservato chiuso. Regione forte su d0 non necessariamente migliore per next-token o memoria.
- Cambiare porte non elimina nodi: resto circuito resta disponibile. Nessuna garanzia che ogni neurone contribuisca a ogni token; registrare copertura invece di imporre firing uniforme.

## Giudizio sulla ricerca Luna

Accolti: separare budget/readout; misurare contributi di E condivisa; porte/pooling informati da annotazioni; distinguere soglie; grafo intero. [Ricerca](REPORT_LUNA_T23_SOLUZIONI.md).

Ordine corretto vs proposta iniziale Luna: **R1 originale prima di H1 untied**. Weight tying ha vantaggi documentati nei LM convenzionali; probe diverso non dimostra che vada rimosso. [Press–Wolf](https://arxiv.org/abs/1608.05859). Controlli probe necessari, ma non prova automatica di capacità linguistica. [Hewitt–Liang](https://aclanthology.org/D19-1275/).

PCGrad/e-prop restano analogie, non correzioni. Loss con contributi diversi non è automaticamente problema multi-task. Surrogate/init richiedono confronto utente anche conservando topologia. Le fonti non danno soglia firing o budget universale.

## Quando chiudere fase3

- Correttezza GPU/reset/cache/gradienti e cleanup verificati; già dimostrati entro smoke.
- Lettura originale apprende oltre frequenze e generalizza; non solo probe esterno.
- Memoria valutata con budget/update comparabili; fallimenti storici80% restano dichiarati.
- Pilot testo nuovo: vantaggio su baseline senza contesto, uso contesto documentato, stabilità e generazioni esaminate; replica se promettente e audit64 una volta.
- Nessuna promessa di battere modelli storici prima di benchmark a parametri/dati/budget comparabili. Fase3 aperta finché mancano evidenze; vietato chiuderla rinominando criteri.
