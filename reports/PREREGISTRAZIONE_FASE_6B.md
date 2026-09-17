# Pre-registrazione fase 6b — test corti, punto 1 (14 settembre 2026, 21:50)

Scritto prima di lanciare i run. Pratica presa da FlyGPT: soglie e letture fissate prima dei risultati, così nessuna lettura viene scelta a posteriori. Vale per i test del punto 1 ("cosa rubare ai sei progetti"); ogni punto successivo avrà la sua pre-registrazione prima del lancio.

Riferimenti fissi: mosca reale H1-10 a 2000 update, CE DEV16 5,087 (seed 17) e 5,239 (seed 23); variabilità da init tra i due seed 0,152 nat; rimescolato a gradi conservati seed 41: 4,931 a 2000, 3,814 a 8000; mosca reale a 8000: 3,959. DEV16: 2.048 target. Tutti i test qui sotto: un seed, nessuna decisione architetturale, grafo intero.

## 1.2 Secondo seed di rimescolamento a gradi conservati (seed 43, 2000 update)

Stessa pipeline di `phase6_rewired_h10_2000` con `--rewire-seed 43`. Differenza pre-registrata: Δ = 5,087 − CE(seed 43).

- Δ ≥ +0,10: il vantaggio del rimescolato è replicato con segno concorde su due seed di cablaggio (0,157 con il seed 41); resta un solo seed di init.
- −0,10 < Δ < +0,10: inconcludente, la differenza del seed 41 rientra nel rumore di cablaggio.
- Δ ≤ −0,10: contraddetto, il seed 41 era un caso fortunato.

## 1.5 Null "configuration model" (seed 41, 2000 update)

Grafo casuale con la stessa distribuzione dei gradi ma non gli stessi gradi per nodo: i ruoli di sorgente e di destinazione vengono ri-assegnati ai nodi con due permutazioni indipendenti, poi self-loop e duplicati vengono riparati con scambi. Cambia rispetto al rimescolato a gradi conservati: quale nodo è hub, il grado di uscita dei neuroni sensoriali e di ingresso dei neuroni di readout, la correlazione per nodo tra grado entrante e uscente, il bilancio eccitatorio/inibitorio per sinapsi (il segno resta quello del nodo presinaptico, legge di Dale). Resta: numero di archi, multiset dei gradi, pesi, porte, readout, init artificiale. Confronto con 4,931 (gradi conservati, stesso seed di init). Grafo costruito e verificato prima del lancio (`results/phase6b_configmodel_stats.json`): 2.753.975 archi, multiset dei gradi identici, correlazione per nodo tra grado originale e nuovo 0,001, archi originali sopravvissuti 1e-4, reciprocità 0,117 → 0,0001, archi uscenti dai neuroni sensoriali 167.131 → 291.847 (i sensoriali reali proiettano meno della media), frazione di archi eccitatori 0,633 → 0,642.

- |CE − 4,931| < 0,10: i gradi per nodo non contano oltre la loro distribuzione; il cablaggio anatomico non porta nulla nemmeno a questo livello.
- CE ≥ 4,931 + 0,10: l'assegnazione dei gradi ai nodi (chi è hub, quanto proiettano i sensoriali) conta; è l'unico livello di struttura con effetto misurabile finora.
- CE ≤ 4,931 − 0,10: anche l'assegnazione anatomica dei gradi è un vincolo sfavorevole.

## 1.3 Sonda lineare sullo stato del nucleo

Checkpoint: mosca reale a 8000 (`pretrain_night8000_h10.latest.pt`), rimescolata a 8000 (`phase6_rewired_h10_8000.latest.pt`), mosca reale a 26.247 (`pretrain_continuous.latest.pt`). Procedura: si fanno scorrere le storie token per token con la stessa procedura di valutazione (attention accesa, stato portato avanti), si registra a ogni posizione lo stato del nucleo dopo i sottopassi; caratteristiche = potenziale di membrana e spike dell'ultimo sottopasso, dopo il token, per un sottoinsieme fisso di 8.192 neuroni scelti a caso (seed 5) più tutti i 2.241 neuroni di readout; sonda = classificatore ridge lineare (caratteristiche standardizzate sulle statistiche di training, regolarizzazione λ = numero di campioni di training) addestrato sulle storie 0-47 dello split di training (mai usate dal pretraining, che parte dalla 130), valutato su DEV16, per predire il token a distanza k ∈ {0, 1, 2, 4, 8} indietro. Controllo di chance: stessa sonda con gli stati permutati a caso tra le posizioni. Riferimento: accuracy della classe più frequente.

- Il nucleo "tiene" il contesto a distanza k se accuracy(k) supera il controllo permutato di almeno 5 punti percentuali.
- Reale e rimescolato differiscono a distanza k se le loro accuracy differiscono di almeno 5 punti; sotto, uguali.
- Limite: la sonda misura contenuto decodificabile linearmente, non l'uso che il modello ne fa; lo stato include l'effetto del feedback dell'attention (porte di feedback), quindi non separa da solo nucleo e canale esterno.

## Punti senza run

1.1 è questo documento. 1.8 (igiene: BPE, corpus vero, baseline a pari dati) è già in atto. 1.9 (FlyGPT full-CNS su 6 GPU) resta da confrontare quando pubblicato. 1.10 (sottografi come banco prova) non si fa: viola la regola del grafo intero.
