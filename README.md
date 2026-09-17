# FlyToLLM · il connettoma parlante

**Il sistema nervoso completo di un moscerino maschio, usato intero come nucleo di un language model.**
166.700 neuroni, 2.753.975 connessioni reali, segni sinaptici presi dal neurotrasmettitore. Nessun taglio anatomico, nessun backbone pre-addestrato. Una sola GTX 1080.

▶ **Animazione della pipeline:** https://artyomita.github.io/flytollm/

*TL;DR (English): the whole male Drosophila CNS connectome (male-cns:v1.0, 166,700 neurons) is used, uncut, as the recurrent spiking core of a next-token language model on TinyStories. Synapse counts initialise the weights, neurotransmitters fix the signs (Dale's law), a small external attention re-enters through the input ports. Trained with surrogate gradients on one GTX 1080. Honest controls included: every null model beats the real wiring; the tests below say why, and which biological choices actually help. All code, protocols and results are in this repo (`src/`, `reports/`).*

## Lo scopo

- Far parlare il cervello della mosca: non un modello "ispirato" alla mosca, ma il cablaggio misurato, tutto, con i suoi segni.
- Usare quanto più possibile della natura del connettoma: topologia, conteggi di sinapsi, neurotrasmettitori, neuroni sensoriali come porte d'ingresso, neuroni discendenti e motori come lettura.
- Scoprire cosa fa davvero il cablaggio quando lo si costringe a un compito che non è suo, con controlli che possono dire di no.

Non è una gara di efficienza contro GRU e Transformer. È un esperimento per capire. Tutto open source: se un numero qui dentro serve a qualcuno, ha avuto senso.

## Come funziona

1. **Connettoma** male-cns:v1.0 (Janelia FlyEM · Google Research, Cell, settembre 2026, CC-BY): 166.700 neuroni, 25,6 M connessioni; entrano quelle con almeno 10 sinapsi.
2. **Neuroni LIF** (leaky integrate-and-fire) su tutto il grafo, spike binari, gradiente surrogato. Segno fisso dal neurotrasmettitore, magnitudine inizializzata dal conteggio di sinapsi e allenata.
3. **Il testo entra dai sensi**: ogni token BPE (vocabolario 4096) diventa un vettore di 256 dimensioni iniettato come corrente nei 17.937 neuroni sensoriali reali (o, nella configurazione standard attuale, in 17.937 neuroni scelti a caso: vedi i test).
4. **Attention fuori dal grafo**: una piccola attention causale (4 teste, cache di 128 token) legge lo stato e rientra dalle stesse porte, con un gate. Il connettoma non ha un asse "posizione nella frase", quindi la memoria di sequenza sta fuori.
5. **La risposta esce dai muscoli**: potenziale e tasso di spike dei 2.241 neuroni discendenti, motori ed efferenti, raggruppati in 77 popolazioni, poi softmax sui 4096 token.
6. **Addestramento**: 4 + 4 sottopassi per token, TBPTT 8 token, Adam, CUDA Graph FP32, 0,65 s per update su una GTX 1080 da 8 GB (0,38 s col kernel fuso). 5,46 milioni di parametri, quasi tutti sinapsi.

## Cosa abbiamo misurato finora

| | CE su 16 storie riservate (nat, più basso = meglio) |
|---|---|
| Mosca, cablaggio reale, porte sensoriali, 26.247 update | **3,634** |
| Mosca, cablaggio reale, porte sensoriali, 8000 update | 3,959 |
| Mosca, cablaggio reale intero, porte random, 8000 update | 3,689 |
| Mosca, cablaggio rimescolato a gradi conservati, 8000 update | 3,814 |
| Configuration model (topologia distrutta), 8000 update | 3,648 |
| GRU 2×256, stessi dati | 3,429 |
| Transformer 2 blocchi, stessi dati | 3,563 |
| Trigramma Kneser-Ney, stessi dati | 3,410 |

Generazione a 8000 update, prompt "Once upon a time": *", there was a little girl named Tim. She loved to play with the ball. The dog was very happy and happy."*

## Test eseguiti (settembre 2026)

Ogni test è pre-registrato (soglia di effetto 0,10 nat, pavimento di rumore misurato 0,01 nat tra due run identici), eseguito con la stessa pipeline e la stessa inizializzazione, e tracciato con curva ogni 500 update, verdetto e considerazioni in [`reports/SUITE_TEST_FASE_6B.md`](reports/SUITE_TEST_FASE_6B.md). Riferimento: mosca reale a 2000 update = 5,087 (a 8000 = 3,959).

**Null model sul grafo (cosa del cablaggio frena)**
- Rimescolamento a gradi conservati (due seed): 4,931 / 4,896 a 2000, 3,814 a 8000. Il cablaggio reale è un vincolo, replicato.
- Configuration model (permutazioni indipendenti di sorgenti e destinazioni): 4,533 / 3,648. Tetto della pipeline.
- Erdős-Rényi a pari archi: 4,676 / 3,697; peggio del configuration model di 0,14 a 2000 e 0,05 a 8000: gli hub aiutano solo all'inizio.
- Rimescolamento entro superclasse (matrice a blocchi conservata): 5,005. Metà del vincolo sta nella struttura a compartimenti.
- Pesi (conteggi di sinapsi) permutati: 5,103, nessun effetto: i pesi sono allenabili, l'informazione anatomica sta in ciò che resta fisso.
- Segni di Dale permutati tra i nodi: 4,645 / 3,723. Il pattern anatomico di inibizione è un freno.
- 6% dei segni capovolti (errore stimato del classificatore dei neurotrasmettitori): 4,982 / 3,897, nessun danno. Modello robusto agli errori di segno.
- Segno dei 541 neuroni monoaminergici (+1 vs −1): nessun effetto (0,4% degli archi).
- Porte random a pari capacità (17.937 nodi non di lettura): 4,639 / 3,689. Il freno principale del cablaggio reale è dove entra il testo: i sensoriali stanno in periferia, dietro l'inibizione locale; con l'intero grafo anatomico e solo le porte spostate si recupera l'87% del divario col tetto.
- Soglia sinaptica relativa a pari archi (un arco entra se pesa ≥ 1,31% dell'input del bersaglio): 4,961 / 3,903, sopra il reale lungo tutta la curva; l'unica scelta di selezione degli archi che aiuta.
- 6,24 M archi (soglia relativa al conteggio della soglia 5): +0,05 sulla baseline, gli archi deboli aggiungono poco.

**Sonde, mappe e lesioni (inferenza sui checkpoint)**
- 85,8% dei neuroni non spara mai nel cablaggio reale (81% con porte random, 74,6% nel configuration model); 10% attivi per token in tutte le condizioni.
- Lesione del rich club (10% dei nodi a grado massimo): +1,3 / +1,5 / +1,75 nat a 2000 / 8000 / 26.247, controllo a pari grado ≈ 0. Gli hub portano il calcolo.
- Lobo ottico intero (89.403 neuroni), Kenyon cell, complesso centrale: rimovibili senza cambiare la CE con la lettura discendente. Fuori percorso.
- Attention esterna: 0,07 nat a 2000, 0,24 a 8000, 0,46 a 26.247, tutta negli ultimi 8-32 token; il feedback cresce da 0,32 a 0,64 della corrente del token.
- Raggio spettrale dei pesi con segno: reale 0,50-0,53, rimescolato 0,19-0,21, porte random 0,73 → 0,35 col training.
- Robustezza al rumore in ingresso: uguale tra reale e null a pari CE (a 2000 sembrava diversa: confondimento del livello di CE).
- Sonda lineare sullo stato: il rimescolato decodifica il passato recente meglio (+6 punti a lag 1-2). Kenyon cell attive 0,02% nel reale, 11,7% con porte random (≈ 440 KC diventano porte).

**Regioni specializzate come porte e lettura (occhi, orecchie, corteggiamento)**
- Occhi (6.098 fotorecettori) in ingresso: 5,906, livello dell'unigramma. Orecchie/tatto (1.733 meccanosensoriali): 5,182, quasi pari con un decimo delle porte. Olfatto (2.639): 5,339.
- Lettura da fru/dsx (corteggiamento, 4.580 neuroni): 5,285; dagli hub (2.241 a grado entrante massimo): 5,781, muti per inibizione (potenziali fino a −33); dal complesso centrale: 5,529.
- Occhi + corteggiamento: 5,127, pari al reale, con interazione −0,98 nat: l'unica coppia anatomica che funziona è quella che il connettoma collega davvero (via visiva del corteggiamento maschile, LC10a → AOTU → P1).
- Analisi delle distanze (BFS): la CE a 2000 è monotona nella distanza media in sinapsi porte → lettura (random 1,14 → 4,639; anatomiche 1,62 → 5,087; udito 2,19 → 5,182; olfatto 3,39 → 5,339; occhi 3,66 → 5,906). Eccezioni: hub (vicini ma inibiti) e occhi → fru/dsx (stessa distanza di occhi → discendenti, esito opposto).
- Porte anatomiche a cammino corto (14.957: meccanosensoriali + neuroni di proiezione visiva, 1-2 sinapsi dai discendenti): 4,859, il miglior ingresso anatomico misurato, ma 0,28 sotto le porte random.

**Natura del connettoma come parametrizzazione**
- Lettura per gruppi anatomici (superclasse, nervo, lato): +0,24 a 2000, +0,015 a 8000. Vantaggio di ottimizzazione, non di capacità: falso positivo della regola di promozione.
- Sparsità APL sulle Kenyon cell, lobo ottico graduato: nessun effetto (regioni fuori percorso).
- Costanti di tempo per tipo cellulare (11.752 leak allenabili): nessun effetto a lr 1e-4 (non si muovono), +0,06 a lr 1e-3 (leak 0,90-0,96, stabile).
- Bias di riposo per tipo cellulare: instabile a lr 1e-3 (−0,46); rerun a lr più bassi in corso.
- Sinapsi a conduttanza (potenziali di inversione E_e = 3, E_i = −1): +0,04 su porte anatomiche, **+0,12 sulla baseline standard**. Potenziali limitati (minimo −1,3 invece di −6,3), stessa attività: gli hub sommersi d'inibizione tornano a portata di soglia.
- Scala dei pesi iniziali ×0,5: +0,07 (stesso meccanismo, in versione grezza).
- Più sottopassi per token (8+8 invece di 4+4): **+0,18**, la leva più forte; la profondità di calcolo conta più della raggiungibilità (sulle porte anatomiche corte vale solo +0,06). Costo ×1,8.

**Velocità (senza cambiare la matematica)**
- Kernel CUDA fuso (cupy: gather-moltiplica-accumula in una passata, salta gli spike a zero): ×1,73, equivalente al riferimento a 100 e 2000 update (loss 1e-6, gradiente 1e-8). Adottato per i controlli.
- Riordino dei nodi per (superclasse, grado): ×2,4 in tempo ma non equivalente (gruppi di lettura e porte definiti sull'indice dei nodi): non adottato.
- Chunk ×1,22; CSR cuSPARSE ×0,60; FP16 storage ×1,07; INT32 e Adam fused: niente. Micro-benchmark: gli atomici dell'accumulo dominano, non la banda dei pesi.
- Pausa vera dei run (sospensione dei processi) e 3 minuti di riposo GPU ogni ~1,5 ore di attività.

**Ricerca bibliografica (tre agenti, ~40 fonti)**: l'86% di neuroni muti è normale per i LIF whole-brain pubblicati (Shiu 2024: 0,03-0,4% attivi); le leve pubblicate per far parlare più cervello sono per tipo cellulare (flyvis: 734 parametri, potenziale di riposo e τ per tipo); occhio composto → discendente ≈ 4 sinapsi; il conteggio di sinapsi è un proxy grezzo dell'efficacia (EPSP da 0,4 a 6,6 mV). Dettagli: [`reports/RICERCA_ATTIVITA_CERVELLO_MOSCA.md`](reports/RICERCA_ATTIVITA_CERVELLO_MOSCA.md), [`reports/RICERCA_CONNETTOMA_LM.md`](reports/RICERCA_CONNETTOMA_LM.md).

**In corso (notte 17 settembre)**: Muon sulle matrici dense dell'attention, 8+8 + conduttanza, tau + conduttanza, promozioni a 8000 delle leve sopra soglia. Configurazione standard corrente: [`reports/STANDARD_MOSCA_6B.md`](reports/STANDARD_MOSCA_6B.md) (grafo intero con soglia relativa, porte random, lettura discendenti a blocchi, LIF, kernel fuso): 4,580 a 2000 e 3,999 a 4000, a 0,013 dal tetto del configuration model.

## Codice

Tutto in [`src/`](src/) (Python 3.12, PyTorch 2.14 + CUDA 12.6, cupy per il kernel fuso; una GTX 1080 da 8 GB basta). Indice dei file in [`src/README.md`](src/README.md). Protocolli, pre-registrazioni e resoconti in [`reports/`](reports/), configurazioni in [`configs/`](configs/).

Servono i dati, non inclusi: il connettoma male-cns:v1.0 (tre file feather da https://www.janelia.org/project-team/flyem/male-cns-connectome, CC-BY) in `dataset/male_cns/`, e TinyStories preparato con `prepare_text.py`. Il grafo a soglia 10 si costruisce al primo uso e va in cache.

Riproduzione di un controllo (esempio: la configurazione standard, 2000 update, curva ogni 500):

```bash
python pretrain_control.py --threshold 10 --head separate --seed 17 --eval-every 500 --checkpoint-every 2000 --fast-mode fused --rewire-kind relthr --rewire-seed 41 --ports random_matched --readout chunks --core-variant lif --updates 2000 --output results/standard_2000.json
```

## Cosa non è

Non è un LLM, non batte nessuno, non è efficiente. È il primo tentativo pubblico con il grafo male-cns intero come nucleo spiking, archi allenati sotto legge di Dale e senza backbone linguistico, con pretraining vero e controlli dichiarati.

## Crediti

Connettoma: male-cns:v1.0, Janelia FlyEM e Google Research (CC-BY). Testo: TinyStories. Ispirazioni dichiarate: Lappalainen et al. 2024 (reti vincolate dal connettoma), Shiu et al. 2024 (simulazione LIF del cervello intero), Pospisil et al. 2024 (effectome), SpikeGPT, Moonlight (Muon).

Codice, pagina e animazione: MIT. I dati del connettoma restano sotto la loro licenza.
