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
6. **Addestramento**: 4 + 4 sottopassi per token, TBPTT 8 token, Adam (nel candidato per il default: Muon 1e-3 sulle matrici dense di attention e lettura, Adam sul resto), CUDA Graph FP32, 0,65 s per update su una GTX 1080 da 8 GB (0,38 s col kernel fuso). 5,46 milioni di parametri, quasi tutti sinapsi.

## Cosa abbiamo misurato finora

| | CE su 16 storie riservate (nat, più basso = meglio) |
|---|---|
| Mosca, cablaggio reale, porte sensoriali, 26.247 update | **3,634** |
| Mosca, cablaggio reale, porte sensoriali, 8000 update | 3,959 |
| Mosca, cablaggio reale intero, porte random, 8000 update | 3,689 |
| Mosca, configurazione standard (soglia relativa + porte random + lettura a blocchi), 8000 update | 3,652 |
| Standard + sinapsi a conduttanza, 8000 update | 3,624 |
| Standard + 8+8 sottopassi, 8000 update | 3,592 |
| Standard + Muon 3e-4 sulle matrici dense, 8000 update | 3,571 |
| **Standard + Muon 1e-3 (candidato per il default), 8000 update** | **3,541** |
| Mosca, cablaggio rimescolato a gradi conservati, 8000 update | 3,814 |
| Configuration model (topologia distrutta), 8000 update | 3,648 |
| GRU 2×256, stessi dati | 3,429 |
| Transformer 2 blocchi, stessi dati | 3,563 |
| Trigramma Kneser-Ney, stessi dati | 3,410 |

Generazione a 8000 update, prompt "Once upon a time": *", there was a little girl named Tim. She loved to play with the ball. The dog was very happy and happy."*

## Test eseguiti (settembre 2026)

Ogni test è pre-registrato (soglia di effetto 0,10 nat, pavimento di rumore misurato 0,01-0,02 nat tra run e tra semi), eseguito con la stessa pipeline e la stessa inizializzazione, e tracciato con curva ogni 500 update, verdetto e considerazioni in [`reports/SUITE_TEST_FASE_6B.md`](reports/SUITE_TEST_FASE_6B.md). Riferimento: mosca reale a 2000 update = 5,087 (a 8000 = 3,959).

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
- Raggio spettrale dei pesi con segno: 0,500 all'init per costruzione (pesi entranti normalizzati a 0,5) su grafo reale e a soglia relativa, 0,20-0,23 nei null (l'inibizione cancella il modo dominante), e **identico alla quarta cifra su tutti i checkpoint allenati**; 0,405 sui soli neuroni attivi. **Errata**: i valori pubblicati prima ("0,73 → 0,35 col training") erano fasi diverse di un'iterazione di potenza che oscilla con periodo 2, non un cambiamento della dinamica. Script corretto.
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
- Costanti di tempo per tipo cellulare (11.752 leak allenabili): nessun effetto a lr 1e-4 (non si muovono), +0,06 a lr 1e-3, +0,12 a 3e-3, **+0,17 a 1e-2** (leak imparati 0,37-0,996: il modello vuole costanti di tempo molto diverse per tipo). La leva era limitata dal passo, non dal segnale.
- Bias di riposo per tipo cellulare: il −0,46 a lr 1e-3 era fame di gradiente (il gradiente del bias dominava il clip globale e affamava gli altri parametri); riparametrizzato è neutro (+0,03, bias imparati ≤ 0,04): scartato.
- Sinapsi a conduttanza (potenziali di inversione E_e = 3, E_i = −1): +0,04 su porte anatomiche, **+0,12 sulla baseline standard**. Potenziali limitati (minimo −1,3 invece di −6,3), stessa attività: gli hub sommersi d'inibizione tornano a portata di soglia.
- Scala dei pesi iniziali ×0,5: +0,07 (stesso meccanismo, in versione grezza).
- Più sottopassi per token (8+8 invece di 4+4): **+0,18** a 2000 con Adam, ma +0,06 a 8000 e +0,09 sopra Muon 1e-3; 12+12: +0,11 sul default, +0,025 su 8+8 (satura oltre il diametro del grafo, 13). Costo ×1,8 / ×2,5: a pari tempo GPU la profondità perde. La conduttanza a 8000 vale +0,03. Leve di dinamica = acceleratori, non alza-tetto.
- Combinazioni a 2000: 8+8 + conduttanza 4,350 (additività 76%); conduttanza + tau 4,382; tutte insieme (8+8 + conduttanza + tau 1e-2 + Muon 1e-3) **4,152**, il miglior valore a 2000 (replica con altro seed 4,173: rumore fra semi 0,02), ma 3,5 volte più lenta: nello stesso tempo il default semplice fa 7000 update e arriva a 3,615.

**Ottimizzatore: Muon sulle matrici dense (attention q/k/v/o e proiezione di lettura, 301.568 parametri; Adam sul resto)**
- Curva del lr a 2000 update sulla configurazione standard (4,580): 1e-4 4,648 (−0,07), 3e-4 4,411 (+0,17), **1e-3 4,276 (+0,30)**, 3e-3 4,250 (plateau). Costo zero (380 contro 365 ms/update). Muon anche sulla testa di uscita 4096×256: −0,33 (la testa resta su Adam).
- A 8000 update: 3e-4 3,571 (+0,08), **1e-3 3,541 (+0,11)**: l'unica leva che resta sopra la soglia a 8000, e la prima configurazione della mosca sotto il Transformer piccolo a pari token (3,563). Avvertenza: quel Transformer è allenato con Adam; non è un sorpasso a pari ottimizzatore.
- Configurazione standard da sola a 8000: 3,652, pari al configuration model (3,648): le scelte fedeli al grafo (soglia relativa, lettura a blocchi) più le porte random arrivano al tetto dei null.

**Quanto cervello impara (17 settembre)**
- **I pesi sinaptici quasi non si muovono**: ‖w − w₀‖/‖w₀‖ = 0,11% a 1000 update, 0,21% a 4000, 0,30% a 8000 (0,26% con Muon, 0,32% con 8+8); solo il 2,9% degli archi si sposta più di 1e-4. Dopo 8000 update il nucleo è di fatto un reservoir fisso: l'apprendimento sta nelle interfacce (embedding, porte, lettura, attention, testa). Cause: il gradiente di un peso arriva solo se il neurone a monte spara (81% muti, 19% degli archi parte da sorgenti attive) e la parametrizzazione softplus con pesi ≈ 0,03 rende il passo effettivo ≈ lr × 0,03. Spiega perché i pesi permutati non cambiavano nulla e perché porte e lettura contano più del cablaggio. Conseguenza: i confronti reale contro null fatti finora confrontavano reservoir.
- Raggiungibilità con segno (init): da porte random il segnale raggiunge l'85% dei neuroni di lettura in un salto; dagli occhi in 2-3 salti con segni che si cancellano (coerenza 0,07).
- Porte anatomiche a cammino corto dentro la configurazione migliore (8+8 + Muon 1e-3): −0,145 rispetto alle porte random (era −0,279 con Adam a 4+4): gli ingressi veri costano ancora, la metà.

**In coda (test corti con codice nuovo, pre-registrati nella suite, sezioni L-M)**
- Micro-sweep del passo sinaptico (run da 300 update: lr dedicato dei pesi 1e-3 … 3e-2, con gradiente normale e "allargato" ai presinaptici vicini alla soglia), poi conferme a 2000; sinapsi rimesse all'init su un checkpoint allenato (quanto lavoro fa l'apprendimento sinaptico).
- **Omeostasi di risveglio per tipo cellulare**: ogni tipo abbassa da solo la propria soglia quando sta sotto il 2% di attività (senza gradiente); **impulsi di arousal** (eccitabilità globale modulata nel tempo, a impulsi o a onda, analogo dell'octopamina in volo); **shock di profondità** (12+12 → 8+8 → 4+4 → ritorno a 12+12 con un solo CUDA Graph).
- **Attention dentro il giro del cervello** (logica dei looped transformer: letture ripetute con gli stessi pesi, cache per lettura, identificativo di passo, input injection), ricerca in [`reports/RICERCA_LOOPED_A.md`](reports/RICERCA_LOOPED_A.md) e [`reports/RICERCA_LOOPED_B.md`](reports/RICERCA_LOOPED_B.md).
- Controlli: nucleo identità (archi spenti), reservoir (archi congelati), seed e numero delle porte, fattoriale grafo reale/rimescolato × attention accesa/spenta.
- Elenco completo con stato e risultati: [`reports/PIANO_TEST_RIMASTI.md`](reports/PIANO_TEST_RIMASTI.md).

**Velocità (senza cambiare la matematica)**
- Kernel CUDA fuso (cupy: gather-moltiplica-accumula in una passata, salta gli spike a zero): ×1,73, equivalente al riferimento a 100 e 2000 update (loss 1e-6, gradiente 1e-8). Adottato per i controlli.
- Riordino dei nodi per (superclasse, grado): ×2,4 in tempo ma non equivalente (gruppi di lettura e porte definiti sull'indice dei nodi): non adottato.
- Chunk ×1,22; CSR cuSPARSE ×0,60; FP16 storage ×1,07; INT32 e Adam fused: niente. Micro-benchmark: gli atomici dell'accumulo dominano, non la banda dei pesi.
- Pausa vera dei run (sospensione dei processi) e 3 minuti di riposo GPU ogni ~1,5 ore di attività.

**Ricerca bibliografica (tre agenti, ~40 fonti)**: l'86% di neuroni muti è normale per i LIF whole-brain pubblicati (Shiu 2024: 0,03-0,4% attivi); le leve pubblicate per far parlare più cervello sono per tipo cellulare (flyvis: 734 parametri, potenziale di riposo e τ per tipo); occhio composto → discendente ≈ 4 sinapsi; il conteggio di sinapsi è un proxy grezzo dell'efficacia (EPSP da 0,4 a 6,6 mV). Dettagli: [`reports/RICERCA_ATTIVITA_CERVELLO_MOSCA.md`](reports/RICERCA_ATTIVITA_CERVELLO_MOSCA.md), [`reports/RICERCA_CONNETTOMA_LM.md`](reports/RICERCA_CONNETTOMA_LM.md).

**Configurazione corrente**: standard in [`reports/STANDARD_MOSCA_6B.md`](reports/STANDARD_MOSCA_6B.md) (grafo intero con soglia relativa, segni dal neurotrasmettitore, porte random, lettura discendenti a blocchi, LIF 4+4, kernel fuso) più il candidato per il default: Muon 1e-3 sulle matrici dense.

## Codice

Tutto in [`src/`](src/) (Python 3.12, PyTorch 2.14 + CUDA 12.6, cupy per il kernel fuso; una GTX 1080 da 8 GB basta). Indice dei file in [`src/README.md`](src/README.md). Protocolli, pre-registrazioni e resoconti in [`reports/`](reports/), configurazioni in [`configs/`](configs/).

Servono i dati, non inclusi: il connettoma male-cns:v1.0 (tre file feather da https://www.janelia.org/project-team/flyem/male-cns-connectome, CC-BY) in `dataset/male_cns/`, e TinyStories preparato con `prepare_text.py`. Il grafo a soglia 10 si costruisce al primo uso e va in cache.

Riproduzione di un controllo (esempio: il candidato per il default, 2000 update, curva ogni 500; self-test CPU delle varianti: `python phase8_selftest.py`):

```bash
python pretrain_control.py --threshold 10 --head separate --seed 17 --eval-every 500 --checkpoint-every 2000 --fast-mode fused --rewire-kind relthr --rewire-seed 41 --ports random_matched --readout chunks --core-variant lif --optimizer muon --muon-lr 1e-3 --updates 2000 --output results/default_2000.json
```

## Cosa non è

Non è un LLM, non batte nessuno, non è efficiente. È il primo tentativo pubblico con il grafo male-cns intero come nucleo spiking, archi allenati sotto legge di Dale e senza backbone linguistico, con pretraining vero e controlli dichiarati.

## Crediti

Connettoma: male-cns:v1.0, Janelia FlyEM e Google Research (CC-BY). Testo: TinyStories. Ispirazioni dichiarate: Lappalainen et al. 2024 (reti vincolate dal connettoma), Shiu et al. 2024 (simulazione LIF del cervello intero), Pospisil et al. 2024 (effectome), SpikeGPT, Moonlight (Muon).

Codice, pagina e animazione: MIT. I dati del connettoma restano sotto la loro licenza.
