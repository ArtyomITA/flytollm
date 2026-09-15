# FlyToLLM · il connettoma parlante

**Il sistema nervoso completo di un moscerino maschio, usato intero come nucleo di un language model.**
166.700 neuroni, 2.753.975 connessioni reali, segni sinaptici presi dal neurotrasmettitore. Nessun taglio anatomico, nessun backbone pre-addestrato. Una sola GTX 1080.

▶ **Animazione della pipeline:** https://artyomita.github.io/flytollm/

*TL;DR (English): the whole male Drosophila CNS connectome (male-cns:v1.0, 166,700 neurons) is used, uncut, as the recurrent spiking core of a next-token language model on TinyStories. Synapse counts initialise the weights, neurotransmitters fix the signs (Dale's law), a small external attention re-enters through the real sensory neurons. Trained with surrogate gradients on one GTX 1080. Honest controls included: the rewired graph currently does better than the real wiring.*

## Lo scopo

- Far parlare il cervello della mosca: non un modello "ispirato" alla mosca, ma il cablaggio misurato, tutto, con i suoi segni.
- Usare quanto più possibile della natura del connettoma: topologia, conteggi di sinapsi, neurotrasmettitori, neuroni sensoriali come porte d'ingresso, neuroni discendenti e motori come lettura.
- Scoprire cosa fa davvero il cablaggio quando lo si costringe a un compito che non è suo, con controlli che possono dire di no.

Non è una gara di efficienza contro GRU e Transformer. È un esperimento per capire.

## Come funziona

1. **Connettoma** male-cns:v1.0 (Janelia FlyEM · Google Research, Cell, settembre 2026, CC-BY): 166.700 neuroni, 25,6 M connessioni; entrano quelle con almeno 10 sinapsi.
2. **Neuroni LIF** (leaky integrate-and-fire) su tutto il grafo, spike binari, gradiente surrogato. Segno fisso dal neurotrasmettitore, magnitudine inizializzata dal conteggio di sinapsi e allenata.
3. **Il testo entra dai sensi**: ogni token BPE (vocabolario 4096) diventa un vettore di 256 dimensioni iniettato come corrente nei 17.937 neuroni sensoriali reali.
4. **Attention fuori dal grafo**: una piccola attention causale (4 teste, cache di 128 token) legge lo stato e rientra dalle stesse porte, con un gate. Il connettoma non ha un asse "posizione nella frase", quindi la memoria di sequenza sta fuori.
5. **La risposta esce dai muscoli**: potenziale e tasso di spike dei 2.241 neuroni discendenti, motori ed efferenti, raggruppati in 77 popolazioni, poi softmax sui 4096 token.
6. **Addestramento**: 4 + 4 sottopassi per token, TBPTT 8 token, Adam, CUDA Graph FP32, 0,65 s per update su una GTX 1080 da 8 GB. 5,46 milioni di parametri, quasi tutti sinapsi.

## Cosa abbiamo misurato finora

| | CE su 16 storie riservate (nat, più basso = meglio) |
|---|---|
| Mosca, cablaggio reale, 26.247 update | **3,634** |
| Mosca, cablaggio reale, 8000 update | 3,959 |
| Mosca, cablaggio rimescolato a gradi conservati, 8000 update | 3,814 |
| GRU 2×256, stessi dati | 3,429 |
| Transformer 2 blocchi, stessi dati | 3,563 |
| Trigramma Kneser-Ney, stessi dati | 3,410 |

Generazione a 8000 update, prompt "Once upon a time": *", there was a little girl named Tim. The girl was a big dog named. The dog was very happy."*

Letture oneste, per ora:
- Il cablaggio anatomico non aiuta: il grafo rimescolato fa meglio del reale a pari dati e a pari init, su due seed di rimescolamento. Più il grafo è casuale, meglio va.
- I neuroni di lettura stanno a 2 salti sinaptici dalle porte: la profondità 4 + 4 basta per costruzione.
- Il lobo ottico intero, 89.403 neuroni, si può rimuovere senza cambiare la cross-entropy: col testo in ingresso non viene usato.
- L'attention esterna vale 0,24 nat e tutto il suo contributo viene dagli ultimi 8 token.
- Il modello ha visto 839.904 token: meno dell'1% di quanto una legge di scala vorrebbe per 5,46 M parametri. Ogni confronto vale in regime di dati scarsi.

## Cosa non è

Non è un LLM, non batte nessuno, non è efficiente. È il primo tentativo pubblico con il grafo male-cns intero come nucleo spiking, archi allenati sotto legge di Dale e senza backbone linguistico, con pretraining vero e controlli dichiarati.

## Stato

Settembre 2026: pretraining fermo a 26.247 update; suite di controlli e sonde in corso (null model del grafo, lesioni per popolazione, sonde sullo stato, varianti "naturali" del nucleo: parametri per tipo cellulare, sparsità del corpo fungiforme, soglia sinaptica relativa). Codice, log e checkpoint verranno pubblicati insieme al resoconto.

## Crediti

Connettoma: male-cns:v1.0, Janelia FlyEM e Google Research (CC-BY). Testo: TinyStories. Ispirazioni dichiarate: Lappalainen et al. 2024 (reti vincolate dal connettoma), Shiu et al. 2024 (simulazione LIF del cervello intero), SpikeGPT.

Pagina e animazione: MIT. I dati del connettoma restano sotto la loro licenza.
