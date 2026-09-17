# Passaggio 0 — verifica della sessione precedente

Data: 12 settembre 2026. Oggetto: tre copypaste allegati, `bench_throughput.py`, `PIANO_OPERA_FACTCHECKED.md`, note e dati locali. Benchmark e piano originali invariati. Nessun training o benchmark completo GPU lanciato.

**Verdetto:** dati e ambiente basi reali, ma fattibilità e prestazioni dichiarate non dimostrate. Benchmark attuale contiene errori che impediscono usarlo per decisioni. Non dimostra che progetto o CNS intero impraticabili.

## 1. Errori dimostrati nel codice

### A. Il benchmark non produce spike né apprende

Righe 164–182: stati iniziali nulli, ingresso uniforme `[0, 0.08)`, aggiunto una volta per carattere, otto decadimenti fattore 0.95, soglia 1.

Finché nessuno spike, corrente ricorrente zero. Anche con ingresso massimo 0.08 SEMPRE, potenziale prima verifica soglia limitato da:

`0.95 * 0.08 / (1 - 0.95**8) = 0.2258009905 < 1`.

Quindi nessuno spike compare, per induzione, indipendente da grafo e pesi. Loss `sum(S.mean())` zero, gradiente ogni peso ricorrente zero. Surrogato trasmette gradienti rispetto stati, ma gradiente pesi contiene spike presinaptico, sempre nullo.

**Prova:** estrazione via AST classi e funzione originali, eseguite CPU con 5 neuroni, 7 archi, B=2, L=64, T=8, drive massimo. Loss=0, massimo gradiente assoluto pesi=0. Lavoro gather/scatter comunque eseguito: rete silenziosa non rende gratis benchmark, ma invalida verifica apprendimento.

### B. Il budget di memoria continua a essere sbagliato dopo il chunking

Funzione custom risolve salvataggio grandi espansioni per arco, non resto grafo autograd. Per ogni sottopasso salvati:

- spike booleani in `Propaga`: 1 byte per elemento;
- `V - soglia` nel surrogato: 4 byte;
- due operandi float reset differenziabile `V * (1 - S)`: 8 byte.

Totale misurato prova CPU: **13 byte per elemento e sottopasso**, non 5. Hook contano storage distinti, evitando contare 512 volte stesso vettore pesi.

Proiezione B=8, N=163743, L=64, T=8:

`13 * 8 * 163743 * 64 * 8 = 8.12 GiB`.

SOLO stati salvati: mancano drive (~0.31 GiB), pesi, indici, gradienti, Adam, temporanei, pool cattura. Su GTX 1080 8 GiB implementazione non rispetta budget. Con tutti 166700 nodi stessa componente sale a ~8.27 GiB. Proiezione da grafo autograd misurato CPU, non misura picco GPU.

Checkpointing, reset/autograd riformulati e gestione stati cambiano costo senza cambiare anatomia. Dire V debba conservarsi ogni sottopasso falso quando si ammette ricomputazione.

### C. Eager e CUDA graph non eseguono lo stesso aggiornamento

Eager azzera gradienti ogni passo. Prima cattura si usa `zero_grad(set_to_none=False)` UNA volta, fuori grafo; replay contiene backward e Adam, nessun azzeramento. Con gradienti già allocati si accumulano precedenti. Bug nascosto da rete silenziosa.

Documentazione PyTorch mostra strategia diversa, gradienti a `None` prima cattura e storage gestiti durante cattura. Serve confronto numerico fra più aggiornamenti, da stati iniziali identici, prima di confrontare tempi. [CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html).

### D. Il segno di Dale non è garantito

Indicizzazione segno per sorgente ora corretta. Ma `mag` parametro reale libero e `pesi = mag * segno_arco`: Adam può rendere negativa `mag`, invertendo segno effettivo. Occorre magnitudine non negativa, es. parametrizzazione positiva o proiezione esplicita.

### E. Non mantiene tutti i neuroni selezionati

Riga 51 costruisce nodi da `unique(pre, post)` DOPO soglia. Nodi senza archi superstiti scompaiono. Per rispettare CNS intero occorre insieme body ID fisso e soglia solo su archi; necessario anche per trasferire pesi e stato da soglia 10 a 5 senza cambiare indici arbitrariamente.

### F. È un microbenchmark del nucleo, non del language model

Mancano tokenizzazione, embedding, selezione ingressi sensoriali, readout, cross-entropy, validazione. Drive raggiunge tutti nodi e stati ripartono da zero ogni finestra. Esistono chiamate Adam, benché commento iniziale dica non allena nulla. Con ingressi correnti non aggiorna pesi perché gradienti zero.

Tempi futura versione corretta microbenchmark non saranno automaticamente tempi end-to-end TinyStories. Inoltre `migliore = car / (med_graph or med_eager)` sceglie sempre graph quando disponibile, anche se più lento.

## 2. Cosa era corretto sul problema delle espansioni

Con E=2753975 e B=32, float32 `[B,E]` pesa 352508800 byte, ~352.5 MB. Ripetuto e trattenuto 512 sottopassi ~180.5 GB decimali. Problema formulazione ingenua reale.

Versione custom conserva spike booleani, usa temporanei `[B,chunk]`. Confrontati forward e gradienti con riferimento gather/scatter su piccolo grafo con sorgenti e destinazioni ripetute, input binari, double precision: errore massimo zero. Riscontro positivo limitato a propagazione primo ordine; non certifica CUDA né intero training. Per input non binari salvataggio come bool altererebbe gradiente pesi.

Ma false generalizzazioni «qualsiasi batch», «nessuna GPU», «per forza a blocchi»: memoria e fattibilità dipendono da B, L, T, ricomputazione, implementazione. `torch.sparse.mm` documenta backward per COO/CSR × dense, ma supporto API non equivale a prova efficienza, memoria o catturabilità su build locale. [Documentazione sparse.mm](https://docs.pytorch.org/docs/main/generated/torch.sparse.mm.html).

## 3. Prestazioni: quali conclusioni ritirare

- **«Dividere per batch 32 corregge tempi di 32 volte»: non giustificato.** Riuso pesi ammortizzabile; gather, scatter, attivazioni e parte traffico crescono con B. Script effettua esplicitamente operazioni su B×E elementi. Fattore esatto non si deduce dal batch.
- **«50 volte più leggero di Qwen, quindi non può essere più lento»: falso ragionamento.** E×T conta contributi sinaptici; un multiply-add vale normalmente due FLOP. Backward, stati, atomiche aggiungono costo. Caratteri e token subword unità diverse, e ricorrenza seriale non sfrutta GPU come matmul. Confronto con CPT richiede batch, lunghezza, token totali, parametri allenati, kernel effettivi. CPT non implica né esclude LoRA.
- **«CUDA graphs daranno almeno +40%»: non dimostrato.** Tua PR documenta miglioramenti veri nei carichi provati. Non offre limite inferiore trasferibile a training spiking: contano durata kernel, dipendenze, banda, contesa, oltre numero nodi. Anche prefill nella PR cambia pochissimo. [PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721).
- **«13348 test passati certificano questo benchmark»: falso.** Test llama.cpp; non verificano autograd custom, surrogate gradient o Adam PyTorch.
- **«FP16 inutile»: troppo assoluto.** Su GP104 aritmetica FP16 lenta (1/64 di FP32), ma storage ridotto e conversioni con calcolo FP32 possono ancora avere utilità da misurare. [NVIDIA Pascal Tuning Guide](https://docs.nvidia.com/cuda/archive/11.3.0/pascal-tuning-guide/index.html).
- Percentuali fisse utilizzo banda sparse/dense ed epoche in ore/giorni non sono misure locali. Messaggio OOM con numero enorme non basta per diagnosticare bug WDDM: manca log completo.

**Non esiste ancora throughput validato per modello richiesto.** Non dimostrato né «ci vogliono mesi» né «macchina ce la fa comodamente».

## 4. Ambiente e fonti: riscontri positivi e limiti

- Locale: `.venv/Scripts/python.exe`, torch **2.14.0+cu126**, runtime **12.6**, CUDA disponibile, `sm_61` nella lista compilata. GPU rilevata: GTX 1080, 8192 MiB, driver 581.57. Test verifica disponibilità e metadati, non esegue training GPU.
- Annuncio ufficiale conferma 2.14 ultima release con wheel precompilate per Maxwell/Pascal/Volta; versione NON inventata. [PyTorch packaging notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432).
- JAX: niente NVIDIA GPU su Windows nativo; WSL2 sperimentale. Motivo valido per preferire PyTorch qui, non impossibilità generale usare JAX sulla macchina. [JAX installation](https://docs.jax.dev/en/latest/installation.html).
- Triton upstream attuale richiede NVIDIA compute capability 8.0+. Via standard Inductor/Triton non soluzione per Pascal; non elimina kernel CUDA C++ o altre ottimizzazioni. [Triton README](https://github.com/triton-lang/triton).
- Progetto MaleCNS, download e licenza CC-BY reali. Homepage distingue **rilascio v1.0: 8 giugno 2026** e **pubblicazione paper: 3 settembre 2026**: piano confonde le due date. [MaleCNS](https://male-cns.janelia.org/).
- Pagina download oggi punta a nomi `*-minconf-0.5.feather`, mentre due file locali senza quel suffisso. Conteggi locali non certificano da soli identità/versione download: manca manifest con URL e checksum. [Download ufficiale](https://male-cns.janelia.org/download/).
- BrainTrace esiste: paper riporta 8.9 GB per pp-prop, oltre 32 GB per BPTT/D-RTRL nella SUA configurazione, 17 minuti di segnali e valutazione su 68 regioni. Dati di compito dinamica cerebrale, non misura nostro language model. Repository espone adattamento low-rank (`n_rank`). [Paper](https://www.nature.com/articles/s41467-026-68453-w_reference.pdf), [codice](https://github.com/chaobrain/fitting_drosophila_whole_brain_spiking_model).
- SpikeGPT usa PyTorch/SpikingJelly e struttura RWKV; paper consultato riporta **45M**, non 46M, e 216M parametri, quattro V100, 12/48 ore. Risultati non predicono automaticamente grafo LIF senza suoi meccanismi memoria. [SpikeGPT](https://arxiv.org/html/2302.13939v4).
- Non trovato `connettoma-parlante.html` nella cartella progetto né URL animazione nei testi allegati. Pubblicazione, fedeltà anatomica e curve mostrate non verificabili da questi materiali.
- Isolamento venv visibile, ma senza confronto precedente non posso certificare storicamente che nessun pacchetto globale toccato. Nessun risultato completo throughput tra file progetto esaminati.

## 5. Affermazioni scientifiche da correggere

**Connettoma e attention non incompatibili per teorema.** Nucleo strettamente vincolato agli archi biologici e modulo attention esterno sono scelte diverse ma combinabili. SpikeGPT sceglie RWKV e discute anche SNN con attention; non prova divieto universale. Rimuovere attention decisione progettuale da motivare, non conclusione obbligata della letteratura.

**T=8 è iperparametro.** Determina quanti aggiornamenti per carattere, non imposto dall'anatomia. Con stati persistenti anche caratteri precedenti influenzano successivi. «Un solo passo non può calcolare qualcosa di complesso» generalizzazione non sostenuta.

**42% completamento postsinaptico non significa che archi da una sinapsi siano rumore.** Incompletezza può far apparire debole connessione reale. Soglia è filtro sperimentale, non classificazione automatica della verità biologica.

**10→5 progressione valida, non da sola ablazione pulita.** Più training e più archi cambiano insieme. Per attribuire eventuale miglioramento agli archi aggiunti serve almeno controllo che continui a soglia 10 per stesso budget. Mancato miglioramento non dimostra archi deboli rumore.

**Neurotrasmettitore e segno funzionale non coincidono universalmente.** Mappa scelta è semplificazione; glutammato può essere eccitatorio o inibitorio, pur spesso inibitorio nel cervello mosca. Modulatori e sconosciuti non sono eccitatori accertati. Segno fisso non dimezza numero parametri continui. [Network statistics](https://www.nature.com/articles/s41586-024-07968-y).

**Escludere lobi ottici e VNC perché «non servono al linguaggio» non è risultato scientifico.** Mosca non possiede circuito linguistico noto; qui si riaddestra topologia. Ritaglio cambia esperimento e contraddice scelta esplicita del CNS intero.

**Aspettative sul testo restano ipotesi.** Da parametri e neuroni non segue né che produrrà frasi coerenti né che impossibile. Servono training e validazione. Confronto con grafo rimescolato misura vantaggio nella configurazione testata: parità risultato non dimostra «la biologia non contribuisce nulla» in assoluto.

Piano confonde inoltre MB testo e token: 20–50 MB non diventano 5–12 milioni di CARATTERI dividendo per quattro. Rapporto può approssimare token subword, non caratteri; byte UTF-8 e caratteri vanno contati separatamente.

## 6. Prove riproducibili e stato di ripartenza

### Riconteggio completato sui file locali

Lettura integrale 151856684 righe in 2318 record batch, ~146 secondi; nessun caricamento integrale grafo in RAM. Processo terminato normalmente. RAM disponibile tornata da 3.69 a 3.60 GiB a fine scansione; controllo GPU successivo: 754 MiB occupati contro 765 MiB iniziali (oscillazioni desktop comprese).

| Dato | Riconteggio | Esito rispetto alla chat |
|---|---:|---|
| Segmenti annotati | 211577 | confermato |
| Corpi con `superclass` non nulla | 166700 | confermato come criterio operativo |
| Archi fra questi corpi, soglia 1 | 25582938 | confermato |
| Somma sinapsi, soglia 1 | 124177617 | confermato |
| Archi soglia 2 | 15283237 | confermato |
| Archi soglia 3 | 10520431 | confermato |
| Archi soglia 5 | 6242118 | confermato |
| Archi soglia 10 | 2753975 | confermato |
| Nodi incidenti agli archi, soglia 5 | 165836 | 864 dei 166700 esclusi dal caricatore |
| Nodi incidenti agli archi, soglia 10 | 163743 | 2957 dei 166700 esclusi dal caricatore |
| Selezione centrale del vecchio piano | 38453 | conteggio confermato, scelta scartata dall'utente |

Anche i cinque conteggi archi selezione centrale riprodotti nel JSON. Condizione `superclass` presente non è di per sé certificazione che ogni corpo sia neurone completamente revisionato: presenti etichette `*_tbc`. Non equiparare automaticamente 166700 selezionati al conteggio editoriale di 166691 neuroni revisionati.

**Neurotrasmettitori: errore popolazione di riferimento.** Numeri 104193 acetilcolina e 59663 GABA+glutammato+istamina riproducibili sull'intera tabella NT, prima del filtro sui body ID. Sui 166700 corpi effettivamente selezionati:

| Categoria | Conteggio |
|---|---:|
| Acetilcolina | 103720 |
| GABA | 22069 |
| Glutammato | 29302 |
| Istamina | 7891 |
| Somma delle tre categorie trattate come inibitorie | 59262 |
| Dopamina / octopamina / serotonina | 392 / 101 / 48 |
| `unclear` | 2999 |
| Nessuna riga NT corrispondente | 178 |

Copertura con etichetta diversa da `unclear`: **163523 / 166700 = 98.09%**, non 98.6%. Presenza riga nella tabella e assegnazione informativa cose diverse. Conteggi restano predizioni NT, non misure dirette del segno funzionale di ciascuna sinapsi.

File TinyStories presenti: train **2227753162 byte**, valid **22502601 byte**. File connettoma presenti: annotazioni **14483314 byte**, NT **43282834 byte**, pesi **1051241946 byte**. Verificata presenza e dimensione, non completezza semantica dell'intero testo né checksum rispetto all'origine.

`audit_step0.py toy` esegue prova CPU sul codice estratto dall'originale. `audit_step0.py data` legge grafo a record batch, con guardia sulla RAM disponibile, senza caricarlo integralmente e senza modificarlo. Risultati in `audit_step0_toy.json` e `audit_step0_data.json`.

SHA256 del benchmark verificato: `20E03132FC204ED3E14A5675B4EB83EC688ED94E6ABEBA04279C735C62A4C01E`.

SHA256 del piano verificato: `3A673110025EDB0FC6E918AEDEB75F66491BE8F985786D3B0AB66CF6185AFCC9`.

Restano vincoli ricevuti: **CNS intero**, percorso **soglia 10→5**, misura prima di scegliere volume testo. Ridurre dimensione test diagnostico non modifica dimensione concordata del modello. Prima del benchmark reale correggere attività, memoria, segni, identità nodi, equivalenza eager/graph; poi misurare con limiti RAM/VRAM e timeout. Nessuna nuova riduzione anatomica decisa in questo audit.
