# FlyToLLM — Piano di opera verificato

> **STORICO — NON VIGENTE (12 settembre 2026).** Contiene errori verificati e scelte anatomiche rifiutate da utente. Leggere `VERIFICA_PASSAGGIO_0.md` e `BENCHMARK_STATO.md` prima di implementazione. Vincoli attuali: CNS intero, soglia 10→5, quantità testo da decidere dopo misure. Testo seguente conservato come traccia sessione precedente.

Documento riferimento per implementazione. Ogni affermazione qui verificata contro fonti primarie o misurata direttamente su macchina. Correzioni rispetto a piano precedente (`ricerca_spikegpt/INDICE_RICERCA.md`) segnalate: dove i due divergono, vale questo.

Obiettivo: prendere connettoma reale del sistema nervoso centrale del moscerino maschio (rilascio Janelia/Google 3 settembre 2026) e usarlo come substrato di rete neurale spiking allenata a predire token successivo, così genera testo. Scopo: non efficienza energetica, ma far "parlare" cervello mosca.

---

## 1. Verdetto sintetico

Progetto fattibile, ma **non in forma descritta in piano precedente**. Tre problemi seri da verifica, tutti con soluzione.

- Stack software raccomandato prima (BrainTrace/BrainPy, su JAX) **non può usare GPU su questa macchina**: JAX no supporto GPU nativo su Windows. Sostituito con PyTorch + SpikingJelly.
- Architettura descritta prima internamente contraddittoria: non puoi avere insieme "connettoma è la rete" e "dentro c'è attention". Scegli una. Letteratura ha risposto: scegli connettoma, attention sparisce.
- Scala descritta prima (connettoma intero, TinyStories intero) richiede mesi calcolo su questo hardware. Va ridotta, c'è modo naturale e scientificamente sensato.

Con queste tre correzioni progetto diventa esperimento eseguibile in giorni, non mesi, su hardware esistente.

---

## 2. Hardware reale della macchina

Misurato, non stimato.

| Componente | Valore | Implicazione |
|---|---|---|
| GPU | NVIDIA GeForce GTX 1080, 8192 MiB | ~7,2 GB utilizzabili (973 MiB già occupati da desktop). Pascal, compute capability 6.1, niente Tensor Core, niente bf16 |
| Driver | 581.57, CUDA 13.0 capable | Driver va bene; problema è toolkit con cui compilate librerie |
| RAM sistema | 13,9 GB | Vincolo serio: file TinyStories da 2,23 GB non si carica e tokenizza ingenuamente in memoria |
| CPU | AMD Ryzen 3 4300GE, 4 core / 8 thread | Debole. Caricamento e tokenizzazione dati saranno collo di bottiglia se fatti male |
| Disco E: | 50,9 GB liberi | Basta per file essenziali connettoma (~1,1 GB), non per volumi immagini EM |
| Python | 3.12.6 | Compatibile con wheel necessarie |
| PyTorch installato | 2.13.0+**cpu** | Build senza CUDA: GPU non usabile ora. Va sostituito |

Pacchetti già presenti e utilizzabili: pyarrow 20.0.0, pandas 2.3.3, numpy 2.0.2, scipy 1.18.0, networkx 3.4.2.

---

## 3. Fact-check del piano precedente, punto per punto

### 3.1 Framework di training — CORREZIONE SOSTANZIALE

Piano precedente raccomandava BrainTrace/pp-prop come unica scelta, perché testato esattamente su connettoma FlyWire alla nostra scala.

Verificato: lavoro esiste ed è reale. Pubblicato su Nature Communications 19 gennaio 2026, titolo "Model-agnostic linear-memory online learning in spiking neural networks", codice parte Drosophila su `github.com/chaobrain/fitting_drosophila_whole_brain_spiking_model`. Algoritmo a memoria lineare si chiama ES-D-RTRL in preprint (pp-prop in versione finale).

Ma due problemi che piano precedente non aveva rilevato.

**Problema bloccante: JAX non supporta GPU su Windows nativo.** BrainScale/BrainTrace costruiti su ecosistema BrainPy, gira su JAX/XLA. Documentazione ufficiale JAX indica supporto GPU limitato a Linux; su Windows solo CPU nativo, GPU richiede WSL2 con supporto dichiarato sperimentale. Installando `brainpy[cuda12]` qui si otterrebbe fallback su CPU. Alternative: usare WSL2 (complessità aggiuntiva, e su 13,9 GB RAM totale memoria condivisa diventa problema) oppure cambiare stack.

**Problema di scala: configurazione pubblicata non entra comunque in questa GPU.** Paper riporta modello whole-brain consuma circa 8,9 GB memoria GPU con ES-D-RTRL, mentre con D-RTRL e BPTT supera capacità di GPU da 32 GB. 8,9 GB più dei ~7,2 GB disponibili qui.

**Inoltre loro risultato più modesto di come riportato.** Non hanno allenato tutte sinapsi su compito cognitivo: hanno trattato connettoma come impalcatura architetturale fissa e allenato una **correzione peso a rango basso**, su 17 minuti di segnali calcium imaging aggregati in 68 regioni cerebrali, obiettivo predire firing rate al passo successivo minimizzando errore quadratico medio. Compito molto più semplice di predizione token successivo su testo.

**Correzione:** stack riferimento diventa **PyTorch + SpikingJelly**, esattamente quello con cui costruito SpikeGPT (paper lo dichiara), funziona nativamente su Windows, supporta Pascal.

### 3.2 Versione di PyTorch — CORREZIONE

Piano precedente diceva fissare PyTorch a ≤ 2.7. Sbagliato e troppo conservativo: confondeva abbandono di Pascal in build CUDA 12.8/12.9 con abbandono generale.

Verificato su indice wheel ufficiali: build **cu126** per Windows e Python 3.12 esistono fino a torch 2.14.0. Annuncio ufficiale PyTorch afferma 2.14 è ultima release a fornire binari precompilati per Maxwell, Pascal e Volta, e da 2.15 wheel CUDA 12.x non più pubblicate.

Quindi versione giusta è **torch 2.14.0+cu126**, non 2.7. Comando verificato:

```
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
```

Contesto confermato: CUDA 13 ha rimosso supporto compilazione per Maxwell, Pascal e Volta; CUDA 12.x resta ultima linea capace di generare codice per queste architetture. Stessa logica per JAX: su build CUDA 12 supporta SM 5.2 e superiori, su build CUDA 13 richiede SM 7.5 e superiori.

### 3.3 "BPTT è impraticabile" — PARZIALMENTE SBAGLIATO

Piano precedente concludeva BPTT è morto e serve per forza algoritmo online a memoria lineare. Vero **solo alla scala del cervello intero con sequenze lunghe**. Alla scala ridotta proposta sotto (cervello centrale, ~38k neuroni, troncamento a 64 token) BPTT troncato entra tranquillamente nei 7,2 GB. Calcolo memoria in sezione 6.

Importante perché BPTT troncato in PyTorch banale da implementare, mentre apprendimento online forward-mode richiede framework specializzato che qui non usabile in GPU.

### 3.4 Conteggi dei neuroni — DATI SOSTITUITI

Numeri di piano precedente (sensory 5.495, descending 1.303, motor 100, ecc.) venivano da connettoma FlyWire, che è **femmina**. Noi useremo connettoma maschile. Numeri reali, misurati su file annotazioni scaricato, in sezione 5.

### 3.5 Legge di Dale — MANCAVA DEL TUTTO

Piano precedente non affrontava decisione scientifica necessaria: in biologia un neurone rilascia stesso neurotrasmettitore su tutte sue sinapsi in uscita, quindi **segno** (eccitatorio o inibitorio) è proprietà del neurone, non della singola connessione. Se lasci discesa del gradiente libera di cambiare segno ai pesi, rompi questo vincolo e modello smette di essere biologicamente sensato.

Verificato dato necessario esiste ed è quasi completo: file neurotrasmettitori assegna predizione a circa 164.400 corpi, praticamente tutti i 166.691 neuroni revisionati. Distribuzione: acetilcolina 104.193, glutammato 29.443, GABA 22.196, istamina 8.024, dopamina 396, octopamina 101, serotonina 48.

**Decisione raccomandata:** segno fisso da neurotrasmettitore (acetilcolina eccitatoria; GABA, glutammato e istamina inibitori nel moscerino), solo magnitudine peso allenabile. Mantiene vincolo biologico e dimezza spazio di ricerca.

### 3.6 Stima dei tempi — ERA SBAGLIATA IN DIFETTO

Piano precedente diceva "giorni non ore". Alla scala piena (connettoma intero, TinyStories intero) calcolo reale dà mesi per singola epoca, cioè infattibile. Alla scala ridotta dà ore per epoca. Differenza non è dettaglio: è differenza tra progetto eseguibile e progetto impossibile. Numeri in sezione 6.

---

## 4. Il problema architettonico centrale, e come si risolve

Punto più importante del documento.

Piano precedente elencava, uno accanto all'altro, "topologia: grafo reale del connettoma, fissa" e "attenzione: WTA-style". **Le due cose incompatibili così come scritte**, e contraddizione va sciolta prima di scrivere una riga codice.

Motivo: i due oggetti vivono su assi diversi. Attenzione mette in relazione **posizioni diverse della sequenza di token**: confronta token 7 con token 3 calcolando prodotti scalari tra query e key. Connettoma invece è grafo fisso tra **neuroni**: neurone A connesso a neurone B, sempre, indipendentemente dal testo. Non esiste asse "posizione nella frase" dentro cablaggio della mosca, quindi non esiste punto naturale dove innestare attenzione senza inventarsi struttura che con la mosca non c'entra.

Non è mia deduzione isolata: è conclusione a cui è arrivata letteratura. Autori SpikeGPT scrivono che rete spiking può generare impulsi solo su base dell'informazione dei passi temporali precedenti, il che rende impossibile accedere all'intera sequenza in una volta come richiede self-attention. Per questo hanno abbandonato self-attention e passati a meccanismo ricorrente stile RWKV. Soluzione chiave: allineare dimensione sequenziale del linguaggio con dimensione temporale della rete spiking, invece di aggiungere dimensione temporale separata.

Anche SpikeDecoder, tentativo più diretto di realizzare blocco decoder GPT in forma spiking, osserva che self-attention richiede almeno due prodotti matriciali e una softmax, operazioni per natura mal adattate a implementazione spiking.

**Conclusione operativa: in architettura core non ci va nessuna attenzione.** Connettoma stesso, essendo grafo ricorrente, fornisce memoria di contesto attraverso potenziali di membrana che persistono da un token al successivo. Modello mentale corretto: non "un GPT fatto di neuroni di mosca", ma "cervello della mosca che riceve testo nel tempo e produce testo nel tempo", allenato con stesso obiettivo e stessa matematica di training di GPT (predizione token successivo, cross-entropy, discesa del gradiente).

Anche risposta onesta al desiderio espresso: è cervello della mosca che parla, non transformer travestito.

### 4.1 Il vincolo di profondità, che nessuno aveva notato

Conseguenza tecnica non ovvia. GPT-3 ha 96 blocchi in sequenza: ogni token attraversa 96 trasformazioni successive. Cervello della mosca invece largo e basso: analisi su percorsi sensori-motori nel connettoma contano tipicamente da 1 a 3 salti sinaptici tra sensore e motoneurone, con alcuni collegamenti addirittura monosinaptici.

Quindi, se esegui connettoma per un solo passo di simulazione per token, profondità computazionale effettiva è 1: troppo poco per calcolare qualcosa di complesso. Profondità effettiva determinata da numero di passi di simulazione interni per token, chiamiamolo T, perché ogni passo propaga segnale di un salto sinaptico.

**T non è iperparametro libero: è profondità della rete.** Raccomandazione: T fra 6 e 10, cioè ordine di profondità sinaptica reale del cervello più cicli ricorrenti. Costa T volte il calcolo, voce di costo dominante. Partire da T=8.

---

## 5. Il dataset del connettoma maschile: cosa scaricare e cosa contiene

Fonte ufficiale: `https://male-cns.janelia.org/download/`, bucket pubblico `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`, accessibile via HTTPS senza autenticazione (verificato con richiesta HTTP: risposta 200). Licenza CC-BY. Dataset neuPrint corrispondente: `male-cns:v1.0`.

Connettoma contiene 166.691 neuroni completamente revisionati e annotati, con 11.691 tipi cellulari. Tasso completamento dichiarato: 94% presinaptico, 42% postsinaptico (quest'ultimo da tenere presente come limite di qualità del dato).

### File necessari (già scaricati in `dataset/male_cns/`)

| File | Dimensione | Contenuto |
|---|---|---|
| `body-annotations-male-cns-v1.0.feather` | 13,8 MB | Superclasse, classe, tipo, lato, soma di ogni corpo. Serve per scegliere ingressi e uscite |
| `body-neurotransmitters-male-cns-v1.0.feather` | 41,3 MB | Predizione neurotrasmettitore per corpo. Serve per segno secondo legge di Dale |
| `connectome-weights-male-cns-v1.0.feather` | 1,05 GB | Grafo completo connessioni, peso corpo-a-corpo. È la rete |

### File da NON scaricare

`syn-points` (12,7 GB), `syn-partners` (6,8 GB), `tbar-neurotransmitters` (2,7 GB) e volumi immagini EM servono per analisi anatomiche a livello di singola sinapsi, non per costruire rete. Scaricarli consumerebbe metà del disco libero senza beneficio.

### Composizione reale della rete, misurata

Distribuzione per superclasse sui 211.577 segmenti annotati (i 44.877 con superclasse assente sono frammenti non identificati):

| Superclasse | Conteggio | Ruolo nel nostro modello |
|---|---|---|
| ol_intrinsic | 89.403 | Lobo ottico, elaborazione visiva pura — **da escludere** |
| cb_intrinsic | 32.164 | Cervello centrale — **è il nostro calcolatore** |
| vnc_intrinsic | 13.161 | Cordone nervoso, controllo motorio — **da escludere** |
| visual_projection | 9.201 | Proiezioni visive — da escludere |
| vnc_sensory | 6.370 | Sensori corporei — da escludere |
| ol_sensory | 6.098 | Fotorecettori — da escludere |
| cb_sensory | 4.868 | Sensori del cervello centrale (olfatto, gusto, meccanico) — **candidati ingresso** |
| ascending_neuron | 1.846 | Dal cordone al cervello |
| descending_neuron | 1.314 | Dal cervello al cordone — **candidati uscita** |
| vnc_motor | 708 | Motoneuroni del cordone |
| cb_motor | 107 | Motoneuroni del cervello — candidati uscita secondari |

Classi funzionali rilevanti dentro cervello centrale: cellule di Kenyon 4.064 (corpo fungiforme, centro di apprendimento e memoria della mosca), complesso centrale 2.950 (navigazione e memoria di lavoro), olfattivi 2.639, gustativi 1.428, MBON 97, neuroni dopaminergici 340.

**Osservazione decisiva per scoping:** lobi ottici da soli (ol_intrinsic + ol_sensory + visual_projection ≈ 104.700 neuroni) sono più della metà dell'intero sistema nervoso, e sono preelaborazione visiva. Cordone nervoso (≈ 20.400) è controllo zampe e ali. Nessuno dei due serve per linguaggio. Escludendoli ottieni parte "cognitiva", che rientra nel budget computazionale. Non è taglio arbitrario per far tornare i conti: è scelta anatomicamente corretta.

### Il grafo, misurato davvero

File pesi contiene **151.856.684 archi** segmento-a-segmento, con schema `body_pre`, `body_post`, `weight` (tutti int64). Grande maggioranza coinvolge frammenti non identificati. Filtrando ai soli neuroni del cervello centrale ottieni rete che interessa:

| Selezione | Neuroni | Archi | Sinapsi totali |
|---|---|---|---|
| Cervello centrale (`cb_intrinsic`, `cb_sensory`, `descending_neuron`, `cb_motor`) | **38.453** | **7.872.083** | 41.470.677 |
| Prototipo (corpo fungiforme + complesso centrale + olfatto) | **10.776** | **1.606.254** | — |

Distribuzione pesi nel cervello centrale: minimo 1, mediana 2, media 5,3, massimo 1.878. Poiché tasso completamento postsinaptico dichiarato è 42%, archi di peso 1 in buona parte rumore, e in connettomica prassi comune applicare soglia. Effetto della soglia:

| Soglia peso | Archi cervello centrale | Archi prototipo |
|---|---|---|
| ≥ 1 | 7.872.083 | 1.606.254 |
| ≥ 2 | 4.479.705 | — |
| ≥ 3 | 3.055.000 | — |
| ≥ 5 | 1.861.681 | 277.306 |
| ≥ 10 | 912.481 | — |

**Raccomandazione: soglia ≥ 5.** Dà 1,86 milioni archi per cervello centrale e 277 mila per prototipo: dimensioni comode per questa GPU, e scarta connessioni meno affidabili. Se vuoi più capacità, scendere a ≥ 2 porta a 4,5 milioni archi restando gestibile.

### Segno sinaptico misurato (legge di Dale)

Sui 38.453 neuroni del cervello centrale, incrociando con file neurotrasmettitori:

| Categoria | Conteggio | Quota |
|---|---|---|
| Eccitatori (acetilcolina) | 24.890 | 64,7% |
| Inibitori (GABA, glutammato, istamina) | 12.017 | 31,3% |
| Modulatori (dopamina, octopamina, serotonina) | 443 | 1,2% |
| Senza assegnazione | 1.103 | 2,9% |

Copertura 97,1%: vincolo di Dale applicabile senza problemi. I 1.103 non assegnati trattabili come eccitatori per default oppure esclusi; modulatori, essendo 1,2%, trattabili come eccitatori in prima approssimazione.

---

## 6. Numeri di fattibilità

### Perché la scala piena è impossibile

Con connettoma intero e TinyStories intero: circa 500 milioni token per epoca, decine di milioni di connessioni, T passi di simulazione per token. Calcolo dominato da traffico di memoria sugli archi del grafo, e su GTX 1080 (320 GB/s banda, niente Tensor Core) esce nell'ordine dei **mesi per singola epoca**. Non è questione di ottimizzazione: è fuori scala di uno o due ordini di grandezza.

### Perché la scala ridotta funziona

Configurazione raccomandata per primo esperimento completo, con numeri ora misurati:

- Rete: cervello centrale, 38.453 neuroni, 1.861.681 archi con soglia peso ≥ 5
- T = 8 passi di simulazione per token
- Troncamento BPTT: 64 token
- Batch: 8-16 sequenze
- Dataset: 20-50 MB di TinyStories (circa 5-12 milioni caratteri), non 2,23 GB
- Vocabolario: a livello di carattere, circa 100 simboli

Conto memoria GPU con questi parametri: pesi sono 1,86 milioni di valori in virgola mobile, circa 7,5 MB, che con gradienti e i due momenti di Adam diventano una trentina di MB — trascurabile. Costo vero è salvataggio stati per backward troncato: 64 token × 8 sottopassi × 38.453 neuroni × batch 8, che in booleani per spike sono circa 157 MB, e in virgola mobile per potenziali di membrana circa 630 MB. Totale ampiamente dentro i 7,2 GB, con margine per alzare batch.

Costo calcolo dominato da traffico memoria sugli archi: 1,86 milioni archi × 8 sottopassi per token, moltiplicato per passaggio all'indietro. Su banda 320 GB/s questo colloca tempo per epoca su 5-12 milioni caratteri nell'ordine delle **ore singole**, non dei giorni né dei mesi.

### Prototipo iniziale ancora più piccolo

Prima di lanciare configurazione sopra, conviene prototipo su **corpo fungiforme più complesso centrale più via olfattiva**: 10.776 neuroni, 277.306 archi con soglia ≥ 5. È sistema di apprendimento e memoria della mosca, con circa un settimo degli archi della configurazione piena, quindi gira in frazione di ora per epoca. Serve per far girare tutta la pipeline end-to-end e scoprire bug prima di impegnare giorni di calcolo.

---

## 7. Cosa aspettarsi realisticamente

Sezione esiste per evitare delusioni, non per scoraggiare.

Due punti di riferimento in letteratura impietosi e da leggere bene.

**SpikeGPT**, unico modello linguistico spiking che abbia davvero funzionato su scala reale, esiste in due varianti: 46 milioni di parametri a livello di carattere addestrato su Enwik8, e 216 milioni con tokenizzazione BPE addestrato su OpenWebText2. Addestrati su **quattro GPU V100**, rispettivamente per 12 e 48 ore. Risultato: supera nettamente architetture LSTM e si avvicina ad alcune varianti semplificate del Transformer, ma autori scrivono esplicitamente che resta divario rispetto al Transformer classico, e che sul dataset più grande (WikiText-103) resta indietro rispetto a GPT-2.

**SpikeDecoder**, tentativo più diretto di fare blocco decoder GPT completamente spiking, lavora **solo a livello di singoli caratteri**. Autori dichiarano che traduzione di token linguistici completi in impulsi è problema complesso fuori dallo scopo del lavoro, e che non hanno nemmeno cercato ottimizzazioni delle prestazioni perché obiettivo era costruire punto di partenza stabile.

Tradotto per noi: modello a livello di carattere costruito su circa 38.000 neuroni di mosca, addestrato su frazione di TinyStories su singola GTX 1080, produrrà plausibilmente testo che assomiglia all'inglese, con struttura di parole riconoscibile e forse frammenti di frase coerenti. **Non** produrrà conversazione, non risponderà a domande, non sarà paragonabile a assistente. Risultato interessante non è qualità del testo: è misura di quanto cablaggio biologico reale aiuti o ostacoli rispetto ai controlli.

---

## 8. Il controllo scientifico obbligatorio

Va eseguito, non è opzionale, ed è differenza tra esperimento e demo.

Per ogni configurazione addestrata su connettoma reale, va addestrata in parallelo rete identica in tutto (stesso numero neuroni, stesso numero connessioni, stessa distribuzione gradi, stesso rapporto eccitatori/inibitori, stessi iperparametri) ma con **grafo rimescolato casualmente**. Se connettoma reale non batte grafo casuale, biologia non contribuisce nulla e risultato va riportato come tale.

Controllo ha precedenti diretti. Lavoro su conn2res usa connettomi biologici reali come reservoir e li confronta con reti riconnesse casualmente, trovando vantaggio del cablaggio reale. Studio su reti basate su connettoma ad alta sparsità fa stesso confronto e trova maggiore robustezza del cablaggio biologico. All'opposto, critica del "digital sphinx" — connettoma di verme messo a controllare corpo di mosca, che produce camminata realistica pur essendo biologicamente insensato — mostra cosa succede quando controllo manca: comportamento convincente che non dimostra nulla.

---

## 9. Setup software esatto

```
pip uninstall torch
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
pip install spikingjelly
```

Verifica che GPU sia effettivamente attiva:

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Deve stampare versione con suffisso `+cu126`, `True`, e `NVIDIA GeForce GTX 1080`. Se stampa `False`, build installata è ancora quella CPU.

Nota: non installare BrainPy, BrainScale o BrainTrace per uso in GPU su questa macchina. Girerebbero su CPU. Restano utili come riferimento concettuale per algoritmo a memoria lineare, se in futuro si passasse a Linux o WSL2.

---

## 10. Pipeline completa, in ordine di esecuzione

**Fase 1 — Costruzione del grafo.** Leggere i tre file feather. Filtrare corpi tenendo superclassi del cervello centrale (`cb_intrinsic`, `cb_sensory`, `descending_neuron`, `cb_motor`). Costruire lista archi dal file pesi, tenendo solo archi fra corpi selezionati. Assegnare a ogni neurone segno dal neurotrasmettitore. Salvare risultato come matrice sparsa in formato CSR più vettore di segni. Verificare e riportare: numero neuroni sopravvissuti, numero archi, grado medio, frazione eccitatoria.

**Fase 2 — Dati.** Estrarre sottoinsieme di TinyStories (partire da 20 MB). Tokenizzare a livello di carattere, vocabolario ricavato dai dati. Salvare come array di interi su disco in formato memory-mapped, per non saturare i 13,9 GB di RAM.

**Fase 3 — Modello.** In PyTorch: matrice embedding dei caratteri, proiezione lineare verso popolazione di ingresso (`cb_sensory`); nucleo ricorrente spiking che esegue T passi per token usando matrice sparsa del connettoma, con neuroni leaky integrate-and-fire e gradiente surrogato; lettura lineare da popolazione `descending_neuron` verso logit del vocabolario. Potenziali di membrana persistono fra token e staccati dal grafo computazionale al troncamento.

Dettagli implementativi già verificati in letteratura e da rispettare: mettere neurone LIF **prima** della somma residua, mai dopo; non usare LayerNorm, che ricalcola media e deviazione dinamicamente anche in inferenza e non è fondibile, ma Power Normalization o BatchNorm fondibile con layer lineare.

**Fase 4 — Training.** Cross-entropy sul token successivo, Adam, BPTT troncato a 64 token. Solo magnitudini dei pesi sinaptici allenabili; segni restano fissi. Log di loss e bit-per-carattere su set di validazione separato.

**Fase 5 — Controllo.** Ripetere identicamente con grafo rimescolato a parità di statistiche. Confrontare curve.

**Fase 6 — Generazione.** Campionare testo dal modello, salvarne esempi a diverse epoche, per vedere progresso qualitativo.

---

## 11. Decisioni ancora aperte, da prendere prima di partire

1. **Prototipo piccolo prima o direttamente cervello centrale?** Raccomandato: prototipo su corpo fungiforme e complesso centrale (~7.500 neuroni), perché fa emergere bug in un'ora invece che in un giorno.
2. **Pesi iniziali dal conteggio sinaptico reale, oppure casuali sulla topologia reale?** Vale la pena provare entrambi: una delle domande scientifiche interessanti dell'esperimento, e costa poco perché cambia una riga.
3. **T = 8 confermato?** Da tarare sul prototipo, misurando se aumentarlo migliora loss o solo costo.

---

## 12. Nota sulla novità

Da ricerca svolta non risulta letteratura che faccia predizione del token successivo su connettoma biologico reale. Lavori più vicini: connettomi di C. elegans usati come architettura per classificare immagini (MNIST e Fashion-MNIST), connettomi biologici usati come reservoir per compiti cognitivi generici, e modelli linguistici spiking costruiti però su architetture generiche inizializzate a caso. Combinazione specifica — connettoma reale come substrato, compito linguistico autoregressivo — non ha precedenti diretti trovati.

Con connettoma maschile del 3 settembre 2026 questo vale a maggior ragione, essendo dataset disponibile da pochi giorni.
