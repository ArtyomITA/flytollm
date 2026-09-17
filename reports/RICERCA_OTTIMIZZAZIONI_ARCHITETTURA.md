# Mosca LM — ricerca ottimizzazioni

Data: 2026-09-12. **Proposte NON approvate; revisione utente prima adozione.** Nessun benchmark/installazione/modifica codice eseguiti. Ricerca documentale + lettura sorgenti.

## Verdetto

Piano plausibile come esperimento; qualità LM ignota. Priorità: **gradiente utile → propagazione sparse → memoria BPTT → più archi**. Attenzione per prima: rendimento probabilmente basso. Conservare nodi/archi/ID; anatomia preservata ≠ fisiologia preservata.

GTX1080: FP32 riferimento. FlashAttention-2/Triton ufficiali incompatibili. CSR trainabile candidato concreto: PyTorch 2.14 supporta backward CSR×dense. Nessuna fonte dimostra speedup di questa architettura su questo hardware. Risultati altri modelli: motivazione, non previsione.

Legenda: **D** documentato; **L** osservazione locale; **I** deduzione/proposta da validare. Benefici tabella: potenziale, non misura.

## Priorità / decisioni proposte

| Ordine | Proposta | Beneficio | Costo / rischio | Decisione proposta |
|---|---|---|---|---|
| P0 | Verificare CE, causalità, gradiente porte/core/KV | Evita training inutile | Basso / basso | Accettare |
| P0 | Telemetria dinamica + norme gradienti per gruppo | Stabilità, neuroni utilizzati | Basso / basso | Accettare |
| P0 | Profilare LM completo; forward/backward separati | Localizza costo reale | Basso / basso | Accettare |
| P1 | CSR×dense + gradiente valori; confronto custom | Potenzialmente alto throughput | Medio / medio | Esperimento prioritario |
| P1 | BPTT/checkpoint puri; correnti generate per token | Alto risparmio attivazioni | Medio / medio | Esperimento prioritario |
| P1 | Porte/readout sensibili a segnale debole | Apprendimento LM | Medio / alto qualità | Esperimento prioritario |
| P1 | T=2/4/8 + B/L separati | Riduzione costo | Basso / alto dinamica | Ablazione, non sostituzione automatica |
| P2 | Indici int32, riordino, fusioni CUDA mirate | Memoria/banda/lanci | Medio-alto / medio | Dopo profilo |
| P2 | BPE 2k/4k/8k; confronto byte/s e qualità | Meno passi per testo | Medio / medio | Piccolo confronto |
| P2 | Recupero archi + prior controllato | Fedeltà strutturale | Alto compute / medio | Progressivo, dopo baseline |
| P2 | 8-bit Adam, con versione compatibile | Memoria stati optimizer | Medio / medio | Solo se stati dominano |
| P3 | E-prop, ALIF, eventi, attenzione lineare | Ricerca alternativa | Alto / alto | Rinviare |
| — | FlashAttention-2, Triton ufficiale, TF32/BF16 accelerati | Assente su GTX1080 | Incompatibilità | Scartare per v1 |

## 1. Dove stanno byte e costo

**L:** `fly_core.py`: per sottopasso, `Propagate.forward` gather + prodotto + `index_add_`; backward gather + riduzione batch + `index_add_`. Tutti archi percorsi anche con firing ~8%. Spike sparsi ≠ esecuzione sparsa per eventi. `weights()` già fuori loop: non riproporre spostamento. `bench_runtime.py` già `foreach=False`.

**I, contabilità FP32**, N=166700, B=8, L=64, T=8; esclusi overhead/autograd/workspace:

| Oggetto | Formula | Dimensione |
|---|---|---|
| Un potenziale per sottopasso | B×N×L×T×4 | ~2605 MiB |
| Spike booleani salvati | B×N×L×T | ~651 MiB |
| Corrente densa intera finestra | B×N×L×4 | ~326 MiB |
| KV, W=128, d=256 | 2×B×W×d×4 | **2 MiB** |
| Embedding legato 4096×256 | V×d×4 | 4 MiB; 16 con grad+Adam |
| Raw pesi + grad + due momenti, E=25.58M | E×16 | ~390 MiB |
| Due liste indici int64, E=25.58M | E×16 | ~390 MiB |

Non sommare alla cieca: alias, segmenti checkpoint, temporanei e vita tensori cambiano picco. Riferimento locale B8/L64/soglia10: picco allocatore ~4241 MiB; GPU totale ~5742 MiB. Full-graph non provato. Attivazioni, non embedding, prima leva plausibile.

**Esperimento minimo:** LM B1/L4; memoria per fase; tempo token/propagazione/LIF/adattatori/SDPA/CE/backward/optimizer; poi B/L crescenti sotto guardie. Riportare byte testo/s, token/s, picco allocatore e fisico. Nessuna estrapolazione da posizioni sintetiche a TinyStories.

## 2. Propagazione: salvare anatomia, cambiare rappresentazione

**D:** PyTorch 2.14 `sparse.mm`: COO×dense e CSR×dense, backward su entrambi input; BSR/CSC/BSC×dense non supportati dalla stessa API. Vecchia frase «CSR senza gradiente valori» non vale come regola attuale. Supporto API ≠ verifica build installata. [PyTorch 2.14 sparse.mm](https://docs.pytorch.org/docs/2.14/generated/torch.sparse.mm.html)

**I:** costruire A[dst,src], `syn=(A @ s.T).T`. Topologia CSR una volta; valori `sign*softplus(raw)` aggiornati. Non passare Parameter sparse all'Adam: mantenere `raw` denso vettoriale, verificare catena gradiente valori→raw. Verificare duplicati prima coalescing: aggregarli può cambiare parametrizzazione/prior. Cache permutazione archi↔raw; nessun cambio body ID pubblico.

Custom backward: `grad_s=A.T @ grad_out`; `grad_w[e]=sum_b grad_out[b,dst[e]]*s[b,src[e]]`. CSR forward + trasposta CSR precalcolata candidato; gradienti archi via kernel dedicato o codice attuale. Trasposta richiede permutation valori e memoria extra. Mai densificare N×N: ~103.5 GiB FP32.

**D:** cuSPARSE 12.6 documenta SpMM CSR/COO, FP32, indici 32/64 bit; algoritmi/layout/workspace differiscono. **I:** int32 sicuro per indici compatti e contatori qui <2^31; body ID originali restano int64. Risparmio due liste ~195 MiB a 25.58M archi; supporto gather/index_add/constructor da verificare. [cuSPARSE 12.6](https://docs.nvidia.com/cuda/archive/12.6.0/cusparse/index.html)

Riordino per destinazione/sorgente, chunk 64k/256k/1M, orientamento `[N,B]`: candidati per località; sorting non garantisce velocità. Grafo identico, ordine riduzioni può cambiare arrotondamenti. Blocchi anatomici densi/BSR: prima misurare occupazione; padding e backward possono annullare vantaggio. Sparsità 2:4/pruning hardware: modifica archi, nessun razionale per v1.

**Esperimento minimo:** toy con archi duplicati/isolati; uguaglianza forward + gradienti s/raw; CNS soglia10 B1/4/8, forward+backward+optimizer. Poi soglia5. Includere conversioni/layout e workspace. Custom CUDA C++ solo se CSR perde o backward resta dominante; compilazione Windows/sm_61 da provare separatamente.

**Event-driven:** forward può saltare sorgenti silenti. Backward `grad_s` non dipende da s: saltare sorgenti silenti cambia surrogato. Compattazione eventi, gradi sbilanciati, dinamica indici e CUDA graph aggiungono costo. Rinviare; prototipo deve preservare backward completo. Bit-packing spike bool: massimo teorico 8× su quel solo storage, unpack/fusione necessari; inferenza ≠ training.

## 3. Autograd, cache, RAM

**D:** activation checkpointing scambia memoria con ricomputazione; `use_reentrant=False` raccomandato. **I:** segmenti devono essere funzioni pure di stato, token, pesi, cache e posizioni. Durante ricomputazione, append/eviction di cache globale produce doppie mutazioni. Restituire stato/cache; nessuna scrittura laterale. [Checkpoint](https://docs.pytorch.org/docs/2.14/checkpoint.html)

Attuale loss sintetica `window()` non basta: API sottopassi + letture, senza loss interna, necessaria al LM. Generare corrente per token dentro segmento, evitando `[L,B,N]` permanente. Segmenti 2/4/8 token: confronto memoria/tempo/equivalenza. Checkpoint intorno al solo core non elimina tensori densi tenuti fuori. `save_for_backward` conserva riferimenti: pesi/indici condivisi non automaticamente duplicati T×L.

TBPTT: detach di v, spike e cache ai confini; preservare valori. KV vecchi conservano info ma non gradiente verso encoder storico. Ricomputare proiezioni da rappresentazioni compatte detach è altra politica: riapre gradiente su Wk/Wv, non sul core passato; cambia costo/semantica, dichiarare. Transformer-XL precedente per memoria di segmento; non prova ottimalità per SNN. [Transformer-XL](https://arxiv.org/abs/1901.02860)

**Rischio staleness:** dopo optimizer.step, stato/cache derivano da pesi precedenti. Continuazione TBPTT ammissibile come scelta; non equivale a rigiocare prefisso con pesi aggiornati. Esperimento: burn-in senza gradiente vs stato mantenuto; medesime storie. Evitare `retain_graph=True` fra aggiornamenti.

Cache training: ring buffer in-place può sovrascrivere tensori salvati per backward. Partire con aggiornamenti funzionali e finestra limitata; ring mutabile solo inferenza o backend autograd verificato. Niente caching pesi trasformati attraverso optimizer.step; invalidazione obbligatoria.

RAM: NPZ attuale materializza array; `.npy` separati memory-mapped + manifest/hash candidato per grafo grande. Loader concatena parti: picco include parti+array finale; allocazione a due passate o riempimento file evita quel picco, aumenta I/O. Pretokenizzazione una volta; uint16 sufficiente per V≤65536, conversione batch al dtype richiesto. Nessun dataframe completo aggiuntivo.

Offload attivazioni CPU disponibile via saved-tensor hooks, ma trasferimenti PCIe e RAM limitata: ultima opzione, non gratis. Limitare pinning/prefetch. [PyTorch saved tensors](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)

CUDA graph: forme/indirizzi statici, pool memoria e vincoli cattura; acquisire solo configurazioni già corrette. Cache/padding/reset per slot devono restare validi. +4–5% locale su B8 non giustifica complessità immediata LM. `empty_cache()` non libera tensori vivi; non usarlo ogni step. [CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)

## 4. SNN: stabilità prima velocità

**L:** gain4→NaN L64; gain0.5 finito tre update. Non dimostra stabilità CE. `raw` softplus può avere derivata piccola per magnitudini piccole: loggare `sigmoid(raw)`, gradienti e aggiornamenti relativi per classi/gradi. Peso piccolo ≠ arco inutile, ma non apprendibile.

**I:** AdamW decay standard su `raw` problematico: raw negativo spinto verso zero → softplus verso ~0.693, quindi magnitudine può **aumentare**. Proposta: decay raw=0; eventuale penalità su magnitudini fisiche o deviazione dal prior, coefficienti separati. Non chiamare weight decay anatomico il decay del parametro trasformato. Equazione AdamW documentata; conseguenza softplus dedotta dal codice. [AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)

**D:** scala surrogato/reset influenza apprendimento SNN; reset detach è variante dell'estimatore, non equivalente numericamente. Mantenere reset attuale come riferimento. Testare solo dopo evidenza instabilità: pendenza/ampiezza separabili, clipping norme, eventualmente reset detach dichiarato. Clipping agisce su gradienti finiti, non ripara NaN presenti. [Zenke–Vogels](https://www.biorxiv.org/content/10.1101/2020.06.29.176925v1.full), [analisi surrogati/reset](https://direct.mit.edu/neco/article/37/5/886/128506/Elucidating-the-Theoretical-Underpinnings-of), [gradient clipping RNN](https://arxiv.org/abs/1211.5063)

Telemetria minima: distribuzione u−soglia, frazione silente/satura, rate per popolazione, norme per distanza temporale, gradiente porte/core/QKV, massimo potenziale, update/raw e update/magnitudine. Regolarizzazione rate debole opzionale; target0.15 sintetico non obiettivo biologico né LM. Budget T ridotto cambia leak effettivo/token (`0.95^T`), input integrato e profondità percorsi: T2/4/8 non semplice accelerazione equivalente.

Parametri leak/threshold trainabili per tipo neurone: poco storage, possibile qualità; più libertà fisiologica. ALIF aggiunge memoria/adattamento e stati; e-prop sostituisce BPTT con apprendimento diverso, tracce spesso per sinapsi: non gratis a E≈25M. Rinviare entrambi finché CE baseline fallisce su dipendenze controllate. [E-prop](https://arxiv.org/abs/1901.09049)

**Esperimento minimo:** overfit poche storie + task ritardo/copia; T e surrogate uno alla volta, budget tempo uguale; metriche sopra. Nessun gradcheck numerico ingenuo su spike hard: verifica backward contro riferimento dello stesso surrogato.

## 5. Attenzione e adattatori

Confermare MHA softmax d256/W128. KV grezzi 2 MiB B8: GQA→1 testa risparmierebbe ~1.5 MiB, non GiB. Sotto queste dimensioni, nuova architettura GQA/lineare poco prioritaria. GQA cambia parametrizzazione; attenzione lineare cambia operatore e stato, non kernel equivalente. [GQA](https://arxiv.org/abs/2305.13245)

**D:** SDPA non-square `is_causal=True` usa allineamento upper-left: Q lungo1 rischia accesso solo prima chiave. Query corrente + cache solo passato: `is_causal=False`, maschera validità esplicita. Booleano SDPA True=ammesso. Memoria vuota gestita senza softmax vuoto; dropout0 anche eval. [SDPA 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

RoPE: ruotare K una volta con posizione assoluta; Q con posizione corrente; eviction non azzera posizioni degli elementi superstiti. Nessuna promessa oltre contesto visto. Test shifting comune Q/K e confine cache. Alternative no-position/relative-bias solo ablation: ricorrenza già codifica ordine, utilità RoPE da dimostrare. [RoPE](https://arxiv.org/abs/2104.09864)

**I:** readout sola media su popolazioni grandi può cancellare segnali e ridurre gradienti singolo neurone. Confrontare mean vs somma/√n + LayerNorm; leggere potenziale e rate separati, gruppi anatomici fini con dimensione finale fissa. Potenziale post-reset da solo cancella neuroni appena attivati; accoppiare spike/rate o segnale pre-reset compatto. Non salvare tutto il grafo per leggere 256 numeri se aggregabile dentro segmento.

Porte sparse: mappa con copertura bilanciata componenti embedding; scala per fan-in; bias/gain piccoli trainabili; verificare correnti positive/negative e neuroni silenti. Seme porte come fattore sperimentale. Reiniezione attenzione con gate scalare iniziale piccolo **non esattamente zero**: zero può bloccare inizialmente gradienti QKV. Gate troppo piccolo limita apprendimento, monitorare.

T_pre+T_post fisso; confrontare 4+4 e 6+2. Porte sensoriali vs popolazioni esplicite ulteriori = cambio interfaccia, non scoperta anatomica. Conservare divieto bypass diretto verso logits; confronto con bypass solo baseline diagnostica separata.

**Esperimento minimo:** CE e task recupero token distante; attenzione on/off, readout mean vs normalizzato, gate monitorato; stessa topologia/parametri/budget dichiarati.

## 6. Tokenizer, dati, training, optimizer

BPE4k candidato ragionevole; non ottimo noto. Byte-level richiede alfabeto completo, decoder coerente, nessuna normalizzazione distruttiva, speciali esclusi dal round-trip ordinario. Tokenizer addestrato solo train e congelato. [Tokenizers](https://huggingface.co/docs/tokenizers/quicktour)

**I:** confrontare V2k/4k/8k offline: byte/token, percentili lunghezza storia, frequenze rare, memoria, tempo tokenizzazione. Poi due migliori con budget wall-clock uguale: più vocab riduce passi core ma aumenta embedding/softmax. Stesso numero token ≠ stesso testo; confronto anche byte e storie viste. Weight tying confermato; output gradiente denso, `Embedding(sparse=True)` non rende sparse l'intero training legato. A V4k softmax completo baseline; adaptive/sampled softmax aggiunge approssimazioni non motivate. [Weight tying](https://arxiv.org/abs/1608.05859)

TinyStories utile palestra inglese sintetico, non benchmark generale né prova confronto con GPT-2. Verificare versione V2, confini storie, split esistenti, deduplicazione train/validation/test. Hash dati/tokenizer. Nessun nuovo corpus grande prima curve apprendimento. [TinyStories](https://arxiv.org/abs/2305.07759)

Batch: bucketing lunghezze, maschera loss, reset v/s/KV/posizioni per slot; padding non deve evolvere stato esempio terminato. B maggiore parallelizza esempi; L maggiore estende credito temporale, T cambia dinamica. Gradient accumulation aumenta batch effettivo ma non finestra gradiente; normalizzare per token validi complessivi, non media delle medie diseguali. Dataloader partire workers0, prefetch limitato; grafi/annotazioni non duplicati nei worker Windows.

Curriculum: prima subset overfit, poi split completo; L corto→lungo solo proposta. T/archi progressivi cambiano distribuzione dinamica: non variare simultaneamente tokenizer/T/gain/soglia. Warmup LR breve, LR separato core/interfacce, clipping e decay separati: ipotesi pratiche da sweep ridotto; nessun numero universale. Early stopping su validation fissata; budget massimo esplicito.

Optimizer: Adam FP32 riferimento. Foreach può aumentare picco ~dimensione parametri; locale già disabilitato. SGD riduce stati ma convergenza diversa. Adafactor fattorizza matrici; `raw[E]` vettore non ottiene vantaggio row/column, quindi scarso interesse sul core dominante. [Adafactor](https://arxiv.org/abs/1804.04235)

8-bit optimizer: documentato CC≥6.0; Windows pacchetti documentati, combinazione wheel/CUDA/sm61 da verificare. Risparmio teorico due momenti da 8 a ~2 byte/peso, prima metadati: ~146 MiB a 25.58M archi; ~36 MiB a soglia5. Non risolve attivazioni multi-GiB. Verificare stabilità piccoli aggiornamenti softplus e cattura; installazione non proposta ora. LLM.int8 richiede CC≥7.5: **non confondere** con optimizer8bit. [bitsandbytes requisiti](https://huggingface.co/docs/bitsandbytes/main/en/installation)

## 7. Precisione / compatibilità

**D:** GP104 FP16 aritmetico 1/64 throughput FP32; FP16 storage può ridurre banda. FP16 non automaticamente più lento per ogni workload, né promessa AMP veloce. Accumuli/LIF/softplus/gradienti FP32 iniziali. Test futuro storage16→compute32 solo se banda/VRAM limita; rounding vicino soglia può cambiare spike. BF16/TF32/Tensor Cores non accelerazioni native GTX1080. [NVIDIA Pascal](https://docs.nvidia.com/cuda/archive/12.2.2/pascal-tuning-guide/index.html)

**D:** FlashAttention-2 CUDA: Ampere/Ada/Hopper; Windows supporto meno consolidato. Triton ufficiale: Linux, NVIDIA CC≥8.0. GTX1080 CC6.1 esclusa. Non investire su `torch.compile`/Inductor GPU come via standard qui; backend diversi richiedono prova specifica e beneficio misurato. [FlashAttention](https://github.com/Dao-AILab/flash-attention), [Triton](https://github.com/triton-lang/triton)

Stack locale già eseguito: conservare versioni e arch list; aggiornare CUDA/PyTorch non ottimizzazione di default. cu126 nel nome torch non garantisce ogni estensione compilabile. INT8/NF4 inferenza e quantizzazione training sono problemi diversi; niente quantizzazione grafo trainabile v1.

## 8. Fedeltà e confronti

Recuperare archi deboli preferibile a eliminare neuroni. **I:** quando 10→5 cambia normalizzazione entrante, preservare pesi archi comuni e definire esplicitamente inizializzazione nuovi; test separato da riaddestramento identico da zero. Alternativa prior normalizzato sul grafo completo: comparabilità migliore ma dinamica iniziale diversa. Non assumere più archi→più qualità.

Conservazione da riportare: N, E, frazione conteggi sinaptici conservati, componenti raggiungibili dalle porte, distribuzioni gradi/segni, copertura neuroni attivi. Congelare archi deboli può ridurre stati optimizer se parametrizzazione separata; propagazione e backprop stato restano. Parametri condivisi per tipo preservano archi ma riducono libertà: variante scientifica, non equivalente.

Studi connectome-constrained mostrano utilità connettoma + task visivo nel sistema studiato; **nessuna prova di vantaggio linguistico CNS completo**. [Lappalainen et al.](https://www.nature.com/articles/s41586-024-07939-3)

Confronti minimi: unigram/bigram; GRU o LSTM piccola; Transformer piccolo stesso tokenizer; core senza attenzione; grafo rimescolato preservando gradi/segni quando possibile; core congelato. Controllare porte/readout e numero parametri. Randomizzazione può alterare raggiungibilità/stabilità: registrarla. Budget uguale wall-clock e confronto secondario a dati uguali; entrambi necessari per distinguere efficienza/capacità.

Metriche: CE/PPL stesso tokenizer; bit/byte = somma NLL /(ln2 × byte testo valutati), politica EOS/BOS/prefisso identica; non media arbitraria per batch. Generazioni prompt/seme/decoding fissi; ripetizioni, coerenza, completamento storia. Test copy/delayed retrieval per credito temporale. Giudice LLM opzionale, versione e rubriche fissate; non unico metro.

## Decisione conclusiva proposta

**Accettare:** FP32, MHA semplice, tokenizer da train, weight tying, API core separata, causalità/gradiente/overfit, profilo completo, confronto sparse corretto, checkpoint puro, metriche byte/s e bit/byte.

**Scartare v1:** ottimizzare KV prima core; Flash/Triton ufficiali; sparse2:4; densificazione grafo; decay raw inconsapevole; eventi con backward tagliato; promessa qualità biologica o speedup.

**Rinviare:** e-prop/ALIF/linear attention/GQA, quantizzazione, CPU offload, kernel custom, curriculum archi complesso. Riaprire solo su collo di bottiglia misurato.

Limiti ricerca: fonti primarie selezionate, non revisione sistematica esaustiva. Letti specifica, stato, risultati, `fly_core.py`, `fly_graph.py`, parti runtime. Nessun audit completo dati NT/recettori/fisiologia; nessuna riproduzione paper; nessun benchmark LM, compatibilità binaria o statistica speedup verificata. Provenienza/checksum ufficiali connettoma e licenze dati restano audit separato. Nessuna dimostrazione che 25.58M archi entrino nel training LM sotto guardie attuali.
