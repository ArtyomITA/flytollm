# Ricerca mirata al design: attention dentro il giro ricorrente (looped transformer) e applicazione al core-connettoma

Data: 17 settembre 2026. Ricerca su fonti primarie (arXiv). Marcatura affidabilità: **[C]** = citazione o numero letto direttamente sulla pagina del paper (HTML/abstract arXiv); **[P]** = parafrasi dalla pagina del paper senza citazione verbatim, oppure fonte secondaria attendibile; **[NV]** = non verificato, o affermazione trovata in un riassunto di ricerca che il controllo diretto sulla fonte ha smentito.

---

## 0. Sintesi

La domanda del committente ("mettere l'attention dentro il giro, ripetuta a ogni sottopasso con gli stessi pesi") non è un'idea marginale: è esattamente il design canonico dei modelli recurrent-depth degli ultimi due anni. Universal Transformer, Huginn, Ouro, Mixture-of-Recursions, Parcae e i lavori 2026 ricalcolano l'attention *per intero, a ogni iterazione del loop, con gli stessi pesi*. Nessuno di questi lavori esegue l'attention una sola volta per token fuori dal blocco ricorrente. L'obiezione "più attention = più lavoro fuori dal grafo" non trova corrispondenza nella letteratura: nei looped transformer l'attention *è* parte del blocco che viene ripetuto, e la contabilità dei FLOP viene fatta a pari parametri o a pari FLOP totali, non separando "dentro" e "fuori".

Il punto di design realmente critico che emerge, e che è specifico e misurabile, è un altro: **la KV cache deve essere separata per iterazione**. Il token t all'iterazione r attende ai token < t *alla stessa iterazione r*. Tutti i modelli che funzionano fanno così, e chi prova a collassare le cache in una sola perde molto, in modo asimmetrico: riusare la cache del *primo* passo costa poco, riusare quella dell'*ultimo* passo distrugge il modello.

Il secondo punto è l'input injection: l'embedding del token va reiniettato a ogni iterazione. Serve meno per la qualità pura (guadagni dell'ordine di 0,02 nat) e molto per la stabilità del punto fisso: senza reiniezione le rappresentazioni collassano.

---

## 1. KV cache attraverso le iterazioni del loop

### 1.1 Il default di chi funziona: una cache per iterazione

**Ouro / LoopLM (Zhu et al., arXiv 2510.25741, ottobre 2025).** Il paper è esplicito: "During the prefilling phase (processing the input prompt), we find that all four recurrent steps require their own KV caches, as each step transforms the representations in ways that cannot be approximated by earlier steps" **[C]**. E ancora: "naively, each recurrent step requires maintaining its own KV cache, leading to 4× memory overhead for our 4-step model" **[C]**. Il tentativo di riuso in prefill è quantificato: "Attempting to reuse KV caches during prefilling leads to performance degradation (>10 points on GSM8K)" **[C]**. In decoding testano "Last-step reuse" e "First-step reuse" **[C]**, con margini migliori ma non a costo zero.

**Huginn / recurrent depth (Geiping et al., arXiv 2502.05171).** Il modello ha bisogno di una propria implementazione di cache (`HuginnDynamicCache`) perché "altrimenti le KV-cache delle chiamate successive al blocco ricorrente sovrascrivono le precedenti" **[P]** — cioè di default ogni iterazione scrive la sua voce. Per limitare la memoria a r fino a 32 introducono un budget: "We set a fixed KV-cache budget for the recurrence at every token k, and at iteration i, read and write the cache entry i mod k" **[C]**, con esempio "we set a maximum KV-cache budget of 16 steps, overwriting the KV-cache of the 1st step when executing the 17th step" **[C]**. Dichiarano che questa condivisione è quasi gratuita: "we find that we can simply share KV-caches in our model with minimal impact to performance" **[C]**, con numeri MTBench riportati per budget molto piccoli (ordine 5,86 contro 5,63 di riferimento) **[P]** — i valori esatti non sono stati estratti in modo pulito dalla pagina HTML, quindi vanno riletti prima di citarli.

Il punto importante: il budget di Huginn non è "una cache condivisa"; è un *anello* di k cache distinte, cioè comunque una separazione per iterazione, troncata modulo k.

**Looped Latent Attention (arXiv 2607.15456, 2026).** Quantifica il problema in modo netto: "A model with T=4 loops over D=24 layers therefore carries the KV cache of a 96-layer decoder while sharing the parameters of a 24-layer one" **[C]**. È la formulazione più chiara del costo: i parametri si condividono, la cache no. La tabella di confronto a budget di cache 4× (GSM8K, distillazione da Ouro-1.4B) è decisiva per il design **[C]**:

| Metodo | GSM8K |
|---|---|
| Teacher (cache piena, per-loop) | 0,794 |
| LLA, compressione sull'asse loop | 0,800 |
| MLA sull'asse teste | 0,522 |
| Condivisione cross-layer | 0,660 |
| Riuso della sola cache dell'ultimo loop (zero parametri) | 0,000 |

Il commento degli autori: riusare solo l'ultimo passo "collapses GSM8K generation to zero", e "the loop trajectory cannot be replaced by its endpoint" **[C]**. Capacità di servizio su H200 a 4k di contesto: 32 sequenze senza compressione, 128 a 4×, 768 a 21,3× **[C]**; qualità 0,795 a 1,33×, 0,750 a 2,0×, 0,590 a 4,0× senza raffinamento **[C]**.

### 1.2 Le varianti economiche e quanto costano davvero

**Mixture-of-Recursions (Bae et al., arXiv 2507.10524, NeurIPS 2025)** è il lavoro che ha isolato esplicitamente questa scelta, con due strategie nominate:

- *recursion-wise KV caching*: "only tokens routed to a given recursion step store their key–value entries at that level" **[C]**, cioè cache separate per ricorsione, ristrette ai token effettivamente attivi a quel livello. Memoria e I/O ≈ (N_r+1)/(2 N_r) del vanilla (≈67% per N_r=3) **[P]**; i FLOP di attention scendono quadraticamente, (k/N_ctx)² per layer **[P]**.
- *recursive KV sharing*: "caching KV pairs exclusively at this initial step and reusing them across all subsequent recursions" **[C]**. Memoria ≈ 1/N_r (≈33% per N_r=3) **[P]**, ma i FLOP di attention scendono solo linearmente, "attention FLOPs only decrease by a factor of k/N_ctx" **[C]**, perché le chiavi restano di lunghezza piena a ogni profondità.

Numeri a N_r=3 (NLL su validazione, accuratezza few-shot media) **[C]**, con l'avvertenza in nota:

| Configurazione | NLL | Acc. media |
|---|---|---|
| Transformer vanilla | 2,7824 | 42,3% |
| MoR expert-choice + recursion-wise caching | 2,7925 | 42,6% |
| MoR expert-choice + recursive KV sharing | 2,7983 | 41,9% |
| MoR token-choice (variante degradata) | 2,9163 | 40,0% |

Nota sull'affidabilità: due letture successive della stessa pagina hanno attribuito il valore 2,9163 una volta al *recursive sharing con token-choice* e una volta al *recursion-wise con token-choice* **[P]**. Il fatto robusto è l'ordine di grandezza: con routing expert-choice, passare da cache-per-ricorsione a cache condivisa dal primo passo costa circa **0,006 nat** e circa 0,7 punti di accuratezza; la scelta del routing pesa molto di più (≈0,12 nat) della scelta di caching.

### 1.3 Terza via: comprimere invece di collassare

**Gated Recurrent Transformers (arXiv 2608.15062, 2026)** parte da cache per-ricorsione naive e poi "evaluates three compressed strategies that each reduce recurrent KV memory over R layers" — riuso dell'ultimo passo, riuso del primo passo, oppure **media di K/V attraverso i passi**; la media risulta la migliore mantenendo l'accuratezza **[P]**.

**Relaxed Recursive Transformers (Bae et al., arXiv 2410.20672, ICLR 2025)** aggiunge un modulo LoRA per ogni iterazione del loop (durante l'i-esimo loop si attiva l'i-esimo LoRA) e introduce il *continuous depth-wise batching*, che calcola in parallelo le KV mancanti a profondità diverse; stimano guadagni di throughput 2–3× **[P]**.

### 1.4 Conclusione operativa del punto 1

La gerarchia empirica, dal migliore al peggiore, è: cache separata per iterazione ≳ compressione low-rank sull'asse del loop ≳ media delle K/V fra passi ≳ condivisione della cache del **primo** passo ≫ riuso della sola cache dell'**ultimo** passo (catastrofico). Il costo in memoria della variante piena è esattamente ×(numero di iterazioni).

---

## 2. Input injection: forma esatta e ablazioni

**Huginn (2502.05171)** — forma esatta, citata: "Our core recurrent block R starts with an adapter matrix A: R^{2h} → R^h mapping the concatenation of s_i and e into the hidden dimension" **[C]**. Quindi *concatenazione* stato-corrente + embedding, poi proiezione lineare a h. Sulla scelta concat vs somma: "While re-incorporation of initial embedding features via addition rather than concatenation works equally well for smaller models, we find that concatenation works best at scale" **[C]**. La motivazione dichiarata è teorica: "we repeatedly inject the data e in our set-up in every step of the recurrence" **[C]**. La struttura complessiva è prelude/core/coda: e = P(x), s_0 ~ N(0, σ²I), s_i = R(e, s_{i-1}) per i = 1..r, p = C(s_r) **[C]**.

Le ablazioni architetturali di Huginn mostrano che *senza* la combinazione giusta il modello si rompe in due modi distinti **[P]**: con RMSNorm parameter-free e adapter a semplice somma lo stato nascosto collassa (correlazione fra token → 1,0); con adapter appreso e pre-norm il modello *ignora* la ricorrenza — "validation perplexity is the same whether 1 or 32 recurrences are used". La configurazione funzionante usa norma "sandwich".

**Saunshi et al. (arXiv 2502.17416, ICLR 2025)** — verifica diretta: cercando "inject"/"input injection" nel testo HTML non si trovano occorrenze **[C]**. La definizione formale è un puro loop di composizione: p_{θ,T} = OUTPUT ∘ (TB_θ)^T ∘ EMBED **[C]**, quindi **nessuna reiniezione** dell'embedding a ogni giro. Questo è un dato di design importante: il lavoro che dimostra "k layer loopati L volte ≈ kL layer" lo fa *senza* input injection. La reiniezione non è quindi necessaria per ottenere il guadagno di profondità effettiva; serve per stabilità e per compiti di generalizzazione in lunghezza.

**Looped Transformers for Length Generalization (arXiv 2409.15647)** e **A Mechanistic Analysis of Looped Reasoning Language Models (arXiv 2604.11791, 2026)** danno il quadro delle ablazioni. Dal secondo, verificato sulla pagina: "input injection results in stable fixed point behavior for all norm types other than Ouro, whereas omitting input injection means that only pre-norm reaches a stable fixed point" **[P]**; i modelli senza input injection mostrano comportamento di punto fisso degenere; l'input iniettato è ricampionato a ogni loop **[P]**. Il messaggio ricorrente è: effetto *modesto* sulla metrica di compito, effetto *critico* sulla stabilità della traiettoria latente.

**Gated Recurrent Transformers (2608.15062)** offre la quantificazione in nat più direttamente confrontabile con le misure del progetto (ablazione sequenziale, modello piccolo, 20.000 passi) **[C]**:

| Componente | Δ loss |
|---|---|
| Sola ricorrenza (blocco condiviso, nessun accorgimento) | +0,107 nat |
| + prelude/coda | −0,035 nat |
| + rumore sullo stato | −0,018 nat |
| + reiniezione del prelude a ogni ricorsione | −0,022 nat |
| + gate elementwise | −0,048 nat |

E il commento: "The elementwise gate (row 6) is the single largest contributor at −0,048 nats" **[C]**. Quindi: input injection ≈ 0,022 nat; un *gate* moltiplicativo per differenziare i passi vale il doppio.

**Universal Transformer (Dehghani et al., arXiv 1807.03819)** usa una forma diversa e più debole di iniezione: non reinietta l'embedding, ma somma a ogni passo una *coordinate embedding* che codifica posizione e numero di passo, P^t_{i,2j} = sin(i/10000^{2j/d}) + sin(t/10000^{2j/d}) **[P]**. È il minimo indispensabile per rompere la simmetria fra iterazioni.

---

## 3. Loopare l'attention o solo l'MLP?

Qui la letteratura è meno esplicita di quanto la domanda meriterebbe, e conviene dirlo chiaramente.

Fatto strutturale, verificabile su tutte le fonti primarie: **nei looped/recurrent-depth transformer l'attention è dentro il blocco ripetuto e viene ricalcolata a ogni iterazione.** Universal Transformer: A^t = LayerNorm((H^{t-1} + P^t) + MultiHeadSelfAttention(H^{t-1} + P^t)), H^t = LayerNorm(A^t + Transition(A^t)), con le stesse matrici di attention a tutti i passi **[P]**. Huginn, Ouro, MoR, Parcae: idem, il blocco condiviso contiene attention e MLP. Gated Recurrent Transformers è esplicito: i blocchi core condivisi contengono attention standard "recomputed at each recurrence step", producendo attention fresca a ogni iterazione invece di riusare i pattern **[P]**. Non ho trovato alcun modello recurrent-depth di rilievo che loopi il solo feed-forward lasciando l'attention fuori dal giro.

Evidenza indiretta ma pertinente, dalla letteratura sulla condivisione di parametri: in ALBERT condividere i parametri di attention costa poco o nulla, mentre condividere quelli del feed-forward causa la maggior parte del degrado **[P]**. Questo dice che la funzione implementata dall'attention è più "universale" fra i livelli di profondità, cioè più adatta a essere applicata ripetutamente con gli stessi pesi — argomento a favore di mettere l'attention nel loop, non fuori.

Evidenza contraria alla tesi "conta solo ripetere l'attention": nell'ablazione di Gated Recurrent Transformers il contributo maggiore non viene dall'attention ma dal gate elementwise (−0,048 nat) **[C]**. E la sola ricorrenza, senza meccanismi che differenzino i passi, *peggiora* di 0,107 nat **[C]**: ripetere non basta, bisogna dare al blocco un modo per sapere a che passo si trova.

Avvertenza su un falso positivo. Una ricerca ha restituito l'affermazione che in "Rethinking the Role of Efficient Attention in Hybrid Architectures" (arXiv 2606.15378) esisterebbe un'ablazione di rapporto 4:1 fra attention piena e Gated DeltaNet sotto looping, con l'attention piena instabile sotto loop. Il controllo diretto sulla pagina del paper smentisce: il paper non contiene alcuna discussione di looping, di iterazioni weight-tied, né di rapporti 4:1 (i rapporti discussi sono 1:1 e 1:3, e riguardano il mix di layer, non il loop) **[NV]**. Questa affermazione va considerata inesistente.

---

## 4. Ibridi ricorrenza + attention: quante letture, e chi fa il lavoro

Questa famiglia è il precedente più vicino al caso connettoma, perché lì l'attention non è il motore ma un *lettore periodico* di uno stato ricorrente.

**Zamba (Glorioso et al., arXiv 2405.16712)** è l'analogia più stretta con il design proposto. Un backbone di blocchi Mamba con **un singolo blocco attention+MLP condiviso, richiamato ogni 6 blocchi Mamba, con parametri condivisi fra tutte le chiamate** **[P]**. E soprattutto: "The input embeddings are always concatenated with the residual stream going into the shared attention block as this provides an additional path for the model to remember the inputs"; gli embedding originali pre-layer-zero vengono concatenati e la combinazione a doppia larghezza è l'input dell'attention **[P]**. Cioè: stessi pesi, letture multiple per token, input injection per concatenazione — esattamente la ricetta che il committente vuole, già validata in un modello 7B.

**Block-Recurrent Transformer (Hutchins et al., arXiv 2203.07852, NeurIPS 2022).** La cella ricorrente usa attention in due direzioni: in verticale i token fanno self-attention fra loro e cross-attention sugli stati ricorrenti; in orizzontale gli stati fanno self-attention fra loro e cross-attention sui token; chiavi e valori sono condivisi fra le due direzioni, le query no **[P]**. I residui sono sostituiti da gate (i gate fissi, tipo media mobile esponenziale, battono i gate LSTM) **[P]**. Ablazioni rilevanti: **un solo layer ricorrente basta** — aggiungere due layer ricorrenti adiacenti non migliora; la variante "skip", che rimuove l'MLP dalla cella, è la migliore, il che suggerisce che il layer ricorrente faccia soprattutto *lookup* e non calcolo complesso **[P]**. Risultati: 3,52 bit/token su PG19 contro 3,58 di Transformer-XL a finestra 2048, a metà del tempo, con guadagno paragonabile al raddoppio dei parametri **[P]**.

**Feedback Transformer (Fan et al., arXiv 2002.09402).** Tutti i layer attendono a una memoria unica che è la somma pesata delle rappresentazioni di tutti i layer dei token passati; la rappresentazione più bassa del token corrente si forma a partire dalla rappresentazione più astratta del passato **[P]**. È il caso limite opposto: una sola memoria condivisa da tutte le profondità.

**Griffin / RecurrentGemma (arXiv 2402.19427, 2404.07839).** Alternanza fissa di due blocchi residui ricorrenti (RG-LRU) seguiti da un blocco di attention locale MQA, finestra 1024 token **[P]**. Cioè circa un terzo dei blocchi di mixing sono attention, e l'attention è *locale*.

**Jamba (Lieber et al., arXiv 2403.19887).** Rapporto 1 layer di attention ogni 7 di Mamba; 4 blocchi da 8 layer **[P]**. L'ablazione di divisione del lavoro è la più citabile: "Pure Mamba fails and struggles to develop in-context learning capabilities, while the Attention-Mamba hybrid exhibits in-context learning similar to vanilla Transformers" **[P]**; in un ibrido 1,3B senza MoE trovano 12 induction head distribuite su tutti e tre i layer di attention **[P]**. Il lavoro che fa l'attention è quindi identificabile e *specifico* (induction/copy), non generico.

**Mamba-2-Hybrid (Waleffe et al., arXiv 2406.07887).** Composizione 8B: 56 layer totali, 24 Mamba-2 (42,9%), 4 self-attention (7,1%), 28 MLP (50%), attention distribuite uniformemente **[P]**. Il criterio: circa l'**8% dei layer come self-attention minimizza la loss di validazione**, risultato consistente a 130M e 840M **[P]**. Costo dell'assenza di attention (Mamba puro, 1,1T token): MMLU 5-shot 29,19% contro 46,28% del Transformer, cioè 17 punti **[P]**; a 3,5T token il divario si chiude su MMLU (Mamba-2 48,7 vs Transformer 50,07) ma l'ibrido resta davanti a entrambi (51,46 su MMLU; media 12 task 55,82 contro 53,17 e 54,69) **[P]**. Rimangono i fallimenti su copia/rubrica telefonica: memoria "sfocata", cifre giuste in ordine sbagliato, degrado oltre ~500 token **[P]**.

**SpikingBrain (arXiv 2509.05276, TMLR 2026).** Il caso spiking più vicino. Il 7B raggiunge complessità lineare pura "by interleaving linear attention and sliding window attention (SWA) layers with a fixed 4K window in a 1:1 ratio" **[C]**; il 76B ibrido-lineare mantiene linear+SWA ma "standard full-attention layers are interleaved at a 1:6 ratio across layers" **[C]**, con 128 sink token appresi. Neuroni spiking adattivi con soglia dinamica V_th(x) = (1/k)·mean(|x|), sparsità complessiva ≈69,15%, conteggio medio di spike 1,13 per canale **[P]**. Non ho trovato in questa letteratura spiking alcun lavoro che *itera* l'attention dentro il passo ricorrente: l'attention resta un layer attraversato una volta.

**Lacuna rilevata.** Cercando esplicitamente reti spiking o RNN con attention iterata *dentro* il passo ricorrente non emergono lavori. Il progetto si troverebbe in uno spazio di design non occupato: la combinazione "core ricorrente biologico + attention esterna riletta a ogni sottopasso con pesi condivisi" non ha, per quanto ho trovato, un precedente diretto. Gli antecedenti più vicini restano Zamba (blocco attention condiviso richiamato più volte, con concatenazione degli embedding) e Block-Recurrent Transformer (attention che legge e scrive uno stato ricorrente).

**Metriche di "chi fa il lavoro".** Non esiste in letteratura una metrica standardizzata chiamata "attention ablation cost" o "recurrence ablation cost". Quello che si usa di fatto sono tre cose: (a) il divario a pari FLOP/pari parametri fra ibrido, ricorrente puro e Transformer puro (Mamba-2-Hybrid, Jamba); (b) il collasso su compiti diagnostici specifici quando si toglie l'attention (ICL, copia, rubrica — Jamba, Mamba-2-Hybrid); (c) la frazione ottima di layer di attention che minimizza la loss (≈8%, Mamba-2-Hybrid; 1:7 Jamba; 1:6 SpikingBrain-76B; 1:3 Griffin).

---

## 5. I numeri: guadagno a pari parametri, a pari FLOP, e saturazione

**A pari parametri (il loop compra profondità).** Saunshi et al.: un modello k-layer loopato L volte va quasi quanto un modello kL-layer non loopato e batte nettamente il k-layer singolo **[P]**. Numeri riportati: (12⊗2) ottiene 51,2 sui "reasoning primitives" contro 47,5 della baseline (24⊗1), cioè meglio con metà dei parametri **[P]**; sui problemi di matematica (12⊗2) fa 34,3 contro 26,7 di (12⊗1) **[P]**. Sul compito sintetico di addizione, (1⊗12) raggiunge 99,9% contro il 100% di (12⊗1) **[P]**.

La legge di scala proposta è logaritmica nella profondità effettiva: Acc = α·log(D) + β, con D = profondità effettiva (numero di layer × numero di loop), e il rapporto fra le pendenze α_loop/α_base ≈ **1,19** sui reasoning primitives **[P]**: il loop compra *più* accuratezza per unità di profondità di quanta ne comprino parametri nuovi, ma solo sui compiti di ragionamento. Sui compiti di memorizzazione (closed-book QA) il vantaggio sparisce: copertura del divario 56–282% sui compiti di ragionamento contro 34–46% su quelli di memorizzazione **[P]**.

**Mixture-of-Recursions (a pari FLOP di training, 16,5e18).** MoR con N_r=2 ed expert-choice: 2,7511 NLL contro 2,7824 del vanilla, e 43,1% contro 42,3% di accuratezza few-shot, **con circa il 47% di parametri in meno** **[P]**. A token fissi (20B), MoR N_r=2 supera il vanilla con circa il 25% di FLOP di training in meno, 19% di tempo di training in meno, 25% di memoria di picco in meno **[P]**. Scalando (135M–1,7B, N_r=3): a 135M MoR è sotto il vanilla (collo di bottiglia di capacità), da 360M in su pareggia o supera con ~67% di parametri unici in meno **[P]**. Osservazione di design: il numero di ricorsioni che vince gli isoFLOP è **2**, non 8 o 32.

**Huginn (3,5B, 800B token).** Forma (l_p, l_r, l_c) = (2, 4, 2), h = 5280, prelude+testa 1,5B, core 1,5B, embedding legato 0,5B **[P]**. Profondità effettiva 2 + 4r + 2 = 132 layer a r=32 **[P]**. Saturazione fortemente dipendente dal compito: HellaSwag satura a ~8 ricorrenze; ARC-C satura a 8–12 zero-shot ma sale a 32 con 25–50 esempi few-shot; GSM8K CoT continua a migliorare fino a 32 e oltre, da 0% a r=1 a 34,80% a r=32 **[P]**. Contro una baseline non ricorrente allenata sugli stessi 180B token, il ricorrente fa circa 5× su GSM8K CoT (9,02/10,24 contro 1,82/2,20) **[P]**.

**Ouro / LoopLM.** Il picco è alla profondità di training e l'estrapolazione peggiora: MMLU per Ouro-1.4B base, T=1 → 41,21%, T=4 → 67,45%, poi T=5–8 scende a 64,49–66,64% **[P]**. Ouro-1.4B eguaglia Transformer densi da 4B, Ouro-2.6B supera densi fino a 8B (MATH500 82,40 contro 59,60 di Qwen3-4B; 90,85 contro 62,30 di Qwen3-8B-Base) **[P]**. Il vantaggio dichiarato non è capacità di conoscenza ma *manipolazione* della conoscenza **[P]**.

**Loopie (arXiv 2607.16051, 2026).** Usa ricorrenza *per layer* invece che per modello: "each layer is applied recurrently before the computation moves to the next layer" **[C]**. Entrambi i modelli (20B-A2B e 6B-A0.6B) usano **R=2**, e la motivazione è esplicita: "the marginal benefit of recurrence is largest at R=2" **[C]**, perché a budget di pre-training fisso aumentare R costringe a meno token o a un'architettura più piccola. Il modello supera la baseline compute-matched dopo circa 600B token **[P]**.

**Parcae (arXiv 2604.12946, 2026).** Legge di scala unificata training + test-time: L̂(T | μ_rec, D) = E + X·N(μ_rec)^{-x} + Y·D^{-y} + Z·exp(−z·T·μ_rec^{-1}) **[P]**, con esponenti γ_μ ≈ 0,40 e γ_D ≈ 0,78 **[P]**. Il termine esponenziale dice che il looping a test-time satura, e che "gains plateau near μ_rec", cioè **la profondità usata in training determina il tetto dello scaling a test-time** **[P]**. A pari parametri, +2,99 e +1,18 punti su Core e Core-Extended contro Transformer **[P]**; sotto budget di FLOP, vantaggi di 1,2–2,0 punti **[P]**.

**Universal Transformer**, per riferimento storico: bAbI 10K joint 0,29% di errore contro 22,1% del Transformer, LAMBADA e +0,9 BLEU su WMT14 En-De (28,9 contro 28,0) **[P]** (il valore di perplexity LAMBADA estratto, 142 contro 7321, è anomalo e va riletto sulla fonte prima di usarlo **[NV]**).

**Sintesi dei numeri utili al progetto.** Il guadagno tipico del loop a pari parametri è dell'ordine di 0,03 nat di NLL o 1–3 punti di benchmark; il numero di iterazioni che vince gli isoFLOP in pre-training è piccolo (2–4); i valori alti (8–32) pagano solo a test-time e solo su compiti di ragionamento multi-passo, e solo se il modello è stato *addestrato* a quella profondità. Su TinyStories, che non è un compito di ragionamento multi-passo, il regime atteso è quello di R piccolo.

---

## 6. Proposta di design per il caso connettoma

Premessa sui costi. Con 2,75 M archi con segno, un sottopasso del grafo costa circa 5,5 MFLOP (una moltiplicazione sparsa, 2 FLOP per arco); 8 sottopassi costano circa 44 MFLOP per token. Una lettura di attention con d=256, 4 teste, KV cache 128 token costa circa 0,52 MFLOP per le quattro proiezioni Q/K/V/O (4 × 256 × 256 × 2) più circa 0,13 MFLOP per punteggi e somma pesata su 128 posizioni: totale ≈ 0,65 MFLOP, cioè circa **l'1,5% del costo del grafo per token**. Queste stime sono aritmetica mia sui parametri dichiarati del progetto, non numeri di letteratura **[NV]**; vanno confermate con un profiling reale, perché su GPU piccole il costo dominante può essere il numero di lanci di kernel e non i FLOP.

### Variante A — Looped pieno (massima fedeltà alla logica looped)

L'attention viene letta **a ogni sottopasso**, 8 letture per token, stessi pesi a tutte le letture. Ogni sottopasso ha la **sua KV cache** (8 cache × 128 token × 256 dimensioni ≈ 8 × 128 × 256 × 2 valori ≈ 0,5 M valori: memoria trascurabile a questa scala). Il token al sottopasso r attende solo alle voci scritte al sottopasso r dei token precedenti. Input injection stile Huginn: concatenazione dello stato di lettura corrente con l'embedding del token e proiezione 512 → 256 prima di entrare nell'attention; in aggiunta, un identificativo di sottopasso (coordinate embedding stile Universal Transformer, oppure un gate elementwise per sottopasso stile Gated Recurrent Transformers, che nella loro ablazione vale più dell'input injection).

Cosa si condivide: i pesi di attention (uno solo), l'adapter, il proiettore neuroni-discendenti → stato di lettura, il proiettore uscita-attention → corrente di feedback. Non si condivide la cache.

Costo stimato: 8 × 0,65 ≈ 5,2 MFLOP, cioè **×1,12 sul totale per token** rispetto a una sola lettura **[NV]**. In wall-clock, realisticamente ×1,2–1,5 per overhead di lancio.

Questa è la variante che corrisponde letteralmente a Universal Transformer, Huginn, Ouro e MoR.

### Variante B — Looped con cache condivisa dal primo sottopasso

L'attention viene comunque letta a ogni sottopasso (8 letture, stessi pesi), ma le chiavi e i valori si **scrivono solo al primo sottopasso** e si riusano a tutti gli altri: è il *recursive KV sharing* di MoR. Solo le query cambiano da sottopasso a sottopasso.

Cosa si condivide: pesi come in A, più la KV cache. KV cache per iterazione: no, una sola. Input injection: identica ad A (qui serve di più, perché è l'unica cosa che differenzia i sottopassi lato query oltre allo stato).

Costo: stessi FLOP di A meno le proiezioni K e V ripetute, quindi circa ×1,08 **[NV]**; memoria di cache 1/8 di A.

Supporto in letteratura: con routing expert-choice MoR paga circa 0,006 nat rispetto alla cache per-ricorsione **[C]**. Da non confondere con il riuso dell'*ultima* cache, che in Looped Latent Attention azzera GSM8K **[C]**, e da non estendere alla fase di prefill se il modello somiglia a Ouro, dove il riuso in prefill costa più di 10 punti GSM8K **[C]**.

### Variante C — Letture multiple ma poche (2 letture per token)

Attention letta 2 volte per token, per esempio dopo il sottopasso 2 e dopo il sottopasso 6, stessi pesi; 2 KV cache distinte; input injection additiva (somma dell'embedding allo stato di lettura), che Huginn dichiara equivalente alla concatenazione ai piccoli modelli **[C]**.

Costo: ×1,03 sul totale **[NV]**.

Supporto: Loopie usa R=2 perché "the marginal benefit of recurrence is largest at R=2" **[C]**; MoR vince gli isoFLOP a N_r=2 **[P]**; Zamba richiama il blocco attention condiviso ogni 6 blocchi Mamba, cioè poche volte e non a ogni blocco **[P]**. È la variante meno fedele alla logica looped ma la più difendibile sul piano dei FLOP a parità di training, ed è anche il miglior esperimento di controllo intermedio fra "1 lettura" e "8 letture", perché permette di tracciare una curva letture → loss con tre punti (1, 2, 8) invece di due.

### Ordinamento e raccomandazione

Per fedeltà alla logica looped: **A > B > C**. Per rapporto informazione/costo sperimentale, conviene eseguirle nell'ordine **A, C, B**: A è la condizione che risponde alla domanda del committente, C dà il terzo punto della curva letture → loss, B risponde alla domanda separata "quanto della differenza viene dall'avere stati di attention distinti per sottopasso e quanto dal solo rileggere". In tutte e tre, aggiungere un identificativo di sottopasso (gate o embedding di passo) non è opzionale: senza, la sola ripetizione peggiora la loss di 0,107 nat nell'ablazione di Gated Recurrent Transformers **[C]**.

Nota sulle misure già fatte dal progetto. Il dato "8+8 sottopassi migliorano di 0,18 nat a 2000 update ma solo 0,07 a 4000" è coerente con il quadro della letteratura: i guadagni del loop si assottigliano con il training e il numero ottimo di iterazioni in pre-training a budget fisso è piccolo. Il dato "KV 8 = KV 128", cioè uso di circa 8 token di contesto, è invece un segnale che l'attention attuale sta facendo un lavoro locale, molto simile a quello che Griffin ottiene con attention locale a finestra e SpikingBrain con SWA; prima di moltiplicare le letture vale la pena verificare se il collo di bottiglia sia il numero di letture o la capacità dello stato di lettura a 256 dimensioni derivato dai soli neuroni discendenti.

---

## 7. Il controllo giusto: "il grafo fa lavoro o lo fa l'attention?"

La tesi del committente — che il controllo corretto non sia spegnere l'attention ma confrontare il grafo reale con un **null del grafo a pari architettura** — è sostenuta dalla letteratura, con due precisazioni importanti.

**Sostegno diretto.** "Topological Sensitivity in Connectome-Constrained Neural Networks" (arXiv 2604.04033, 2026) rifà esattamente questo esperimento su flyvis con il connettoma di Drosophila, confrontando il connettoma empirico (45.669 nodi, 1.513.231 archi, 12.380 self-loop) con un random naive che eguaglia solo i conteggi globali e con un null **degree-preserving** ottenuto con double-edge swap diretti che preservano in-degree e out-degree di ogni neurone, out-strength, l'intero multiinsieme dei pesi e i self-loop **[C]**. Risultati: con inizializzazione da checkpoint e null naive, il connettoma sembra vincere (loss a 5 passi 0,514 contro 0,698; attività media 0,656 contro 1,861) **[C]**; passando a inizializzazione random condivisa il divario collassa ("At 5 steps, the loss difference between the naive random graph and the connectome is −0,0020, compared to +0,1841 under checkpoint initialization") **[C]**; passando al null degree-preserving anche il vantaggio di attività si inverte (−0,0106) **[C]**. Con un ensemble di 5 riconnessioni degree-preserving × 3 semi: loss a 5 passi 0,5155 ± 0,0067 per il connettoma contro 0,5172 ± 0,0061 per il controllo **[C]**. La prescrizione: "Treating null-model construction as an explicit part of the experimental design is therefore essential for isolating topology effects" **[C]**.

**Prima precisazione: il null deve essere quello giusto.** Un null "random a pari conteggi" non è un controllo valido — gonfia il vantaggio apparente del connettoma. Serve un null degree-e-peso-preservante, in ensemble (≥5 riconnessioni), con inizializzazione dei parametri da zero e identica fra le condizioni. Questo è rilevante perché il progetto ha già osservato che "ogni null batte il reale": è esattamente il regime che quel paper descrive, e significa che la domanda "il grafo fa lavoro?" ha già una risposta sfavorevole al grafo, indipendentemente dall'attention.

**Seconda precisazione: spegnere l'attention non è un controllo nullo, ma non è nemmeno inutile.** Nella letteratura ibrida l'ablazione dell'attention è usata regolarmente, ma sempre come *diagnostica di capacità specifica*, non come misura di "chi fa il lavoro": Jamba mostra che senza attention il modello non sviluppa in-context learning **[P]**, Mamba-2-Hybrid che senza attention MMLU crolla di 17 punti a 1,1T token e la copia resta "sfocata" **[P]**. Il limite metodologico è noto: togliere un componente non ne isola la funzione, perché cambia anche capacità e FLOP; per questo la pratica standard è affiancare confronti *matched-compute* e *isoFLOP* (MoR, Parcae, Loopie sono tutti valutati così) **[P]**. Un lavoro 2026 sostiene esplicitamente che le firme comportamentali e le ablazioni non bastano a identificare la struttura portante e che servono controlli a compute appaiato **[NV]** — segnalato perché pertinente, ma non ho potuto leggerne il testo, quindi non va citato come prova.

**Disegno sperimentale raccomandato.** Un fattoriale 2 × 2 a parità di architettura, parametri, FLOP di training, token e inizializzazione:

| | grafo reale | null degree-preserving (ensemble ≥5) |
|---|---|---|
| **1 lettura di attention per token** | condizione già misurata | controllo |
| **attention loopata (variante A o C)** | condizione proposta | controllo |

Le tre quantità da leggere: l'effetto principale del grafo (reale meno null, mediato sulle due righe) dice se la topologia fa lavoro; l'effetto principale delle letture dice quanto vale il looping; e soprattutto il **termine di interazione** dice se il grafo trae beneficio dall'attention ripetuta *più* di quanto ne tragga un grafo qualsiasi con lo stesso grado. Se l'interazione è nulla, l'attention ripetuta è un miglioramento architetturale generico e il connettoma resta ininfluente; se è positiva, c'è un argomento specifico per il connettoma. Aggiungere come quarta colonna un blocco Transformer denso a pari FLOP per ancorare la scala.

Da riportare per ogni cella: media e deviazione standard su almeno 3 semi, e la curva loss vs numero di letture (1, 2, 8) per vedere dove satura — è la curva che tutta la letteratura looped riporta e l'unica che rende confrontabile questo lavoro con Huginn, Ouro e Parcae.

---

## Fonti

Recurrent-depth e looped transformer:

- Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" (Huginn-3.5B), https://arxiv.org/abs/2502.05171 e https://arxiv.org/html/2502.05171v2 — **[C]** per adapter A: R^2h → R^h su concat(s_i, e), per la frase concat vs somma, per il budget di cache "i mod k"; **[P]** per saturazioni per compito e numeri MTBench.
- Saunshi et al., "Reasoning with Latent Thoughts: On the Power of Looped Transformers", https://arxiv.org/abs/2502.17416 e https://arxiv.org/html/2502.17416v1 — **[C]** per l'assenza di input injection e la definizione (L⊗T); **[P]** per i numeri di tabella e α_loop/α_base ≈ 1,19.
- Zhu et al., "Scaling Latent Reasoning via Looped Language Models" (Ouro), https://arxiv.org/abs/2510.25741 e https://arxiv.org/html/2510.25741v1 — **[C]** per le frasi su cache per passo, 4× memoria, >10 punti GSM8K; **[P]** per MMLU per T e i confronti con Qwen3.
- Bae et al., "Mixture-of-Recursions", https://arxiv.org/abs/2507.10524 e https://arxiv.org/html/2507.10524v1 — **[C]** per le definizioni di recursion-wise caching e recursive KV sharing e i valori NLL; **[P]** per isoFLOP, risparmi di memoria e attribuzione del valore 2,9163.
- Bae et al., "Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA", https://arxiv.org/abs/2410.20672 — **[P]**.
- Dehghani et al., "Universal Transformers", https://arxiv.org/abs/1807.03819 — **[P]** per equazioni, coordinate embedding e risultati; **[NV]** per il valore di perplexity LAMBADA estratto.
- "Looped Latent Attention: Cross-Loop KV Compression for Looped Transformers", https://arxiv.org/html/2607.15456v2 — **[C]** per la frase sui 96 layer di cache, la tabella a budget 4× e il crollo a 0,000 del riuso dell'ultimo loop.
- "Gated Recurrent Transformers: Expressive Depth through Recurrent Modulation", https://arxiv.org/html/2608.15062v3 — **[C]** per l'ablazione in nat; **[P]** per le tre strategie di compressione della cache.
- "Loop the Loopies!" (Loopie), https://arxiv.org/pdf/2607.16051 e https://arxiv.org/html/2607.16051v1 — **[C]** per la ricorrenza per-layer e "marginal benefit of recurrence is largest at R=2".
- "Parcae: Scaling Laws For Stable Looped Language Models", https://arxiv.org/html/2604.12946v1 — **[P]**.
- "A Mechanistic Analysis of Looped Reasoning Language Models", https://arxiv.org/html/2604.11791v1 — **[P]**.
- "Looped Transformers for Length Generalization", https://arxiv.org/pdf/2409.15647 — **[NV]** (PDF non leggibile dallo strumento; citato solo come puntatore alle ablazioni di input injection).
- "The Recurrent Transformer: Greater Effective Depth and Efficient Decoding", https://arxiv.org/html/2604.21215v1 — **[P]**.

Ibridi ricorrenza + attention:

- Hutchins et al., "Block-Recurrent Transformers", https://arxiv.org/abs/2203.07852 e https://ar5iv.labs.arxiv.org/html/2203.07852 — **[P]**.
- Fan et al., "Addressing Some Limitations of Transformers with Feedback Memory", https://arxiv.org/abs/2002.09402 — **[P]**.
- Glorioso et al., "Zamba: A Compact 7B SSM Hybrid Model", https://arxiv.org/pdf/2405.16712 — **[P]** per il blocco attention condiviso ogni 6 blocchi Mamba e la concatenazione degli embedding originali.
- De et al., "Griffin: Mixing Gated Linear Recurrences with Local Attention", https://arxiv.org/abs/2402.19427; RecurrentGemma, https://arxiv.org/abs/2404.07839 — **[P]**.
- Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model", https://arxiv.org/abs/2403.19887 — **[P]**.
- Waleffe et al., "An Empirical Study of Mamba-based Language Models", https://arxiv.org/html/2406.07887v1 — **[P]**.
- "SpikingBrain: Spiking Brain-inspired Large Models", https://arxiv.org/abs/2509.05276 e https://arxiv.org/html/2509.05276v1 — **[C]** per i rapporti 1:1 linear/SWA e 1:6 full attention; **[P]** per neurone adattivo e sparsità.
- "Rethinking the Role of Efficient Attention in Hybrid Architectures", https://arxiv.org/pdf/2606.15378 — **[NV]**: un riassunto di ricerca attribuiva a questo paper ablazioni su looping e rapporto 4:1; il controllo diretto sulla pagina smentisce, il paper non tratta looping.

Metodologia dei controlli:

- "Topological Sensitivity in Connectome-Constrained Neural Networks", https://arxiv.org/html/2604.04033v1 — **[C]** per null degree-preserving, inizializzazione condivisa, ensemble e numeri.
- "Induction Signatures Are Not Enough: A Matched-Compute Study of Load-Bearing Structure in In-Context Learning", https://arxiv.org/pdf/2509.22947 — **[NV]** (PDF non leggibile; citato solo come puntatore).
- ALBERT, https://arxiv.org/pdf/1909.11942 — **[P]** per la condivisione dei parametri di attention meno costosa di quella del feed-forward.

Stime di costo in FLOP per il caso connettoma: aritmetica propria sui parametri dichiarati del progetto, **[NV]**, da confermare con profiling.
