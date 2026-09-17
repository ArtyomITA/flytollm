# Looped Transformer / Recurrent-Depth / Recursive Transformer (2018-2026)

Ricerca sulle fonti primarie, finalizzata alla progettazione di una variante in cui il "blocco" ricorrente e' un grafo spiking (connettoma di mosca, 166.700 neuroni) e la multi-head attention viene portata *dentro* il giro anziche' essere letta una sola volta per token.

Marcatura delle affermazioni: **[C]** = confermato leggendo il testo pieno della fonte primaria (HTML/ar5iv del paper); **[P]** = parziale, ricavato dal solo abstract arXiv o da fonte secondaria; **[NV]** = non verificato.

---

## 1. Sintesi esecutiva

I looped transformer non sono una famiglia omogenea. Condividono una sola cosa: **gli stessi pesi vengono applicati piu' volte in profondita' sullo stesso token, dentro un singolo forward pass, prima di emettere il token successivo**. Tutto il resto varia, e le varianti sono esattamente i punti di progetto che contano per il caso FlyToLLM.

Le sei decisioni ricorrenti in letteratura, e la posizione dominante nel 2025-2026:

1. **Cosa sta dentro il loop.** Quasi sempre *attention + MLP insieme*, ossia uno o piu' layer transformer completi. Nessun lavoro rilevante mette solo l'MLP nel loop lasciando l'attention fuori: il caso "attention letta una volta sola" non ha praticamente precedenti nella letteratura looped. La sola eccezione strutturale sono i modelli ibridi (Jamba, Block-Recurrent), dove pero' l'alternanza e' *in profondita' sullo stack*, non *dentro un loop di pesi condivisi*.
2. **Input injection.** Il consenso 2025-2026 e' che l'embedding del token vada **ri-iniettato ad ogni iterazione**. Huginn lo concatena e lo passa in un adattatore; HRM e TRM lo sommano ad ogni passo; Yang 2023 lo somma e mostra che senza injection il modello degrada appena si superano i loop visti in training; Parcae 2026 dimostra che i parametri di injection sono *proprio* la sorgente dell'instabilita' e li vincola spettralmente; l'analisi Jacobiana 2026 mostra che e' l'injection, non lo stato ricorrente, a tenere in vita l'informazione. I lavori che *non* iniettano (Saunshi 2025, ALBERT, Loop-Think-Generalize) funzionano su task sintetici o su profondita' modeste.
3. **KV cache per iterazione.** Tre regimi distinti, tutti documentati con numeri (dettaglio in sezione 4.3). Il default e' **una cache separata per ogni step di ricorrenza**: al giro *r* l'attention legge gli stati dei token precedenti *allo stesso giro r*. Ouro misura che riusare la cache di un altro giro in prefill costa oltre 10 punti su GSM8K.
4. **Numero di loop.** Chi vuole scalare a test-time **randomizza il numero di loop in training** (Huginn: log-normale-Poisson con media 32; Parcae: media campionata in {2,4,6,8,10,12}; Ouro: prior uniforme su 1..4 con regolarizzazione entropica). Chi usa un numero fisso ottiene estrapolazione fragile.
5. **Come si allena.** Tre scuole: backprop completo su tutti i loop (Saunshi, Fan, MoR, TRM), backprop troncato agli ultimi *k* (Huginn: k=8), gradiente a un passo in stile deep-equilibrium (HRM). TRM mostra con un'ablazione diretta che il gradiente a un passo costa 31 punti su Sudoku rispetto al backprop completo.
6. **Stabilizzazione.** Sandwich norm, pesi separati per prelude e coda, controllo spettrale sull'injection, damping dei sotto-passi. Senza queste, i modelli looped esplodono o collassano quando si aumenta la profondita'.

**Perche' il loop aiuta**, secondo gli autori: (a) profondita' effettiva senza parametri, con la profondita' che e' la risorsa vera per i problemi iterativi; (b) "pensieri latenti" equivalenti a passi di chain-of-thought ma senza token emessi; (c) bias induttivo verso algoritmi iterativi e punti fissi; (d) *manipolazione* della conoscenza migliore a parita' di *capacita'* di conoscenza. L'analisi meccanicistica 2026 ridimensiona la retorica: le iterazioni **non si specializzano**, ripetono le stesse operazioni, e i pattern di attention si stabilizzano gia' dopo la prima iterazione.

---

## 2. Schede per lavoro

### 2.1 Universal Transformer — Dehghani et al., 2018 (arXiv 1807.03819)

**Dentro il loop** [C]: un layer transformer completo — multi-head self-attention seguita da una funzione di transizione (convoluzione separabile oppure fully-connected position-wise con ReLU). I pesi sono condivisi sia sui passi temporali di profondita' sia sulle posizioni.

**Update** [C]:
- `A^t = LayerNorm((H^{t-1} + P^t) + MultiHeadSelfAttention(H^{t-1} + P^t))`
- `H^t = LayerNorm(A^t + Transition(A^t))`

**Input injection** [C]: l'embedding originale **non** viene ri-sommato. Quello che viene aggiunto ad ogni passo e' un *coordinate embedding* `P^t` bidimensionale (posizione + timestep), sinusoidale, sommato:
`P^t_{i,2j} = sin(i/10000^{2j/d}) + sin(t/10000^{2j/d})`, e analogo con coseno per gli indici dispari. E' quindi un'iniezione di *indice di iterazione*, non di contenuto del token.

**KV per iterazione** [C]: al passo *t* la self-attention legge `H^{t-1}`, cioe' gli stati di tutte le posizioni **allo stesso livello di ricorrenza precedente**. E' il modello "parallel-in-time": tutte le posizioni avanzano di un passo insieme. Questo e' il regime che poi diventa standard.

**Halting** [C]: ACT per-posizione. Una densa con sigmoide produce la probabilita' di halting; quando una posizione si ferma, **il suo stato viene semplicemente copiato in avanti** ai passi successivi. Esiste un numero massimo di passi, variabile per esperimento.

**Training** [C]: backprop su tutti i passi ricorrenti. Profondita' fisse 6-9 per LAMBADA, variabile su bAbI.

**Risultati** [C]: bAbI 10K joint 0,29% di errore; subject-verb agreement 99,2% con ACT; copy/reverse/addition 91%/96%/34% di accuratezza a carattere; WMT14 En-De **28,9 BLEU** (+0,9 sul Transformer) [P sul delta, dall'abstract]; LAMBADA perplessita' 142.

**Perche' aiuta** [C]: il transformer standard ha un numero di operazioni sequenziali costante e indipendente dalla lunghezza dell'input; rendendo il numero di passi funzione della lunghezza si ottiene la Turing-completezza (sotto ipotesi).

### 2.2 ALBERT — Lan et al., 2019 (arXiv 1909.11942)

Non e' un looped transformer adattivo, ma e' il precedente canonico del weight sharing in profondita'.

**Cosa si condivide** [C]: l'ablazione distingue all-shared, solo-attention, solo-FFN, non-shared. Su ALBERT-base con E=128: all-shared 12M parametri e media 80,1; shared-attention 64M e 81,7; shared-FFN 38M e 80,2; non-shared 89M e 81,6. **Il calo viene quasi tutto dalla condivisione dell'FFN; condividere i pesi dell'attention non costa nulla a E=128.** Questo dato e' direttamente rilevante: nel caso FlyToLLM l'oggetto da riusare a ogni giro sarebbe proprio l'attention.

**Parametri** [C]: BERT-base 108M vs ALBERT-base 12M; BERT-large 334M vs ALBERT-large 18M (24 layer); ALBERT-xlarge 60M; ALBERT-xxlarge 235M (12 layer, hidden 4096).

**Input injection** [C]: nessuna. E' pura composizione di funzioni a profondita' fissa pari al numero di layer, senza halting adattivo.

**Convergenza** [C]: le distanze L2 e le similarita' coseno fra input e output di ogni layer sono molto piu' *lisce* in ALBERT che in BERT, ma **non convergono a zero nemmeno dopo 24 layer**. Gli autori lo contrappongono esplicitamente alla predizione dei Deep Equilibrium Model: lo spazio di soluzioni trovato da ALBERT oscilla, non si assesta su un punto fisso.

**Risultati** [C]: GLUE ensemble 89,4; SQuAD 2.0 test 92,2 F1; RACE test 89,4%.

### 2.3 Looped Transformers as Programmable Computers — Giannou et al., 2023 (arXiv 2301.13196)

**Pesi costruiti a mano, non addestrati** [P]. Il contributo e' di espressivita': un transformer di **13 layer** messo in loop emula un calcolatore general-purpose. La sequenza in input funge da "scheda perforata" contenente sia istruzioni sia memoria per letture/scritture; il loop emula program counter e salti condizionati. Emulano una calcolatrice base, una libreria di algebra lineare di base e algoritmi di in-context learning che usano backpropagation. [P] — estratto dall'abstract; il numero di iterazioni del loop non e' fissato.

Rilevanza per il progetto: stabilisce che **il loop con pesi condivisi e' computazionalmente universale**, ma non dice nulla su come addestrarlo.

### 2.4 CoTFormer — Mohtashami, Pagliardini, Jaggi, 2023 (arXiv 2310.10845)

Questo e' il lavoro piu' direttamente rilevante alla domanda "cosa vede l'attention ai giri successivi", perche' e' l'unico che rompe deliberatamente il regime parallel-in-time.

**Meccanica** [C]: ad ogni ripetizione la rappresentazione intermedia viene **appesa alla sequenza**, cosi' che alla ripetizione *i+1* l'attention del token corrente vede *tutte* le rappresentazioni precedenti, da tutte le profondita':

`x_{n+1}^{(i+1)} := B( x_{n+1}^{(i)} , [ x_{1:n}^{(i)}, ..., x_{1:n}^{(0)} ] )`

La sequenza cresce linearmente: dopo `n_repeat` applicazioni il modello vede `n_repeat` copie della sequenza originale, impilate cronologicamente.

**Il confronto esplicito** [C] con il Block Universal Transformer, che e' il regime standard:

`x_{n+1}^{(i+1)} := B( x_{n+1}^{(i)} , x_{1:n}^{(i)} )`  — solo il giro corrente e' visibile.

**La "sottigliezza"** [C]: nel chain-of-thought vero i thought token generati possono attendere direttamente ai thought token precedenti; un universal transformer che ri-applica il blocco *non puo'*, perche' vede solo le rappresentazioni dei token precedenti alla stessa profondita'. CoTFormer ripristina questa capacita'.

**Profondita' adattiva** [C]: "Mixture of Repeats". Per ogni ripetizione *i* c'e' un embedding appreso `e^(i)`; il punteggio e' `s_j^(i) = sigma( e^(i)ᵀ x_j^(i) )`; si ordinano i token e si selezionano i top `k = floor(c_i · n_seq)`. In training le capacita' `c_i` sono campionate casualmente per batch (la prima ripetizione e' fissata a 1,0). Il blending e' interpolato: `x^(i+1) := (1-s_i)·x_i + s_i·B(x^(i))`. Viene aggiunto anche un depth embedding `(n_repeat - i)·e^(depth)`.

**Risultati** [C], OpenWebText2, hidden 768, 12 teste, seq 256: baseline standard a 48 layer perplessita' 24,17; CoTFormer 24x5 ripetizioni 24,48; Block Universal Transformer 24x5 24,85; LN-CoTFormer 24,11 (batte il 48-layer). CoTFormer adattivo 23,83 a budget pieno, con riduzione del 20-30% di calcolo a perdita trascurabile.

### 2.5 Looped Transformers are Better at Learning Learning Algorithms — Yang et al., 2023 (arXiv 2311.12424)

**Dentro il loop** [C]: **un solo layer** (L=1), contro il baseline standard a 12 layer.

**Input injection** [C], formula esplicita: `Y_{t+1} = M(Y_t + P)`, dove `P` e' il prompt in input e `M` il transformer. Gli autori scrivono che senza injection la prestazione "si deteriora oltre le iterazioni viste in training". E' **somma**, non concatenazione.

**Training** [C]: loss mediata su una finestra di iterazioni, non solo sull'ultima:

`min_theta E_P [ 1/(b-b0) · sum_{t=b0..b} 1/(k+1) · sum_{i=0..k} loss( Y_t(P^i|theta), f(x_{i+1}) ) ]`  con `b0 = max(b-T, 0)`

E' backpropagation troncata: `T` e' la finestra. Curriculum: `b` cresce progressivamente e la finestra scivola da [1,T] a [b-T, b]. Valori usati: regressione lineare b=20, T=15; regressione sparsa b=20, T=10; decision tree b=70, T=15; rete ReLU a 2 layer b=12, T=5.

**Test-time** [C]: il modello continua oltre i loop di training e raggiunge una soluzione di punto fisso stabile.

**Risultati** [C]: looped 0,79M parametri contro 9,48M dello standard a 12 layer (circa 1/12). Sulla regressione lineare eguaglia lo standard e si allinea al risolutore ai minimi quadrati; sulla regressione sparsa lo batte (errore 6,12e-04 contro 0,0017 con 40 campioni). Limite dichiarato: prestazioni scarse su prompt fuori distribuzione — impara l'algoritmo solo dentro la distribuzione di training.

**Perche' aiuta** [C]: la discesa del gradiente richiede solo moltiplicazioni di matrici applicate ripetutamente, mentre la soluzione in forma chiusa richiede un'inversione, che e' difficile per un transformer. Il loop emula l'iterazione di punto fisso.

### 2.6 Looped Transformers for Length Generalization — Fan et al., 2024 (arXiv 2409.15647)

**Dentro il loop** [C]: un singolo blocco decoder condiviso (piu' layer al suo interno), con attention causale e MLP, riusato per tutti i `T` passi.

**Input injection** [C]: "gli embedding di input vengono sommati agli embedding di output del passo precedente come input del passo corrente". Somma, ad ogni passo.

**Numero di passi** [C]: in training si usa il `T(n)` ground-truth derivato dalla struttura n-RASP-L del problema (per esempio `T(n)=n` per parity e copy, `T(n)=n+1` per l'addizione). A test due criteri: **oracle** (usa il `T(n)` predefinito per la lunghezza di test) e **maximum confidence** (sceglie il passo che minimizza la cross-entropy sul batch).

**Supervisione** [C]: **solo la risposta finale**. Non c'e' supervisione sugli stati intermedi. Il segnale che rende il blocco generalizzabile e' che esempi diversi richiedono conteggi di passi diversi, quindi lo stesso blocco riceve segnali eterogenei. Backprop completo su tutti i `T(n)` passi.

**Risultati** [C], addestrati su lunghezze 1-20: parity accurato a lunghezza 40; copy a 35; addizione a 30; binary sum a 24; unique set a 35; moltiplicazione addestrata 1-12 e generalizzata a 16. NTP autoregressivo fallisce, i pause token fanno meglio dell'NTP ma peggio del loop. NTP-Loop a 20 passi fissi migliora su NTP ma resta sotto il loop adattivo.

### 2.7 Relaxed Recursive Transformers — Bae et al., 2024 (arXiv 2410.20672)

**Costruzione** [C]: un modello pre-addestrato a N layer viene convertito in N/B blocchi di K layer condivisi, ciclati B volte (strategia CYCLE). Esempio: Gemma 2B a 18 layer diventa 2 blocchi da 9 layer condivisi ciclati due volte. Tre inizializzazioni: **Stepwise** (seleziona layer intermedi a intervalli, migliore per i recursive puri), **Average** (media i pesi dei layer legati, migliore per i relaxed), **Lower**.

**Rilassamento** [C]: LoRA per-layer che rompe il legame esatto:
`h_t^l = f( h_t^{l-1} ; Phi'_{((l-1) mod L/B)+1} , DeltaPhi'_l )`
Le LoRA sono inizializzate con SVD troncata sulle matrici residue fra pesi originali e pesi legati. A rango pieno il modello relaxed coincide esattamente con il pre-addestrato.

**Input injection** [C]: **assente**. Gli stati nascosti scorrono sequenzialmente.

**Inferenza** [C]: "Continuous Depth-wise Batching" — sfruttando i pesi condivisi si schedulano nello stesso batch campioni a profondita' diverse; con early-exiting i posti liberati si riempiono subito. Speedup dichiarato 2-3x con oracle-exiting, fino a quasi 4x teorico su Gemma ricorsivo.

**Risultati** [C]: uptraining su SlimPajama, 15B e 60B token, con distillazione KL forward da teacher full-size. Recursive Gemma 1B (0,99B) con Stepwise: 51,7% media few-shot, +13,5 punti percentuali rispetto ai baseline a dimensione ridotta. Relaxed Gemma con rango 512: 58,4%, praticamente pari al Gemma originale addestrato su 3T token (58,6%). Il passaggio da 15B a 60B token di uptraining vale +4,1 punti.

### 2.8 Huginn-3.5B — Geiping et al., 2025 (arXiv 2502.05171) — *il riferimento architetturale principale*

**Struttura** [C]: tripletta `(l_P, l_R, l_C)` = **prelude, core ricorrente, coda**. Il modello grande e' **(2, 4, 2)** con hidden 5280, 55 teste da 96, MLP interno 17920. Ogni layer contiene self-attention causale con RoPE (base 50000), MLP SiLU gated, RMSNorm. **Prelude e coda hanno pesi propri, non condivisi**; solo i 4 layer del core sono ripetuti.

**Input injection** [C], il punto chiave: l'embedding `e` prodotto dal prelude viene **concatenato** allo stato ricorrente e passato in una matrice adattatore:
`A : R^{2h} -> R^h`, `s_i = R(e, s_{i-1})`
Il blocco core comincia con l'adattatore che mappa la concatenazione di `s_i` ed `e` nella dimensione `h`. **Alla scala usata, la concatenazione batte la somma.** L'iniezione avviene ad *ogni* passo di ricorrenza, non solo all'inizio.

**KV cache** [C]: di default ogni step di ricorrenza ha la propria cache, e al giro *r* l'attention legge gli stati dei token precedenti **allo stesso giro r**. In piu' gli autori mostrano un **KV-cache sharing zero-shot**: si fissa un budget `k` per token e al giro `i` si legge e scrive l'entry `i mod k`. Funziona perche' tutte le entry ricorrenti sono prodotte dalle stesse proiezioni K,V applicate a stati successivi, quindi sono commensurabili: un token puo' attendere all'ultima entry in cache di ogni token precedente, indipendentemente dalla profondita' ricorrente effettivamente spesa su quel token.

**Numero di loop** [C]: in training `r` e' campionato per sequenza da una **log-normale-Poisson**:
`tau ~ N( log(r_medio) - sigma²/2 , sigma )` con `sigma = 1/2`; `r ~ Poisson(e^tau) + 1`, con `r_medio = 32`.
A test valutano r = 1, 4, 8, 16, 32, 64, 128.

**Halting** [C]: zero-shot, senza training dedicato. Si calcola la **divergenza KL fra le distribuzioni di output di due passi consecutivi**; se scende sotto **5x10⁻⁴** si smette di iterare, si campiona il token e si passa al successivo.

**Training** [C]: **backprop troncata agli ultimi k = 8 passi** del core. Il prelude riceve gradiente ad ogni passo perche' il suo output e' iniettato ovunque. Memoria e calcolo del backward indipendenti da `r`.

**Stabilizzazione** [C]: normalizzazione **sandwich** (una norm prima e una dopo ciascun sotto-blocco):
`x_hat_l = n2( x_{l-1} + Attn( n1(x_{l-1}) ) )`
`x_l = n4( x_hat_l + MLP( n3(x_hat_l) ) )`
RMSNorm; la versione senza parametri **fallisce alla scala**. Inizializzazione alla Takase: `sigma_h² = 2/(5h)`; proiezioni di output `sigma_out² = 1/(5·h·l)` con `l = l_P + r_medio·l_R + l_C = 132`; stato iniziale casuale `s_0` con `sigma_s² = 2/5` (normale troncata). Scala `gamma` sull'embedding. Learning rate 5x10⁻⁴, critico: valori piu' alti causano collasso dello stato.

**Risultati** [C]: 3,5B parametri (1,5B prelude+coda, 1,5B core, 0,5B embedding), 800B token, 4096 GPU AMD MI250X. A r=32: GSM8k CoT 34,80%/42,08% (flexible/strict), MATH Minerva 12,58%, HumanEval 23,17% pass@1, ARC-C 38,23%. A 180B token il ricorrente fa 9,02% su GSM8k CoT contro 1,82% del baseline, circa 5x.

**Perche' aiuta e cosa succede davvero** [C]: il ragionamento latente non richiede dati appositi; il modello fa piu' FLOP per parametro, riducendo i costi di comunicazione. Nell'analisi dello spazio latente emergono **tre regimi di traiettoria** senza che siano stati addestrati: convergenza a **punti fissi** (task semplici), **orbite** (token aritmetici e di deliberazione — meccanismi periodici per contare o raffinare), **slider** (deriva direzionale, forse un contatore di iterazioni). Path independence: partendo da inizializzazioni diverse le traiettorie convergono a pattern simili. La velocita' di convergenza dipende dal contesto: su ARC-C il modello satura a 8-12 iterazioni senza esempi, ma usa 32 iterazioni con 25-50 esempi few-shot.

### 2.9 Reasoning with Latent Thoughts — Saunshi et al., ICLR 2025 (arXiv 2502.17416)

**Dentro il loop** [C]: `k` layer transformer completi, notazione `(k ⊗ L)` = blocco di k layer ripetuto L volte. Definizione formale dal paper: `p_{theta,T} = OUTPUT ∘ (Tb_theta)^T ∘ EMBED`. Configurazioni testate k ∈ {1,2,3,4,6,8,12}.

**Input injection** [C]: **nessuna**. E' pura composizione di funzioni. Questo e' importante: il lavoro che piu' sostiene la tesi "il loop e' ragionamento latente" e' anche quello che *non* usa injection, e infatti opera su task sintetici e su modelli fino a 1B.

**Loss** [C]: solo l'output dell'ultimo loop. Gli output intermedi non sono supervisionati direttamente.

**Regolarizzazione proposta** [C]: non impone il weight sharing ma lo incoraggia via coseno fra blocchi successivi:
`L = L_xent + lambda_reg · |G|⁻¹ · sum_G R_G(k)`,
con `R_G(k) = 1/(L-k) · sum_{i=0}^{L/k-2} sum_{j=0}^{k-1} Cosine( theta_G^{(ik+j)} , theta_G^{((i+1)k+j)} )`. Testata con lambda_reg ∈ {1, 10}.

**Training** [C]: backprop su tutti gli L loop, nessun troncamento. L fisso per esperimento. Adafactor, batch 512-1024, lr 0,01 (LM) o 0,005 (aritmetica).

**Risultati** [C]:
- Addizione n-aria: `(1⊗12)` 99,9% (n=8), 100,0% (n=16), 99,9% (n=24), 99,6% (n=32); baseline `(12⊗1)` 100% ovunque; `(2⊗6)` 100,0/99,8/99,7/99,5.
- p-hop induction: `(1⊗6)` 99,9% (p=16), 99,5% (p=32) contro baseline `(6⊗1)` 99,9% e 99,6%.
- i-GSM: `(1⊗8)` 73,2%, identico al baseline `(8⊗1)` 73,2%; `(2⊗4)` 73,6%.
- Language modeling 1B su Pile: perplessita' di validazione baseline `(24⊗1)` **7,40**, looped `(12⊗2)` **7,90** (iso-FLOP, quindi *peggiore* in perplessita'). Ma: math word problems `(12⊗2)` **34,3%** contro **29,3%** del baseline; reasoning primitives **51,2%** contro **47,5%**.
- La metrica decisiva: **il modello looped copre il 34-50% del gap di perplessita' ma il 78-282% del gap di ragionamento.** E' la formulazione quantitativa della tesi "il loop compra ragionamento, non memorizzazione".

**Teoria** [C]: Teorema 5.1, un looped a 1 layer con `T = ceil(log2 n)` loop risolve la composizione di gruppo su n elementi, eguagliando il lower bound di profondita'. Teorema 5.2, un transformer non-looped a L layer con R layer distinti si simula con un looped a L loop e dimensione di embedding `d + R + 2`. Corollario 5.3, p-hop con `floor(log2 p) + 2` loop su un looped a 1 layer. **Teorema 5.4, m passi di chain-of-thought di un transformer a L layer si simulano con un looped a L + O(1) layer e m loop.**

**Middle looping** [C]: variante che mantiene indipendenti i primi e ultimi k layer e cicla solo il blocco centrale. Da perplessita' migliore del looping dell'intero modello, mantenendo i benefici di ragionamento.

### 2.10 Hierarchical Reasoning Model (HRM) — Wang et al., 2025 (arXiv 2506.21734) — *il piu' vicino all'architettura attuale di FlyToLLM*

Rilevante perche' e' **esattamente** il pattern "N sotto-passi di un core, poi un aggiornamento a scala piu' lenta".

**Struttura** [C]: due moduli ricorrenti, entrambi **blocchi transformer encoder-only** con RoPE, GLU, RMSNorm, post-norm, init LeCun Normal:
- `z_L^i = f_L( z_L^{i-1} , z_H^{i-1} , x_tilde ; theta_L )`
- `z_H^i = f_H( z_H^{i-1} , z_L^{i-1} ; theta_H )` se `i ≡ 0 (mod T)`, altrimenti `z_H^i = z_H^{i-1}`

Il modello esegue `N` cicli alti da `T` passi bassi ciascuno. Negli pseudocodici N=2, T=2.

**Input injection** [C]: `x_tilde = f_I(x; theta_I)` viene iniettato **ad ogni singolo timestep** del modulo L, per somma elemento a elemento. Gli autori annotano esplicitamente che "meccanismi di gating potrebbero migliorare".

**Training** [C]: gradiente a un passo, giustificato dal teorema della funzione implicita, con `(I - J_F)⁻¹ ≈ I`. Il gradiente fluisce solo attraverso gli stati finali: testa di output -> stato finale di H -> stato finale di L -> embedding di input. Memoria O(1) contro O(T) del BPTT. Deep supervision: si eseguono M segmenti, ad ogni segmento si calcola la loss, si aggiorna e si **stacca** (detach) lo stato dal grafo, cosi' i gradienti del segmento m+1 non tornano nel segmento m.

**Halting** [C]: Q-head che predice `Q_hat^m = (Q_halt, Q_continue)` con `sigma(theta_Q^T z_H^{mNT})`. Si ferma se il conteggio dei segmenti raggiunge `M_max` (fino a 16) oppure se `Q_halt > Q_continue` e il conteggio supera `M_min`. `M_min` e' campionato uniformemente in {2..M_max} con probabilita' epsilon, altrimenti vale 1. Loss combinata: `L_ACT^m = Loss(y_hat^m, y) + BCE(Q_hat^m, G_hat^m)`.

**Risultati** [C]: 27M parametri, circa 1000 campioni di training, nessun pre-training, nessuna etichetta CoT, init casuale. ARC-AGI-1 **40,3%** (sopra o3-mini-high 34,5% e Claude 3.7 21,2%); Sudoku-Extreme quasi perfetto; Maze-Hard 30x30 quasi perfetto contro 0% dei baseline CoT.

**Perche' la gerarchia serve** [C] — il punto piu' importante per un core ricorrente: *"Man mano che lo stato nascosto si assesta verso un punto fisso, l'ampiezza degli aggiornamenti si riduce, di fatto bloccando la computazione successiva e limitando la profondita' effettiva della rete."* La soluzione HRM: dentro ogni ciclo il modulo L converge a un equilibrio locale condizionato da `z_H`; dopo T passi il modulo H si aggiorna e fornisce **contesto fresco** che "resetta" L verso un equilibrio diverso. Il risultato e' una sequenza di computazioni annidate stabili e distinte, con profondita' effettiva N×T invece di T. La Figura 3 mostra convergenza gerarchica: H converge stabilmente, L converge ripetutamente dentro ogni ciclo prima di essere resettato.

### 2.11 Mixture-of-Recursions (MoR) — Bae et al., 2025 (arXiv 2507.10524)

**Struttura** [C]: blocco condiviso `Phi'` riusato su `N_r` step. Strategie: Cycle, Sequence, **Middle-Cycle** (la migliore: mantiene a capacita' piena il primo e l'ultimo layer, condivide gli intermedi — stessa lezione di Huginn e del middle-looping di Saunshi). Un modello vanilla da 315M diventa circa 118M con N_r=3.

**Router** [C]: due modalita'. *Expert-choice*: ogni step di ricorsione e' un esperto che seleziona i top-k token, con punteggio `g_t^r = G(theta_r^T H_t^r)` e soglia percentile `g_t^r > P_beta(G^r)`; filtro gerarchico, solo i token selezionati allo step r sono rivalutati a r+1. *Token-choice*: una sola decisione all'inizio, `i = argmax_j g_t^j`, il token riceve `i` applicazioni sequenziali. Residuo expert-choice: `H_t^{r+1} = g_t^r · f(H_t^r, Phi') + H_t^r`.

**KV cache** [C], due strategie esplicite e quantificate — il chiarimento piu' netto in letteratura su questo asse:
- *Recursion-wise caching*: solo i token instradati a un dato step memorizzano K/V a quel livello, e **l'attention e' ristretta ai token cacheati localmente a quel livello**. Memoria KV ridotta a circa `(N_r+1)/(2·N_r)` del vanilla; FLOP di attention ridotti di un fattore `(k/N_ctx)²`.
- *Recursive KV sharing*: tutti i token attraversano il primo blocco; K/V sono cacheati **solo allo step 1** e riusati su tutti gli `N_r` step. Memoria circa `1/N_r`, ma i FLOP di attention scendono solo di `k/N_ctx`.

**Training** [C]: backprop su tutte le N_r ricorsioni. Loss ausiliaria per expert-choice (mitiga la violazione di causalita' meglio di un router ausiliario separato); balancing loss per token-choice.

**Risultati** [C]: a FLOP fissi 16,5e18 — vanilla 315M: NLL 2,7824, few-shot 42,3%, 20B token; recursive 118M: 2,8466 e 41,5%, 19B token; MoR 118M: 2,7925 e **42,6%**, 30B token. Con expert-choice MoR arriva a **43,1% contro 42,3%** del vanilla usando meta' dei parametri. A 20B token fissi, MoR con N_r=2 riduce i FLOP di training del 25%. Throughput: MoR-4 fino a **2,06x**. IsoFLOP da 135M a 1,7B: da 360M in su MoR eguaglia o supera il vanilla con 1/3 dei parametri unici.

### 2.12 Less is More: Tiny Recursive Model (TRM) — Jolicoeur-Martineau, 2025 (arXiv 2510.04871)

**Ricorsione** [C]:
- `z <- net(x, y, z)` ripetuto `n` volte
- `y <- net(y, z)` una volta per ciclo

Valori: n=6, T=3, N_sup=16 passi massimi di supervisione.

**Input injection** [C]: `x` compare **in ogni** aggiornamento latente. L'aggiornamento della risposta `y <- net(y, z)` invece esclude `x`, per permettere un raffinamento indipendente dell'output.

**Architettura** [C]: reti a **2 layer**. Versione con self-attention per ARC-AGI (griglie 30x30); versione **MLP-Mixer senza attention** per Sudoku 9x9, che porta da 74,7% a **87,4%**. La regola dichiarata: quando la lunghezza di contesto `L <= D` (dimensione di embedding) un layer lineare e' piu' conveniente dell'attention.

**Training** [C]: **backprop completo** su tutta la ricorsione (n+1 passi), eliminando l'approssimazione a un passo di HRM. Ablazione diretta: con gradiente a un passo Sudoku scende a **56,5%** contro **87,4%** con backprop completo — **31 punti**. ACT semplificato a una singola BCE, eliminando il secondo forward pass.

**Risultati** [C]: ARC-AGI-1 **45%** con 7M parametri (HRM 40,3% con 27M); ARC-AGI-2 8% (HRM 5,0%); Sudoku-Extreme 87,4% con 5M (HRM 55%); Maze-Hard 85,3% (HRM 74,5%).

**Perche'** [C] — onesta ammissione degli autori: *"la questione del perche' la ricorsione aiuti cosi' tanto rispetto a una rete piu' grande e profonda resta da spiegare; sospettiamo abbia a che fare con l'overfitting, ma non abbiamo una teoria."* Su 2 layer contro 4: con 1000 esempi le reti piu' grandi fanno overfitting; 2 layer con `n` aumentato mantengono profondita' effettiva equivalente (circa 42 layer) dimezzando i parametri e migliorando la generalizzazione.

### 2.13 Ouro / LoopLM — ByteDance et al., 2025 (arXiv 2510.25741)

Il primo looped LM pre-addestrato su scala industriale.

**Struttura** [C]: `F^(t) = lmhead ∘ H^L ∘ ... ∘ H^L (t volte) ∘ emb`. Blocchi transformer identici con MHA + SwiGLU, **sandwich normalization** (RMSNorm prima di attention e FFN), RoPE. Ouro-1.4B: 24 layer, hidden 2048, 4 step ricorrenti. Ouro-2.6B: 48 layer, hidden 2048, 4 step. **Nessun prelude/coda separato dichiarato.**

**Input injection** [C, negativo]: nel paper **non compare** un'equazione di re-iniezione dell'embedding. L'evoluzione e' `H^(t) = TransformerLayer_theta( H^(t-1) )`, con `H^(0)` l'embedding iniziale. E' una differenza sostanziale rispetto a Huginn.

**KV cache** [C], il dato piu' utile sull'asse KV: in **prefill** tutti e quattro gli step ricorrenti richiedono **la propria cache** — circa 4x di overhead di memoria — perche' *"ogni step trasforma le rappresentazioni in modi che non possono essere approssimati dagli step precedenti"*. Provare a riusarla degrada **oltre 10 punti su GSM8K**. In **decoding** il riuso diventa praticabile: testano last-step reuse e first-step reuse.

**Halting** [C]: gate appreso, `lambda_t(x) = sigma( Gate( F^(t)(x) ) )` = probabilita' di uscire allo step t. Criterio Q-exit basato su CDF: `min{ t : CDF(t|x) >= q }` con `CDF(t|x) = sum_i lambda_i · prod_{j<i} (1 - lambda_j)`.

**Obiettivo entropico** [C]: `L = sum_t q_phi(t|x) · L^(t) - beta · H( q_phi(·|x) )` con **prior uniforme** `pi_t = 1/T_max` invece di un prior geometrico, per esplorare le profondita' senza bias. La loss e' quindi **l'attesa su tutti gli step**, non solo l'ultimo. Stage II addestra i gate sul miglioramento per-token: `I_i^(t) = max(0, L_{i,stop}^{(t-1)} - L_{i,stop}^{(t)})`.

**Pretraining** [C]: 7,7T token. Stage 1a: 3T token con **8 step ricorrenti — instabile**; Stage 1b: 3T token con **4 step — stabilizzato**; Stage 2 annealing 1,4T token; Stage 3 LongCT 20B token a 64K; Stage 4 mid-training 300B token. Stabilizzazione: riduzione da 8 a 4 step per problemi di flusso del gradiente, batch da 4M a 8M token, beta da 0,1 a 0,05 per ridurre i gradienti in conflitto, learning rate conservativi.

**Risultati** [C]: Ouro-1.4B vs Qwen3-4B — MMLU 67,35 vs 73,19; BBH 71,02 vs 70,95; GSM8K **78,92 vs 72,86**; MATH500 **82,40 vs 59,60**. Ouro-2.6B vs Qwen3-8B — MMLU 74,60 vs 76,63; BBH **80,46 vs 77,65**; GSM8K 81,58 vs 83,09; MATH500 **90,85 vs 62,30**. Ouro-Thinking R4: AIME2024 pass@1 65% (1,4B) e 64,7% (2,6B) contro 73% di Qwen3-8B; OlympiadBench 71,55% e 76,44% contro 75,25%.

**Perche' aiuta** [C]: **il loop non aumenta la capacita' di conoscenza ma la manipolazione**. Su un task sintetico di biografie sia looped sia non-looped raggiungono circa **2 bit per parametro** — identici. Il vantaggio appare su QA multi-hop e composizione di fatti.

### 2.14 Think-at-Hard (TaH) — 2025 (arXiv 2511.08577)

**Selezione** [C]: un "decider" leggero (MLP che legge gli stati nascosti concatenati di layer basso, medio e finale) decide quali token ricevono iterazioni extra. Salta l'iterazione su circa il **93%** dei token.

**Duo-causal attention** [C] — la formalizzazione piu' pulita del "cosa vede l'attention ai giri successivi": alla profondita' `d`, il token in posizione `i` accede a
`X_{<=i}^{(<=d)} = { x_j^{(k)} | j <= i , k <= d }`
cioe' **causale sia sull'asse sequenza sia sull'asse profondita'**. Maschera 2D: attention permessa se e solo se `j <= i` e `k <= d`. La cache visibile si costruisce concatenando tutte le profondita' da 1 a d lungo l'asse sequenza: `KV^{(<=d)} = [ KV^{(1)} ; KV^{(2)} ; ... ; KV^{(d)} ]`.

**Iterazioni** [C]: `d_max = 2` negli esperimenti principali, variante TaH-3 con profondita' 3. La LoRA depth-aware si applica **solo alle iterazioni d>1**, lasciando il modello base intatto al primo passo.

**Risultati** [C], modello 1,7B: GSM8K 82,1% -> 84,5%; MATH500 68,4% -> 74,4%; AIME25 13,3% -> 17,9%. Media +5,0% sul baseline standard, +8,1-11,3% su "AlwaysThink" (iterare su tutti i token). Meno del 3% di parametri aggiuntivi.

**Fenomeno documentato** [C]: **"latent overthinking"** — predizioni corrette al primo passo vengono talvolta *rovinate* dalle iterazioni successive. E' la ragione stessa dell'iterazione selettiva.

---

## 3. Il 2026

### 3.1 Parcae: Scaling Laws For Stable Looped Language Models (arXiv 2604.12946)

Il lavoro piu' importante del 2026 per chi deve *progettare* un loop, perche' identifica il punto di rottura e lo chiude.

**Diagnosi** [C]: i looped soffrono di **esplosione del residuo e loss spike**, attribuiti tramite analisi dei sistemi dinamici a **grandi norme spettrali nei parametri di injection**.

**Ricorrenza** [C]: `h_{t+1} = A_bar · h_t + B_bar · e + R_bar( h_t , e )`, con `e = LN( P(s) )` l'embedding normalizzato. L'injection lineare e' o somma o concatenazione con proiezione: `h_{t+1} = R( W1·h_t + W2·e )`.

**Il fix** [C]: **parametrizzazione diagonale negativa**
`A := Diag( -exp( log_A ) )` con `log_A` apprendibile,
discretizzata con zero-order hold `A_bar = exp( Delta ⊙ A )`, che garantisce **raggio spettrale `rho(A_bar) < 1`**, condizione di stabilita' per un sistema LTI discreto. E' letteralmente la parametrizzazione dei moderni SSM applicata all'asse della profondita'.

**Setup** [C]: profondita' di training `mu_rec ∈ {2,4,6,8,10,12}` campionata per sequenza; profondita' di test fino a T=24; modelli 100M, 140M, 350M, 370M, 770M, 1,3B; 100B token di FineWeb-Edu.

**Legge di scala** [C]:
`L_hat(mu_rec, D) = E + X · N(mu_rec)^{-x} + Y · D^{-y}`, con esponenti `gamma_mu ≈ 0,40` e `gamma_D ≈ 0,78`.
Versione unificata training + test-time:
`L_hat_unified(T | mu_rec, D) = [ E + X·N(mu_rec)^{-x} + Y·D^{-y} ] + Z·exp( -z · T · mu_rec^{-1} )`
Il secondo termine dice che **il guadagno a test-time decade esponenzialmente in `T/mu_rec`**: iterare molto oltre la profondita' media vista in training rende sempre meno.

**Risultati** [C]: contro i recurrent-depth precedenti a 100M, **-6,3% di perplessita'** e +1,8 punti medi downstream. A 1,3B contro Transformer: perplessita' di validazione 11,42 vs 11,95 (-4,4%); Core 28,44 vs 25,45 (+2,99); Core-Extended 17,08 vs 15,90 (+1,18). Parcae 770M ≈ Transformer 1,3B.

### 3.2 STARS — Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning (arXiv 2605.26733)

[P, abstract] Problema: nei LoopLM **la prestazione raggiunge un picco a una certa profondita' e poi collassa** aumentando la ricorrenza. Soluzione: concepire il ragionamento come riduzione di incertezza e vincolare gli stati latenti ad avvicinarsi a **punti fissi asintoticamente stabili**, tramite **regolarizzazione del raggio spettrale dello Jacobiano** con campionamento casuale dei loop. Risultato: scaling test-time affidabile su aritmetica, degradazione fortemente mitigata all'aumentare della profondita', picco migliorato su matematica complessa.

### 3.3 Fixed-Point Reasoners (FPRM) (arXiv 2606.18206)

[P, abstract] Usa la **convergenza a punto fisso come meccanismo di halting end-to-end** dentro un'architettura looped. Stabilizza la propagazione del segnale con **layer pre-norm e scaling del residuo**. Adatta il calcolo alla difficolta' del task. Valutato su Sudoku, Maze, state-tracking e ARC-AGI.

### 3.4 A Mechanistic Analysis of Looped Reasoning Language Models (arXiv 2604.11791)

Il correttivo empirico alla narrativa del "ragionamento latente". Modelli analizzati [C]: **Ouro 1.4B**, **Huginn-0125**, Llama retrofittato, OLMo retrofittato.

Risultati [C]:
- I blocchi ricorrenti **imparano stadi di inferenza che rispecchiano da vicino quelli dei modelli feedforward, e li ripetono in profondita' ad ogni iterazione**. Le iterazioni **non si specializzano**.
- *"Ogni layer nel ciclo converge a un punto fisso distinto; di conseguenza il blocco ricorrente segue una traiettoria ciclica consistente nello spazio latente."* Il comportamento dominante e' un **punto fisso ciclico**, non un singolo punto fisso.
- **I pattern di attention sembrano convergere dopo la prima iterazione** in alcuni modelli; il comportamento delle teste di attention si stabilizza e diventa costante fra ricorrenze, con convergenza immediatamente dopo il prelude.
- Il residual stream tende ad essere simile fra ricorrenze.
- L'analisi usa 128 iterazioni per stabilire punti fissi approssimati.

Implicazione diretta e scomoda: se l'attention converge subito, **rileggerla ad ogni sotto-passo puo' essere quasi ridondante dal secondo giro in poi** — a meno che qualcosa (l'injection, o un reset alla HRM) non la costringa a cambiare.

### 3.5 Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence? (arXiv 2609.01924)

Il risultato piu' utile in assoluto per decidere quanto pesare l'input injection.

Modelli [C]: Huginn-0125 (focus), Ouro-2.6B, e un feedforward di confronto.

Risultati [C]:
- **La norma di trasporto dal prelude e' indipendente dalla distanza verso ogni target (0,26-0,37), mentre le sorgenti a meta' ricorrenza collassano sotto 0,06 entro 4-5 ricorrenze.** Cioe': **e' l'encoding di input ri-iniettato, non lo stato ricorrente, a sostenere il contenuto del workspace.**
- Test di attenuazione: scalando gli embedding iniettati per `alpha <= 0,25` **la qualita' dell'output viene distrutta** da qualunque punto di inizio; `alpha = 0,75` e' innocuo.
- Test di scambio: sostituendo gli embedding con quelli di un prompt diverso, **l'output finale si ribalta sulla risposta del nuovo input entro circa 4 ricorrenze**.
- Analisi a r=16 ricorrenze, profondita' virtuale 68.
- Conclusione di progetto [C]: lo stato ricorrente va letto come **memoria di lavoro continuamente ri-ancorata a un encoding di input fisso**, non come residual stream. L'accesso persistente al workspace richiede cicli di refresh allineati ai confini di ricorrenza, non accumulo di stato.

### 3.6 Two-Scale Latent Dynamics for Recurrent-Depth Transformers (arXiv 2509.23314)

[C] Modello scala GPT-2 (12 layer, 12 teste, hidden 768, block 512), FineWeb fino a 300M token. Tre regioni ciclano indipendentemente: layer 4 in self-loop, layer 5-6 in loop accoppiato, layer 7 in self-loop; tutti gli altri single-pass.

Risultato: decomposizione a due scale — **dentro i loop gli aggiornamenti sono piccoli e sempre piu' ortogonali** (raffinamenti locali, "archi stretti"), **fra i blocchi si ha una deriva su scala maggiore** ("salti piu' grandi"). Il coseno fra aggiornamenti consecutivi si stabilizza fra **0,5 e 0,65**: raffinamenti non collineari, non spinte ripetute nella stessa direzione. La geometria e' a spirale.

Applicazione pratica [C]: criterio di uscita **basato sull'accelerazione** (differenza seconda fra aggiornamenti), che batte i baseline a divergenza KL e a norma del passo, riducendo la latenza da circa 580 a circa 360 ms/token a soglie alte.

### 3.7 Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers (arXiv 2604.07822)

[C] Transformer a 4 layer (testati anche 6 e 8) con ricorrenza R ∈ {1,...,8} su knowledge graph (2000 entita', 200 relazioni per la generalizzazione sistematica; 200 entita', 10 relazioni per l'estrapolazione in profondita'). **Esplicitamente senza input injection, senza gated halting, senza middle looping.**

Risultati [C]:
- Anche la ricorrenza piu' semplice R=2 ottiene generalizzazione non banale su composizioni mai viste, dove il transformer vanilla (R=1) **fallisce completamente**.
- R=4 converge a circa 2000 epoche contro 7000 di R=2. L'accuratezza OOD richiede training prolungato: ordine 10⁴ epoche contro 10² per quella in-distribution (dinamica grokking a tre stadi: memorizzazione -> generalizzazione in-distribution -> composizione sistematica).
- Estrapolazione: modelli addestrati su dati a 12 hop — R=6 estrapola a 14 hop, R=8 arriva a 19 hop. Un modello addestrato fino a 5 hop raggiunge 10 hop a test aumentando le iterazioni.
- **Overthinking**: la prestazione degrada oltre il numero ottimale di iterazioni; il margine sui logit cresce fino a un picco e poi cala. **I modelli con ricorrenza dinamica (Poisson) sono piu' resistenti all'overthinking** di quelli a iterazioni fisse.

### 3.8 LOTUS — Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers (arXiv 2606.31779)

[C] Problema quantificato: i metodi di CoT latente **stanno alla pari con l'esplicito a scala GPT-2 (124M) ma perdono 9,2 punti a 3B**, e il gap si allarga con la scala.

Architettura [C]: la domanda e' seguita da **K=6 blocchi di c=25 token latenti apprendibili** (150 posizioni latenti totali), poi i suffissi di risposta. Il modello base processa la sequenza **R=6 volte**, con `h^(t) = f_theta( E + h^(t-1) | C_pre )`, dove `C_pre` cachea il prefisso della domanda ed `E` sono gli embedding latenti apprendibili. **Input injection per somma (E + h^(t-1)), R fisso a 6 sia in training sia in inferenza, nessun halting.**

Training [C]: due loss sui latenti post-loop `h^(R)` — `L_step` (cross-entropy su ogni posizione latente contro il token del passo CoT corrispondente, supervisione **parallela**) e `L_ans` (next-token standard). `L = L_ans + lambda_step · L_step`.

Risultati [C] su Llama-3.2-3B-Instruct: CoT esplicito GSM8K 71,5% e OOD medio 62,1%; CODI+SIM-CoT 62,3% e 62,8%; **LOTUS 70,0% e 63,9%**; LOTUS+CODI 70,6% e 63,4%. Gap in-domain chiuso a 1,5 punti, superamento del CoT esplicito fuori dominio. Latenza della fase di pensiero **2,5x piu' veloce** (133 ms contro 339 ms).

### 3.9 Training-Free Looped Transformers (arXiv 2605.23872)

[C] Wrapper a tempo di inferenza che cicla **un blocco contiguo di layer a meta' stack** di un checkpoint congelato, senza fine-tuning ne' modifiche architetturali. Gli autori avvertono che **la ri-applicazione ingenua del blocco di solito degrada la prestazione**. Il meccanismo di stabilizzazione: vedere i blocchi pre-norm come passi di Eulero in avanti di un'ODE e trattare il loop come raffinamento in **sotto-passi piu' piccoli e smorzati** (damped), non come un singolo grande aggiornamento. Risultati: Qwen3-4B-Instruct +2,64 punti su MMLU-Pro; Qwen3-30B-A3B +1,14 su CommonsenseQA; Moonlight-16B-A3B +1,20 su OpenBookQA.

### 3.10 Looped Latent Attention: Cross-Loop KV Compression (arXiv 2607.15456)

[P/C, abstract dettagliato] Chiarisce il costo del regime "una cache per giro": *"i looped weight-tied riducono i parametri riusando un singolo blocco, ma il decoding memorizza comunque una cache K/V separata per ogni step di ricorrenza"*, cioe' R volte la memoria.

Osservazione chiave [C]: **per un dato token, layer e testa, i vettori K/V tracciano una traiettoria a basso rango attraverso i loop.** Si memorizzano quindi latenti K e V compatti e si ricostruiscono i K/V specifici del loop solo quando l'attention li legge. Due varianti: codec per-testa, e LLA-2D che ripiega le teste in un unico latente.

Numeri [C]: a contesto 4k su H200 la capacita' di batch passa da 32 a **768 sequenze a compressione 21,3x**; un codec SVD e' quasi senza perdita fino a **32x**; a compressione 4x su MATH-500 l'accuratezza **sale** da 0,43 a 0,66. Testato su Ouro-1.4B, Ouro-2.6B-Thinking, Huginn-3.5B.

### 3.11 T-LoopFormer (arXiv 2609.15160)

[P, abstract] Profondita' elastica a livello di token con router dinamico che decide se un token continua a ricorrere o esce presto. **Recursion-wise KV cache**: ad ogni step di ricorsione i token mantengono cache indipendenti, e *"i token a profondita' diverse attendono solo ai corrispondenti stati cacheati"*, eliminando la computazione ridondante per i token gia' usciti. Dichiarano di superare il modello base anche a 24x di riduzione dei FLOP.

### 3.12 RecurTrace (arXiv 2609.03379)

[P, abstract parziale] **Loop Memory Attention**: *"permette a ciascun layer ciclato di attendere ai propri stati delle iterazioni precedenti lungo l'asse del loop-time, cosi' che il modello possa rivisitare computazioni precedenti invece di affidarsi solo allo stato piu' recente"*. Halting head supervisionata da un oracolo che identifica quando profondita' aggiuntiva riduce ancora la loss. Risultati su MathQA: RecurTrace **56,9% con 2,0 loop medi**; LoopUS-Conf 55,3% a 3,2 loop; TaH-Mismatch 55,7% a 2,1 loop; CALM 54,1% a 5,6 loop. Guadagni da 0,6 a 3,4 punti su modelli 0,6B / 1,7B / 4B / 8B.

### 3.13 Think Shallow, Solve Deep (arXiv 2608.18222)

[P/C] Abstract: *"i reasoner a profondita' ricorrente puntano a risolvere problemi piu' difficili iterando piu' a lungo a test-time, ma iterazioni aggiuntive possono migliorare, preservare o degradare una risposta."* Tassonomia degli operatori ricorrenti in tre regimi: **settling** (stabile, l'output non cambia), **marginal** (preserva senza migliorare), **drifting** (degrada). Diagnosi: si misura lo spostamento per-passo rispetto al margine del decoder; quando lo spostamento e' piccolo rispetto al margine, la risposta decodificata e' **depth-safe**. Un singolo obiettivo terminale di punto fisso spinge l'operatore nel regime settling; rimuoverlo induce drift.

### 3.14 Stabilizing Extrapolation via Learned Stochastic Stopping (arXiv 2606.29983)

[P, abstract] Trova una **correlazione spuria fra lunghezza della sequenza e numero di loop** nei task algoritmici semplici, che rende l'estrapolazione fragile. *"Introdurre stocasticita' nel numero di loop durante il training riduce nettamente la varianza OOD e stabilizza le predizioni fra i conteggi di loop a inferenza."* Tesi esplicita degli autori: **"quando fermarsi" e' una scelta di progetto a tempo di training, non solo una regola di allocazione a tempo di inferenza.** Testato su addizione binaria, Dyck-1, Unique Set, Copy. RL-Halting migliora il compromesso accuratezza/stabilita' ma puo' anche stabilizzare una computazione subottimale.

### 3.15 Fractal basins trap latent reasoning (arXiv 2609.04963)

[C, abstract verbatim] *"Mostriamo che i modelli di reasoning esibiscono caos transiente, una conseguenza fisica della complessita' computazionale dei task difficili. Di conseguenza, mostriamo che diversi modelli di reasoning di frontiera sono sistemi dinamici con bacini frattali, con la frattalita' che aumenta con la difficolta' del task... Mostriamo che il caos transiente emerge perche' il ragionamento resta intrappolato per lunghi periodi vicino a punti di sella, che corrispondono a tentativi di soluzione quasi corretti del problema sottostante."* Implicazione: i rallentamenti nel ragionamento sono una conseguenza inevitabile della durezza del problema.

### 3.16 Recurrent Looped Transformer (RLT), 2026

[P, pagina di progetto e fonte secondaria] Estende il loop **attraverso i token** invece che solo dentro il token. Un **encoder causale costruisce una memoria key-value globale**; un **decoder ricorrente porta il proprio stato nascosto finale e la cache di sliding-window attention per layer attraverso ogni token** di prompt e risposta. "Profondita' infinita" significa un cammino computazionale temporale estensibile al crescere della sequenza, non calcolo infinito dentro un token: con un decoder a 48 layer, dopo `t` token il cammino attraversa `48·t` blocchi, ma ogni token ne esegue un numero fisso. Co-design con hardware (separazione del lavoro parallelo dell'encoder da quello ricorrente del decoder) e con RL.

### 3.17 Altri, catalogati ma non approfonditi

[P, dalla lista Awesome-Loop-Models e da ricerca] Memory-Efficient Looped Transformer (2605.07721); Latent Recurrent Transformer: Architecture Exploration (2605.26797); Looped Diffusion Language Models (2605.26106); Solve the Loop: Attractor Models (2605.12466); Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization (2603.21676); Adaptive Depth in Looped Transformers (2607.20519); Persistent Recurrent Memory Between Transformer Layers (2609.17251) — inserisce uno stato latente con aggiornamenti GRU fra le due meta' del transformer; LoopSpec (2609.17184) — decoding self-speculative che bozza dagli stati ricorrenti intermedi; Right Direction, Wrong Step (2609.16665); Looped World Models (2606.18208); Tiny Autoregressive Recursive Models (2603.08082); Test-time Adaptation of Tiny Recursive Models (2511.02886).

---

## 4. Risposte trasversali alle otto domande

### 4.1 Cosa sta dentro il loop

| Lavoro | Contenuto del blocco ripetuto |
|---|---|
| Universal Transformer | 1 layer: MHSA + transizione (conv separabile o FC) [C] |
| ALBERT | tutti i 12/24 layer, condivisi (attention + FFN) [C] |
| CoTFormer | blocco da 12 o 24 layer [C] |
| Yang 2023 | **1 layer** [C] |
| Fan 2024 | 1 blocco decoder (piu' layer) [C] |
| Huginn | **4 layer** (attention + MLP ciascuno), con prelude 2 e coda 2 separati [C] |
| Saunshi | k layer, k ∈ {1..12} [C] |
| HRM | 2 blocchi transformer encoder-only (f_L e f_H) [C] |
| MoR | blocco condiviso, Middle-Cycle: primo e ultimo layer a capacita' piena [C] |
| TRM | **rete a 2 layer**, con o senza attention [C] |
| Ouro | 24 o 48 layer condivisi, nessun prelude/coda [C] |
| Parcae | blocco ricorrente + injection lineare esplicita [C] |

**Nessun lavoro isola l'attention fuori dal loop.** Il "blocco" e' sempre un'unita' completa che contiene mixing fra token (attention) e trasformazione per-token (MLP).

### 4.2 Input injection

| Lavoro | Injection? | Come |
|---|---|---|
| Universal Transformer | no per il contenuto | somma di coordinate embedding (posizione + timestep) [C] |
| ALBERT | no | — [C] |
| Yang 2023 | **si'** | **somma**: `Y_{t+1} = M(Y_t + P)` [C] |
| Fan 2024 | **si'** | somma degli embedding di input all'output del passo precedente [C] |
| Relaxed Recursive | no | — [C] |
| Huginn | **si'** | **concatenazione + adattatore** `A: R^{2h} -> R^h`; la concatenazione batte la somma alla scala [C] |
| Saunshi | **no** | pura composizione `(Tb)^T` [C] |
| HRM | **si'**, ad ogni timestep | somma elemento a elemento di `x_tilde` nel modulo L [C] |
| MoR | no (residuo) | `H^{r+1} = g·f(H^r) + H^r` [C] |
| TRM | **si'**, ad ogni passo latente | `z <- net(x, y, z)` [C] |
| Ouro | **no** | `H^(t) = f(H^(t-1))` [C] |
| LOTUS | **si'** | somma: `h^(t) = f(E + h^(t-1) | C_pre)` [C] |
| Parcae | **si'**, esplicita e vincolata | `h_{t+1} = A_bar·h_t + B_bar·e + R_bar(h_t, e)` [C] |
| Loop-Think-Generalize | **no**, per scelta | dichiarato esplicitamente [C] |

**Verdetto**: due schieramenti netti. Chi punta a scalare il test-time compute e a lavorare su linguaggio naturale inietta (Huginn, Parcae, LOTUS, HRM, TRM, Yang, Fan); chi studia l'espressivita' su task sintetici spesso non inietta (Saunshi, Loop-Think-Generalize, ALBERT). L'analisi Jacobiana 2026 e Parcae danno la ragione meccanicistica: **senza injection l'informazione a meta' ricorrenza decade sotto 0,06 di norma di trasporto entro 4-5 giri** [C], e **con injection non vincolata il sistema esplode** [C].

### 4.3 KV cache per iterazione — i tre regimi

**Regime A, "parallel-in-time" (default).** Al giro *r* l'attention legge gli stati dei token precedenti **allo stesso giro r**. Cache separata per ogni giro, memoria R volte. Adottato da: Universal Transformer [C], Huginn (default) [C], Ouro [C], Saunshi [C], MoR con recursion-wise caching [C], T-LoopFormer [P]. Ouro quantifica il costo del *non* farlo: riusare la cache in prefill costa **oltre 10 punti su GSM8K** [C], perche' "ogni step trasforma le rappresentazioni in modi che non possono essere approssimati dagli step precedenti".

**Regime B, "cross-loop" (attention sull'asse profondita').** Al giro *r* l'attention vede **tutti i giri fino a r** dei token precedenti. Adottato da: CoTFormer (`[x^{(i)}, ..., x^{(0)}]`, la sequenza cresce di R volte) [C]; Think-at-Hard duo-causal (`j <= i AND k <= d`, cache concatenata su tutte le profondita') [C]; RecurTrace loop-memory-attention (ogni layer attende ai propri stati delle iterazioni precedenti) [P]. E' strettamente piu' espressivo. Il costo e' che la lunghezza effettiva della sequenza si moltiplica per R. CoTFormer motiva questo regime dicendo che e' **l'unico che riproduce la vera semantica del chain-of-thought** [C].

**Regime C, "condiviso/compresso".** Huginn KV-sharing zero-shot con budget `k` e indice `i mod k`, giustificato dal fatto che tutte le entry vengono dalle stesse proiezioni [C]; MoR recursive KV sharing (cache solo allo step 1, riusata su tutti gli N_r step; memoria 1/N_r) [C]; Looped Latent Attention (i K/V tracciano una traiettoria **a basso rango** attraverso i loop, quindi si comprimono: 21,3x con batch da 32 a 768, quasi lossless fino a 32x, e a 4x l'accuratezza su MATH-500 *sale* da 0,43 a 0,66) [C].

### 4.4 Numero di loop, adattivita', halting

**In training**: fisso (Saunshi, Fan con T(n) derivato dal task, ALBERT, LOTUS R=6) oppure **campionato** (Huginn log-normale-Poisson media 32; Parcae media in {2..12}; Ouro prior uniforme su 1..4; Loop-Think-Generalize Poisson). Il paper sullo stochastic stopping sostiene esplicitamente che **la stocasticita' nel conteggio dei loop durante il training e' cio' che riduce la varianza OOD** [P].

**A test**: Huginn valuta 1, 4, 8, 16, 32, 64, 128 [C]; Parcae fino a T=24 contro mu_rec fino a 12 [C]; Ouro fino a 4 con exit appreso [C].

**Halting**, quattro famiglie:
1. **ACT classico** con probabilita' di halting per posizione, lo stato dei token fermati viene copiato in avanti (Universal Transformer) [C].
2. **Criterio di convergenza a costo zero**: KL fra output di passi consecutivi sotto **5x10⁻⁴** (Huginn) [C]; differenza seconda / accelerazione, migliore della KL e con latenza da 580 a 360 ms/token (Two-Scale) [C]; convergenza a punto fisso come halting end-to-end (FPRM) [P].
3. **Gate appreso**: Ouro con `lambda_t = sigma(Gate(F^(t)))` e criterio CDF Q-exit [C]; Q-learning halt/continue (HRM, M_max fino a 16) [C]; router expert-choice/token-choice (MoR) [C]; decider MLP che salta il 93% dei token (Think-at-Hard) [C]; halting head distillata da oracolo (RecurTrace, 56,9% con 2,0 loop medi contro CALM 54,1% con 5,6) [P].
4. **Oracolo dal task**: T(n) derivato dalla struttura n-RASP-L (Fan) [C].

**Overthinking**: fenomeno documentato indipendentemente da tre lavori. Think-at-Hard chiama "latent overthinking" il fatto che predizioni corrette al primo passo vengono rovinate dalle iterazioni successive [C]; Loop-Think-Generalize misura il margine sui logit che cresce, picca e poi cala [C]; STARS lo formula come picco seguito da collasso [P]; Think Shallow lo classifica come regime "drifting" [P].

### 4.5 Come si allena

- **Backprop completo su tutti i loop**: Saunshi [C], Fan [C], MoR [C], TRM [C], Universal Transformer [C].
- **Troncato agli ultimi k**: Huginn, **k=8** — prelude escluso, che riceve gradiente ovunque perche' iniettato ad ogni passo; memoria del backward indipendente da r [C]. Yang 2023 con finestra T mobile (T=5..15 a seconda del task) [C]. Block-Recurrent: BPTT troncato su segmenti da 4096 token in 8 blocchi da 512, con cache K/V ma non gradienti fra segmenti [C].
- **Gradiente a un passo / deep equilibrium**: HRM, `(I - J_F)⁻¹ ≈ I`, memoria O(1), con deep supervision a segmenti e detach dello stato fra segmenti [C]. **TRM lo confuta con un'ablazione: 56,5% contro 87,4% su Sudoku, 31 punti di differenza** [C].
- **Loss sugli step intermedi**: Yang media la loss su una finestra di iterazioni [C]; Ouro usa l'attesa `sum_t q_phi(t|x)·L^(t)` su tutti gli step, con regolarizzazione entropica [C]. Saunshi invece supervisiona solo l'ultimo loop [C]; Fan supervisiona solo la risposta finale, senza alcun target intermedio [C].

### 4.6 Stabilizzazione

| Tecnica | Chi | Dettaglio |
|---|---|---|
| Sandwich norm (norm prima e dopo ogni sotto-blocco) | Huginn [C], Ouro [C] | Huginn: RMSNorm; la versione parameter-free fallisce alla scala |
| Prelude/coda con pesi propri | Huginn (2,4,2) [C], MoR Middle-Cycle [C], Saunshi middle-looping [C] | tenere fuori dal loop il primo e l'ultimo layer migliora sistematicamente |
| Controllo spettrale sull'injection | **Parcae** [C] | `A := Diag(-exp(log_A))`, ZOH `A_bar = exp(Delta ⊙ A)`, garantisce `rho(A_bar) < 1` |
| Regolarizzazione del raggio spettrale dello Jacobiano | STARS [P] | con campionamento casuale dei loop |
| Pre-norm + residual scaling | FPRM [P] | per la propagazione del segnale a grande profondita' |
| Sotto-passi smorzati (vista Eulero/ODE) | Training-Free Looped [C] | la ri-applicazione ingenua degrada; il damping la salva |
| Init scalata sulla profondita' *effettiva* | Huginn [C] | `sigma_out² = 1/(5·h·l)` con `l = l_P + r_medio·l_R + l_C = 132`, non il numero di layer fisici |
| Learning rate conservativo | Huginn [C], Ouro [C] | Huginn 5x10⁻⁴; sopra, collasso dello stato |
| Ridurre il numero di step ricorrenti | Ouro [C] | da 8 a 4 dopo 3T token instabili; batch da 4M a 8M |
| Obiettivo terminale di punto fisso | Think Shallow [P] | sposta l'operatore nel regime "settling" |
| Gate LSTM o fisso sul residuo ricorrente | Block-Recurrent [C] | il **fixed gate** batte l'LSTM gate empiricamente |

### 4.7 Risultati chiave, a parita' di parametri e di FLOP

Il confronto che conta per un progettista e' iso-FLOP, non iso-parametri.

- **Saunshi** [C]: a **iso-FLOP**, looped `(12⊗2)` ha perplessita' **peggiore** (7,90 vs 7,40) ma ragionamento **migliore** (math word problems 34,3% vs 29,3%; reasoning primitives 51,2% vs 47,5%). Copre il 34-50% del gap di perplessita' ma il 78-282% del gap di ragionamento.
- **MoR** [C]: a **FLOP fissi** (16,5e18), MoR con 118M batte il vanilla da 315M sia in NLL (2,7925 vs 2,7824 — quasi pari) sia in few-shot (43,1% vs 42,3%) con **meta' dei parametri**.
- **Parcae** [C]: 770M ≈ Transformer 1,3B; a 1,3B contro Transformer +2,99 su Core e -4,4% di perplessita'.
- **Ouro** [C]: 1,4B batte Qwen3-4B su GSM8K (78,92 vs 72,86) e MATH500 (82,40 vs 59,60) ma perde su MMLU (67,35 vs 73,19) — **coerente con la tesi "manipolazione si', capacita' no"**.
- **Yang** [C]: 0,79M contro 9,48M, prestazione pari o superiore su in-context learning.
- **TRM** [C]: 7M, ARC-AGI-1 45% contro HRM 27M al 40,3%.
- **Huginn** [C]: 3,5B con 800B token, a r=32 GSM8k CoT 42,08% strict; a 180B token 9,02% contro 1,82% del baseline.

### 4.8 Perche' il loop aiuta — le spiegazioni degli autori, e la revisione del 2026

**Le tesi originali:**
1. **Profondita' effettiva senza parametri**. I problemi di ragionamento richiedono profondita', non parametri (Saunshi) [C]. Un k-layer ciclato L volte eguaglia quasi un kL-layer [C].
2. **Pensieri latenti equivalenti al CoT**. Teorema 5.4: m passi di CoT si simulano con m loop [C]. Non serve costruire dati appositi (Huginn) [C].
3. **Bias induttivo verso algoritmi iterativi e punti fissi**. La discesa del gradiente e' moltiplicazione ripetuta, l'inversione no (Yang) [C]. Il loop emula l'iterazione di punto fisso [C].
4. **Manipolazione della conoscenza, non capacita'**. Ouro misura circa **2 bit per parametro identici** fra looped e non-looped sul task sintetico di biografie: il loop non aggiunge conoscenza, migliora la composizione [C].
5. **Regolarizzazione / spazio di ipotesi ridotto**. Looped SSMs conclude che il vantaggio **non** viene da maggiore espressivita' ma dal fatto che la condivisione dei parametri restringe lo spazio di ipotesi e semplifica l'ottimizzazione [C]. TRM ammette di non avere una teoria e sospetta l'overfitting [C].
6. **Piu' FLOP per parametro**, quindi meno comunicazione nel training distribuito (Huginn) [C].

**La revisione empirica del 2026** — necessaria da conoscere prima di progettare:
- Le iterazioni **non si specializzano**: ripetono gli stessi stadi di inferenza dei modelli feedforward [C].
- I **pattern di attention convergono dopo la prima iterazione** in alcuni modelli, e il comportamento delle teste diventa costante fra ricorrenze [C].
- Il comportamento tipico non e' un punto fisso singolo ma un **ciclo**: ogni layer converge a un punto fisso distinto, il blocco segue una traiettoria ciclica [C]. Coerente con ALBERT (non converge a zero nemmeno dopo 24 layer) [C] e con Huginn (orbite osservate sui token aritmetici) [C].
- E' **l'input injection, non lo stato ricorrente**, a sostenere il contenuto: trasporto dal prelude 0,26-0,37 indipendente dalla distanza, sorgenti a meta' ricorrenza sotto 0,06 entro 4-5 ricorrenze [C].
- Il guadagno test-time **decade esponenzialmente in T/mu_rec** [C].
- Il sistema puo' essere caotico: bacini frattali, punti di sella che corrispondono a soluzioni quasi corrette [C].

---

## 5. Cosa implica per un core ricorrente non-transformer

Questa e' la sezione che riguarda direttamente FlyToLLM: un grafo ricorrente spiking come "blocco", con attention esterna.

### 5.1 Esiste letteratura su "attention ad ogni sotto-passo di una rete ricorrente"?

**Si', in tre forme distinte, tutte con precedenti solidi.**

**Forma 1 — Spiking transformer: la stessa attention ri-applicata ad ogni timestep del core spiking.** Questo e' **esattamente** il pattern richiesto, ed e' lo standard di fatto nei spiking transformer.

Spikformer [C]:
- L'input ha forma `T x N x D` con `T` = timestep di simulazione. **Lo stesso blocco di Spiking Self-Attention, con le stesse matrici W_Q, W_K, W_V, viene applicato ad ogni timestep.**
- *"La SSA e' condotta indipendentemente su ogni time step"*: l'attention **non** attraversa i timestep. L'informazione fra timestep passa **solo attraverso il potenziale di membrana dei neuroni LIF**.
- `SSA'(Q,K,V) = SN( Q·Kᵀ·V·s )`, poi `SSA = SN(BN(Linear(SSA')))`. Nessun softmax; `s` e' un fattore di scala per controllare la magnitudine del prodotto matriciale.
- Numeri: ImageNet **74,81%** top-1 con **T=4** (66,3M parametri, contro SEW-ResNet-152 a 60,2M e 69,26%); CIFAR10 95,19% con Spikformer-4-384 a T=4; CIFAR10-DVS 78,9% con T=10.

Il messaggio operativo: **T=4 riletture dell'attention con gli stessi pesi, una per sotto-passo del core spiking, e' una configurazione validata e funzionante.** Corrisponde numericamente ai 4 sotto-passi gia' usati in FlyToLLM.

**Forma 2 — Attention come operatore ricorrente.** Block-Recurrent Transformer [C]:
- La cella usa **self-attention sui vettori di stato + cross-attention ai token** (direzione orizzontale, aggiornamento dello stato) e **self-attention sui token + cross-attention allo stato** (direzione verticale, elaborazione dei token). Le due attention sono eseguite **in parallelo**, non in serie.
- **512 vettori di stato ricorrente**, finestra/blocco **W = 512 token**.
- Il residuo e' sostituito da un **gate**. Il *fixed gate* — `g = sigma(b_g)` costante, indipendente dall'input, `c_{t+1} = c_t ⊙ g + z_t ⊙ (1-g)`, cioe' una media mobile esponenziale sui blocchi — **batte empiricamente il gate LSTM**. Init critica: si aggiunge -1 all'input gate e +1 al forget gate per polarizzarli verso "ricorda", e i pesi partono piccoli (sigma=0,1) per evitare il collasso sul baseline non ricorrente.
- La ricorrenza e' **sul tempo** (blocchi successivi di token), non sulla profondita'. Un solo layer ricorrente, al **layer 10 su 12**; nei modelli da 1,3B, due layer ricorrenti ai layer 10 e 20 su 24.
- PG19 a livello di token: XL:2048 0,990 bit/token con 2,11x di rallentamento; Rec:fixed:skip **0,952** a costo pari. Perplessita' migliore di un Transformer-XL con finestra 4 volte piu' grande, girando **due volte piu' veloce**.
- Perche' funziona [C]: lo stato ricorrente e' **di ordini di grandezza piu' grande** di quello di un LSTM (512 vettori contro un singolo hidden), il che aumenta drasticamente la capacita' di catturare il passato; e processare a blocchi riduce di ordini di grandezza il numero di passi ricorrenti, quindi di applicazioni del forget gate, migliorando la propagazione del gradiente. Analisi qualitativa: il modello usa la ricorrenza soprattutto per **lookup di nomi a lungo raggio** oltre la finestra di attention, cioe' come memoria esterna.

**Forma 3 — Ibridi attention + core ricorrente in profondita'.** Jamba [C]:
- Rapporto **1:7** attention:Mamba. Blocco di **8 layer, di cui 1 di attention (GQA) e 7 Mamba**. 4 blocchi Jamba, 32 layer totali. 12B parametri attivi su 52B disponibili.
- Memoria KV a contesto 256K in 16 bit: Llama-2 70B **128 GB**, Mixtral 8x7B **32 GB**, Jamba **4 GB**.
- **Ablazione decisiva** (1,3B, 250B token, log-prob su C4): pure Attention **-0,543**; pure Mamba **-0,543**; Jamba 1:3 **-0,533**; Jamba 1:7 **-0,533**. Cioe': l'ibrido batte entrambi i puri, e **non c'e' differenza sostanziale fra 1:3 e 1:7**.
- Il motivo [C]: Mamba puro fallisce sull'**in-context learning** — su IMDB, QuAC e NarrativeQA "spesso non segue il formato corretto". Le teste di attention abilitano le **induction head**, che fanno copia approssimata e sostengono l'ICL. *"Anche quando solo 1 layer su 8 e' di attention"*, emergono capacita' ICL comparabili a un transformer puro.

Tiny Recursive Reasoning con ibrido Mamba-2 [C/P]: la variante TR-mamba2attn usa una pipeline **Mamba-2 -> Mamba-2 -> Attention -> MLP** per passo ricorsivo, rapporto SSM:attention **2:1**, e *"questo stesso stack di blocchi viene riusato identico attraverso tutte le iterazioni ricorsive: l'attention e' presente in ogni passo ricorsivo, non interlacciata selettivamente"*. Parametri quasi identici: 6,86M ibrido contro 6,83M del TRM base. ARC-AGI-1 pass@2 **45,88% contro 43,88%** (+2,0), pass@100 **65,25% contro 60,50%** (+4,75), pass@1 in parita' (40,50% vs 40,75%). H_cycles=3. **Nessuna ablazione sul numero di layer di attention** — la domanda "quanta attention serve dentro il loop" non e' ancora stata studiata sistematicamente.

**Spiking puro senza attention**: URCHIN (BabyLM Challenge 2026) [P] — neuroni leaky integrate-and-fire accoppiati da un **connettoma laterale con legge di Dale**, che risolvono ogni token tramite un **loop multi-trasmissione che ricircola l'attivita' fino a un punto fisso**. Singolo layer orizzontale con **128 neuroni**, **4,23M parametri**, **nessuna attention**. Doppia implementazione con un solo set di pesi: uno **scan SSM parallelo** efficiente su GPU per il training, e una **RSNN event-driven** a costo costante per l'inferenza su CPU o hardware neuromorfico. E' il precedente piu' vicino al caso FlyToLLM sul lato core spiking, e conferma che il ricircolo a punto fisso su un connettoma e' addestrabile.

**Looped SSM in profondita'** [C]: un blocco SSM riusato L volte contro L blocchi indipendenti, testato su S5, LRU, LinOSS, LrcSSM e sei benchmark. Un looped SSM con `k` parametri eguaglia o supera modelli indipendenti con `k·L` parametri. Conclusione degli autori: la depth-recurrence e' **un bias induttivo benefico che semplifica l'ottimizzazione**, non un aumento di espressivita'.

### 5.2 Cosa dice la letteratura sul rapporto fra quante volte si legge l'attention e quanto lavoro fa il core ricorrente

Non esiste uno studio dedicato con questo titolo, ma tre linee di evidenza convergono e vanno lette insieme.

**Evidenza 1 — poca attention basta, se il core ricorrente e' buono.** Jamba: 1 layer di attention su 8, e 1:3 non e' meglio di 1:7 [C]. Il ruolo dell'attention e' specificamente l'**in-context learning / induction head**, cioe' il richiamo associativo di contenuto dal contesto, non la trasformazione generale.

**Evidenza 2 — l'attention nel loop converge presto.** L'analisi meccanicistica 2026 trova che i pattern di attention si stabilizzano gia' dopo la prima iterazione e restano costanti fra ricorrenze [C]. Se cosi' e' anche nel caso spiking, rileggere l'attention a ogni sotto-passo darebbe ritorni fortemente decrescenti **a meno che** qualcosa non cambi lo stato in modo abbastanza sostanziale fra una lettura e l'altra.

**Evidenza 3 — quel "qualcosa" e' il reset gerarchico.** HRM identifica il problema esatto e la sua soluzione: *"man mano che lo stato nascosto si assesta verso un punto fisso, l'ampiezza degli aggiornamenti si riduce, bloccando la computazione successiva e limitando la profondita' effettiva"*; il modulo alto, aggiornandosi ogni T passi, fornisce **contesto fresco che resetta il modulo basso verso un equilibrio diverso**, portando la profondita' effettiva da T a N×T [C]. Nell'architettura FlyToLLM attuale, **l'attention letta una volta ogni 4 sotto-passi occupa gia' il ruolo strutturale del modulo H di HRM**: e' il segnale lento che ri-eccita il core prima che si assesti.

Conseguenza logica per il committente: la scelta "attention dentro il giro" non e' binaria e non e' primariamente una questione di *frequenza*. Le domande decisive, nell'ordine in cui la letteratura le mette:
1. **A ogni sotto-passo il core spiking riceve un input nuovo?** Se si' (via injection e/o via attention), la profondita' effettiva cresce; se no, converge e i passi successivi sono sprecati (HRM [C], Jacobian Lens [C]).
2. **A che giro legge l'attention, il giro corrente o tutti i precedenti?** Regime A (stesso giro) e' il default e Ouro misura che rompere questa coerenza costa oltre 10 punti GSM8K [C]; regime B (cross-loop, CoTFormer/duo-causal) e' piu' espressivo ma moltiplica la lunghezza effettiva per R [C].
3. **L'injection e' controllata spettralmente?** Parcae dimostra che e' li' che il sistema esplode [C], e nel caso spiking il problema e' aggravato dal fatto che un grafo ricorrente ha gia' una sua dinamica propria.

### 5.3 Trasferimento operativo a FlyToLLM

Mappatura dell'architettura attuale (4 sotto-passi del grafo, lettura attention, 4 sotto-passi) sulla letteratura:
- E' **HRM con N=2, T=4** [C], dove il grafo spiking e' `f_L` e l'attention e' il segnale di `f_H`.
- E' anche **Spikformer con T=8 e attention a meta'** invece che ad ogni timestep [C].
- Non e' un looped transformer in senso stretto, perche' il blocco ripetuto (il grafo) non contiene mixing fra token.

Se l'attention va dentro il giro, ripetuta con gli stessi pesi, i precedenti diretti sono Spikformer (T=4, SSA identica ad ogni timestep, informazione cross-timestep solo via potenziale di membrana) [C] e TR-mamba2attn (stack Mamba-Mamba-Attention-MLP riusato identico ad ogni passo ricorsivo) [C]. Entrambi funzionano. Nessuno dei due ha ablato il numero di letture.

Le prescrizioni che la letteratura sostiene con numeri, ordinate per forza dell'evidenza:

1. **Iniettare l'embedding del token ad ogni sotto-passo.** E' la singola variabile piu' supportata: Huginn (concatenazione + adattatore, batte la somma alla scala) [C], HRM (somma ad ogni timestep) [C], TRM (x in ogni aggiornamento latente) [C], Yang (senza injection degrada oltre i loop di training) [C], Jacobian Lens (alpha<=0,25 distrugge l'output; le sorgenti a meta' ricorrenza muoiono in 4-5 giri) [C]. Per un core spiking, dove lo stato e' binario/sparso e la memoria e' nel potenziale di membrana, questo argomento e' se possibile piu' forte, non piu' debole.

2. **Vincolare spettralmente l'injection.** `A := Diag(-exp(log_A))` con discretizzazione ZOH garantisce `rho < 1` [C]. E' la differenza fra un loop che scala a test-time e uno che esplode.

3. **Randomizzare il numero di sotto-passi in training.** Log-normale-Poisson con media scelta (Huginn r_medio=32) [C], o media campionata in un insieme (Parcae {2..12}) [C]. Serve anche a evitare la correlazione spuria lunghezza/loop [P]. Ricordare pero' il tetto: il guadagno test-time decade come `exp(-z·T/mu_rec)` [C].

4. **Tenere prelude e coda fuori dal loop, con pesi propri.** Huginn (2,4,2) [C], MoR Middle-Cycle [C], Saunshi middle-looping [C], Relaxed Recursive (Stepwise preserva i layer esterni) [C]. Convergenza indipendente di quattro lavori.

5. **Backprop completo se la memoria lo consente, altrimenti troncato a k, non a 1.** TRM: 87,4% con backprop completo contro 56,5% con gradiente a un passo [C]. Huginn con k=8 su r medio 32 funziona [C]. Il gradiente a un passo di HRM e' l'opzione da evitare se si puo'.

6. **Decidere esplicitamente il regime KV.** Se si sceglie il regime A (ogni sotto-passo ha la sua cache), preventivare R volte la memoria, con l'avvertenza che i K/V attraverso i giri sono **a basso rango** e quindi comprimibili fino a 21-32x quasi senza perdita [C]. Se si sceglie il regime B (cross-loop), si ottiene la semantica CoT vera [C] al costo di una sequenza R volte piu' lunga.

7. **Sandwich norm e learning rate conservativo.** Huginn: RMSNorm sandwich obbligatoria, la variante parameter-free fallisce; lr 5x10⁻⁴, sopra collassa [C]. Ouro: stessa scelta, e ha dovuto scendere da 8 a 4 step dopo 3T token di instabilita' [C].

8. **Prevedere l'overthinking.** E' sistematico [C][P]. Le contromisure con numeri: iterazione selettiva sui token difficili (Think-at-Hard salta il 93% dei token e guadagna 5,0 punti medi) [C]; criterio di uscita per accelerazione anziche' per KL (latenza 580 -> 360 ms/token) [C]; ricorrenza dinamica in training, che rende piu' resistenti all'overthinking rispetto alle iterazioni fisse [C].

9. **Aspettarsi un ciclo, non un punto fisso.** ALBERT non converge nemmeno a 24 layer [C]; Huginn osserva orbite e slider oltre ai punti fissi [C]; l'analisi meccanicistica 2026 trova traiettorie cicliche come regime dominante [C]. Un core spiking, che ha gia' dinamiche oscillatorie proprie, cadra' verosimilmente in questo regime. I criteri di halting basati sull'assunzione di punto fisso vanno testati contro questa possibilita'.

10. **Non aspettarsi guadagni di conoscenza.** Ouro misura circa 2 bit per parametro identici fra looped e non-looped [C]; Saunshi misura che il loop copre solo il 34-50% del gap di perplessita' ma il 78-282% del gap di ragionamento [C]. Il ritorno atteso da piu' giri e' su composizione, multi-hop e manipolazione, non su memorizzazione o perplessita'.

---

## 6. Fonti

### Fonti primarie lette in testo pieno (HTML/ar5iv)

- Universal Transformers, Dehghani et al. 2018 — https://arxiv.org/abs/1807.03819 , testo pieno https://ar5iv.labs.arxiv.org/html/1807.03819 — **[C]**
- ALBERT, Lan et al. 2019 — https://arxiv.org/abs/1909.11942 , testo pieno https://ar5iv.labs.arxiv.org/html/1909.11942 — **[C]**
- Spikformer, Zhou et al. 2022 — https://arxiv.org/abs/2209.15425 , testo pieno https://ar5iv.labs.arxiv.org/html/2209.15425 — **[C]**
- Block-Recurrent Transformers, Hutchins et al. 2022 — https://arxiv.org/abs/2203.07852 , testo pieno https://ar5iv.labs.arxiv.org/html/2203.07852 — **[C]**
- CoTFormer, Mohtashami, Pagliardini, Jaggi 2023 — https://arxiv.org/abs/2310.10845 , testo pieno https://arxiv.org/html/2310.10845v2 — **[C]**
- Looped Transformers are Better at Learning Learning Algorithms, Yang et al. 2023 — https://arxiv.org/abs/2311.12424 , testo pieno https://arxiv.org/html/2311.12424v3 — **[C]**
- Jamba, Lieber et al. 2024 — https://arxiv.org/abs/2403.19887 , testo pieno https://arxiv.org/html/2403.19887v2 — **[C]**
- Looped Transformers for Length Generalization, Fan et al. 2024 — https://arxiv.org/abs/2409.15647 , testo pieno https://arxiv.org/html/2409.15647v3 — **[C]**
- Relaxed Recursive Transformers, Bae et al. 2024 — https://arxiv.org/abs/2410.20672 , testo pieno https://arxiv.org/html/2410.20672v2 — **[C]**
- Scaling up Test-Time Compute with Latent Reasoning (Huginn-3.5B), Geiping et al. 2025 — https://arxiv.org/abs/2502.05171v2 , testo pieno https://arxiv.org/html/2502.05171v2 — **[C]**
- Reasoning with Latent Thoughts, Saunshi et al. ICLR 2025 — https://arxiv.org/abs/2502.17416 , testo pieno https://arxiv.org/html/2502.17416v1 e https://ar5iv.labs.arxiv.org/html/2502.17416 — **[C]**
- Hierarchical Reasoning Model, Wang et al. 2025 — https://arxiv.org/abs/2506.21734 , testo pieno https://arxiv.org/html/2506.21734v3 — **[C]**
- Mixture-of-Recursions, Bae et al. 2025 — https://arxiv.org/abs/2507.10524 , testo pieno https://arxiv.org/html/2507.10524v2 — **[C]**
- Less is More: Recursive Reasoning with Tiny Networks (TRM), Jolicoeur-Martineau 2025 — https://arxiv.org/abs/2510.04871 , testo pieno https://arxiv.org/html/2510.04871v1 — **[C]**
- Scaling Latent Reasoning via Looped Language Models (Ouro), 2025 — https://arxiv.org/abs/2510.25741 , testo pieno https://arxiv.org/html/2510.25741v1 — **[C]**
- Think-at-Hard, 2025 — https://arxiv.org/abs/2511.08577 , testo pieno https://arxiv.org/html/2511.08577v1 — **[C]**
- Two-Scale Latent Dynamics for Recurrent-Depth Transformers, 2025 — https://arxiv.org/pdf/2509.23314 , testo pieno https://arxiv.org/html/2509.23314 — **[C]**
- Parcae: Scaling Laws For Stable Looped Language Models, 2026 — https://arxiv.org/abs/2604.12946 , testo pieno https://arxiv.org/html/2604.12946v1 — **[C]**
- Loop, Think, & Generalize, 2026 — https://arxiv.org/abs/2604.07822 , testo pieno https://arxiv.org/html/2604.07822v1 — **[C]**
- A Mechanistic Analysis of Looped Reasoning Language Models, 2026 — https://arxiv.org/pdf/2604.11791 , testo pieno https://arxiv.org/html/2604.11791v1 — **[C]**
- Looped SSMs: Depth-Recurrence and Input Reshaping, 2026 — https://arxiv.org/pdf/2605.16048 , testo pieno https://arxiv.org/html/2605.16048 — **[C]**
- LOTUS / Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers, 2026 — https://arxiv.org/abs/2606.31779 , testo pieno https://arxiv.org/html/2606.31779v2 — **[C]**
- Looped Transformers under the Jacobian Lens, 2026 — https://arxiv.org/html/2609.01924 — **[C]**
- Tiny Recursive Reasoning with Mamba-2 Attention Hybrid, 2026 — https://arxiv.org/abs/2602.12078 , testo pieno https://arxiv.org/html/2602.12078v1 — **[C]** su architettura e risultati; **[P]** sul dettaglio "L_cycles = 4-66"

### Fonti primarie lette solo in abstract

- Looped Transformers as Programmable Computers, Giannou et al. 2023 — https://arxiv.org/abs/2301.13196 — **[P]**
- SpikeGPT, Zhu et al. 2023 — https://arxiv.org/abs/2302.13939 — **[P]**
- STARS / Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning, 2026 — https://arxiv.org/abs/2605.26733 — **[P]**
- Training-Free Looped Transformers, 2026 — https://arxiv.org/abs/2605.23872 — **[P]** (abstract citato verbatim)
- Fixed-Point Reasoners (FPRM), 2026 — https://arxiv.org/abs/2606.18206 — **[P]**
- Looped Latent Attention: Cross-Loop KV Compression, 2026 — https://arxiv.org/abs/2607.15456 — **[P]** (numeri dall'abstract esteso)
- Think Shallow, Solve Deep, 2026 — https://arxiv.org/abs/2608.18222 — **[P]**
- Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping, 2026 — https://arxiv.org/abs/2606.29983 — **[P]**
- Fractal basins trap latent reasoning, 2026 — https://arxiv.org/abs/2609.04963 — **[P]** (abstract verbatim)
- RecurTrace, 2026 — https://arxiv.org/abs/2609.03379 — **[P]**
- T-LoopFormer, 2026 — https://arxiv.org/abs/2609.15160 — **[P]**
- URCHIN, 2026 — https://arxiv.org/abs/2609.13899 — **[P]**

### Fonti secondarie / cataloghi

- Awesome-Loop-Models (catalogo) — https://github.com/huskydoge/Awesome-Loop-Models — **[P]**, usato solo per individuare i titoli, non per i numeri
- Recurrent Looped Transformer, pagina di progetto — https://yifanzhang-pro.github.io/recurrent-looped-tranformer/ e https://github.com/yifanzhang-pro/recurrent-looped-tranformer — **[P]**
- Huginn-0125 model card — https://huggingface.co/tomg-group-umd/huginn-0125 — **[P]**
- Ouro-2.6B model card — https://huggingface.co/ByteDance/Ouro-2.6B — **[P]**

### Citati in lista ma non approfonditi — **[NV]** sui contenuti

Memory-Efficient Looped Transformer (2605.07721); Latent Recurrent Transformer (2605.26797); Looped Diffusion Language Models (2605.26106); Solve the Loop: Attractor Models (2605.12466); Thinking Deeper, Not Longer (2603.21676); Adaptive Depth in Looped Transformers (2607.20519); Persistent Recurrent Memory Between Transformer Layers (2609.17251); LoopSpec (2609.17184); Latent Recurrent Thoughts (2609.01117); Right Direction, Wrong Step (2609.16665); Looped World Models (2606.18208); Tiny Autoregressive Recursive Models (2603.08082); Test-time Adaptation of Tiny Recursive Models (2511.02886); A Survey on Latent Reasoning (2507.06203); Are Latent Reasoning Models Easily Interpretable? (2604.04902); SpikeDecoder (2606.12287).
