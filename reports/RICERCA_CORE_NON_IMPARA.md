# Perché il nucleo connettomico non impara — ricerca bibliografica e piano sperimentale

Data: 18 settembre 2026
Autore: ricerca su richiesta del committente (agente Claude Opus 5, sessione flytollm)
Ambito: sezioni A–F richieste + lista esperimenti ordinati per costo GPU.

Convenzioni di marcatura usate in tutto il documento:
- **[C]** = confermato leggendo la fonte primaria (abstract o testo completo scaricato in questa sessione).
- **[P]** = parziale o secondaria (riassunto di motore di ricerca, pagina di proceedings, blog dell'autore, preprint invece della versione finale).
- **[NV]** = non verificato in letteratura: derivazione aritmetica mia a partire dai numeri forniti dal committente, oppure inferenza di raccordo. Va trattata come ipotesi da testare, non come fatto pubblicato.

---

## 0. Sintesi esecutiva

La domanda del committente è: *perché il nucleo è l'unica parte che non impara, e come si fa a farlo imparare davvero?*

La risposta breve, che il resto del documento argomenta con fonti, è che **non c'è una sola causa ma cinque cause indipendenti che si sommano**, e che almeno tre di esse sono prevedibili dalla teoria prima ancora di misurare. In ordine di importanza stimata:

1. **Regime lazy strutturale.** Un substrato ricorrente enorme e fisso con interfacce dense addestrabili è, per costruzione, il caso canonico del "lazy training" di Chizat & Bach e del "kernel regime" di Woodworth: il modello si comporta come la propria linearizzazione attorno all'inizializzazione e l'apprendimento si concentra nella parte lineare/di lettura. Schuessler et al. (2020) mostrano quantitativamente il punto chiave per le RNN: **più grande è la connettività casuale iniziale, più piccola è la modifica necessaria** — ‖ΔW‖ *decresce* al crescere dell'ampiezza g della connettività iniziale [C]. Con 166.700 neuroni e 2,75 M archi il nucleo è enormemente sovradimensionato rispetto al compito che gli resta da fare, quindi la modifica richiesta è minuscola. Lo 0,30% misurato non è un bug: è la predizione della teoria.

2. **Gradient starvation.** Le interfacce (embedding 4096×256, testa 4096×256, attenzione esterna, lettura a 77 gruppi) sono un percorso *facile e veloce* che cattura la loss. Pezeshki et al. (2021) formalizzano proprio questo: quando una feature viene imparata più in fretta delle altre, il contributo di gradiente degli esempi che la contengono si riduce, e le altre feature restano non scoperte [C]. Il vostro dato (3) — con lr sinaptico 10–300× i pesi si muovono fino al 15% e la CE *non cambia* — è la firma sperimentale di questo: **la loss è piatta nelle direzioni del nucleo**, non è che il gradiente sia semplicemente piccolo.

3. **Estinzione strutturale del gradiente nel substrato spiking.** g(peso) = δ(bersaglio) × spike(sorgente), e l'81% dei neuroni non spara mai: l'81% delle righe della matrice di gradiente è *esattamente* zero. È il "dead neuron problem" della letteratura SNN [P], e la sua causa a monte è l'inizializzazione: con soglia 1 e somma dei pesi entranti normalizzata a 0,5 il potenziale di membrana è disegnato per stare sotto soglia in media e senza controllo delle fluttuazioni. Rossbroich, Gygax & Zenke (2022) danno il criterio esatto che manca: ξ ≡ (θ − μ_U)/σ_U con 1 ≤ ξ ≤ 3 e σ_U tra 1/3 e 1 [C]; senza questo criterio le reti profonde restano quiescenti e collassano (7 strati su SHD: 80,9% con init a fluttuazione contro 4,5% con Kaiming) [C].

4. **Vincolo di Dale via softplus: passo effettivo ridotto ~34×.** A w ≈ 0,03 si ha dw/d(raw) = sigmoid(raw) ≈ 0,0296 [NV, aritmetica verificabile]. Adam normalizza il passo nello spazio *raw*, quindi il passo nello spazio dei *pesi* è ~1/34 del nominale. Combinando questo con la frazione attiva del 19% e l'accumulo a random walk su 8000 update si predice una variazione relativa di norma ≈ 0,38%, contro lo 0,30% misurato (§D.4). L'immobilità misurata è quindi **quantitativamente spiegata** e non richiede ipotesi esotiche.

5. **Rapporto segnale/rumore del gradiente bassissimo.** TBPTT 8 token e batch 2 significano ~16 bersagli per update contro 2,75 M parametri sinaptici. Per il nucleo il gradiente è quasi tutto rumore; Adam divide per la radice del secondo momento, che in regime dominato dal rumore è quasi tutto rumore, e la componente di segnale dell'update diventa una frazione piccola.

Su **grokking**: è *implausibile* nel vostro setup attuale, e la ragione non è "8000 update sono pochi" ma che **manca il meccanismo**. Nanda et al. (2023) mostrano che le tre fasi del grokking sono guidate dalla norma dei pesi e che la fase di cleanup è guidata dal weight decay; senza weight decay il circuito generalizzante non si forma affatto [C/P]. Omnigrok mostra che dataset più grandi "de-grokkano" e che il tempo di generalizzazione scala come γ⁻¹ (inverso del weight decay) [C/P]. Voi avete dataset grande, ~1 passata, weight decay non menzionato. Soprattutto: Barak et al. (2022) chiariscono che nei casi di apprendimento ritardato **SGD sta amplificando un segnale già presente**, non cercando a caso [C] — e la vostra misura (2) dimostra che nel nucleo dopo 8000 update il segnale accumulato vale 0,0006 nat, cioè zero. Questo è già un test di "progresso nascosto" con esito negativo. §E dà i criteri per riconoscere per tempo il caso contrario.

La strada praticabile non è "aspettare", ma **cambiare le condizioni**: scala della lettura, learning rate per modulo, inizializzazione a fluttuazione, parametri per neurone invece che per sinapsi, tracce di eleggibilità, bottleneck sulle interfacce, weight decay. §Esperimenti li ordina per costo.

---

## A. Regime lazy contro regime rich

### A.1 Chizat & Bach — il lazy training è una scelta di scala, non una proprietà delle reti

Il lavoro di riferimento è *On Lazy Training in Differentiable Programming* (Chizat, Oyallon, Bach; arXiv:1812.07956, NeurIPS 2019).

Dall'abstract [C]: il fenomeno del lazy training «is not specific to over-parameterized neural networks, and is due to a choice of scaling, often implicit, that makes the model behave as its linearization around the initialization, thus yielding a model equivalent to learning with positive-definite kernels».

Il formalismo [C]: si studia il modello scalato

    F_α(w) := (1/α²) · R(α · h(w))

dove h è la mappa parametro→funzione e R la loss. La quantità che controlla la "pigrizia" è

    κ_h(w₀) := ‖h(w₀) − y*‖ · ‖D²h(w₀)‖ / ‖Dh(w₀)‖²

e il lazy training è il caso in cui κ_h(w₀) ≪ 1: «lazy training refers to the case where the differential of h does not sensibly change while the loss enjoys a significant decrease» [C].

Il teorema di prossimità [C] dà, per h(w₀) = 0:

    sup_{t∈[0,T]} ‖w_α(t) − w₀‖ = O(1/α)
    sup_{t∈[0,T]} ‖w_α(t) − w̄_α(t)‖ = O(1/α²)

dove w̄ è la traiettoria del modello linearizzato. Cioè: **al crescere della scala di uscita α, il movimento relativo dei parametri scala come 1/α e la traiettoria coincide con quella del modello linearizzato a meno di 1/α²**. La moltiplicazione della loss per 1/α² è la riparametrizzazione temporale corretta, cioè equivale a scalare il learning rate [C].

Empiricamente il prezzo è alto: su VGG-11/CIFAR-10, «For large α, we obtain a low limit training accuracy and do not observe overfitting… This behavior is due to a poorly conditioned linearized model» [C]; i valori riportati nell'estrazione sono ~61,7% di accuratezza test nel regime lazy estremo contro ~89,7% nel training standard [P].

**Traduzione al vostro caso.** Voi non avete un α esplicito, ma ne avete due impliciti: (a) la scala di uscita della lettura (2.241 neuroni → 77 gruppi → 256 → testa 4096×256) e (b) l'enorme ridondanza del nucleo, che rende il contributo del singolo peso alla funzione ~1/2,75M. Entrambi spingono κ_h verso zero per il sottoinsieme di parametri "nucleo". Il nucleo è nel regime lazy, le interfacce no.

### A.2 Woodworth et al. — la scala dell'inizializzazione è il quadrante

*Kernel and Rich Regimes in Overparametrized Models* (Woodworth, Gunasekar, Lee, Moroshko, Savarese, Golan, Soudry, Srebro; arXiv:2002.09277, COLT 2020) [C].

Nel modello lineare diagonale a due strati, f(w,x) = Σᵢ (w₊,ᵢ² − w₋,ᵢ²)xᵢ con inizializzazione simmetrica w₊(0) = w₋(0) = α·w₀, il risultato principale è [C]:

- α → ∞ (grande init): **regime kernel**, bias implicito di norma ℓ₂ minima —
  lim_{α→∞} β_α^∞ = argmin_{Xβ=y} ‖β‖₂
- α → 0 (piccola init): **regime rich**, bias implicito di norma ℓ₁ minima —
  lim_{α→0} β_α^∞ = argmin_{Xβ=y} ‖β‖₁

E, punto operativamente decisivo [C]: «α must become exponentially small [to approximate the ℓ₁ solution], but polynomially large α suffices for the ℓ₂ solution». **Entrare nel regime rich è molto più difficile che restare in quello kernel**: basta una scala moderatamente grande per cadere nel kernel, mentre servono scale drasticamente piccole per uscirne. Questo spiega perché tentativi timidi (fattore 2–3 sulle scale) tipicamente non spostano nulla, ed è la ragione per cui gli esperimenti proposti in §Esperimenti usano fattori ÷10 e ÷100 e non ÷2.

### A.3 Yang & Hu — abc-parametrizzazione e μP: la dicotomia è netta

*Feature Learning in Infinite-Width Neural Networks* (Yang & Hu; arXiv:2011.14522, ICML 2021).

Dall'abstract [C]: «we show that the standard and NTK parametrizations of a neural network do not admit infinite-width limits that can learn features».

La abc-parametrizzazione [C]: per ogni strato l,
- W_l = n^{−a_l} · w_l, con w_l parametro effettivamente addestrato;
- w_l ~ N(0, n^{−2b_l});
- learning rate η · n^{−c}.

La **Dicotomia Dinamica** [C]: una abc-parametrizzazione stabile e non banale o ammette feature learning o è nel regime kernel, mai entrambi. Il discriminante è un esponente r: **feature learning se r = 0, regime kernel se r > 0**, con

    r = min(a_{L+1} + b_{L+1}, 2a_{L+1} + c) + c − 1 + min_{l=1..L}[2a_l + 𝟙(l=1)]

La **Maximal Update Parametrization (μP)** è data dalla tabella [C]: c = 0; b_l = 1/2 per tutti gli strati; a₁ = −1/2 (strato di ingresso); a_l = 0 per gli strati nascosti; a_{L+1} = +1/2 (strato di uscita).

Il contenuto pratico di quella tabella, per i nostri scopi: **lo strato di uscita va moltiplicato per un fattore ~1/larghezza**, mentre gli strati nascosti restano a moltiplicatore O(1). È esattamente l'opposto di quello che succede nel vostro modello, dove la lettura e la testa sono dense, a piena scala, e per giunta addestrate con Muon a lr 1e-3, cioè 10× il lr del nucleo. La parametrizzazione standard «can only allow O(1/width) learning rate (i.e. c=1), so as to avoid blowup, and yield kernel limits» [C].

**Come si spinge una rete nel regime rich** — sintesi delle tre fonti sopra:
1. ridurre la scala dell'*uscita* (moltiplicatore della lettura/testa), che è il modo più diretto e quello con la teoria più pulita (Chizat: 1/α; μP: a_{L+1} = 1/2);
2. ridurre la scala dell'*inizializzazione* dei parametri che devono imparare — ma attenzione, per i pesi sinaptici questo è in tensione con la biologia del connettoma e con la normalizzazione a 0,5;
3. alzare il learning rate *relativo* del modulo che deve imparare e abbassare quello del modulo che assorbe la loss (§C);
4. weight decay, che spinge fuori dal regime lazy (Kumar et al., §E).

### A.4 Il caso ricorrente: Schuessler et al. 2024 — l'ampiezza della lettura è la manopola

*Aligned and oblique dynamics in recurrent neural networks* (Schuessler, Mastrogiuseppe, Ostojic, Barak; eLife 2024, 10.7554/eLife.93060; preprint arXiv:2307.07654).

Definizioni [C]: dinamica **allineata** = «the output weights and the largest PCs are strongly correlated»; dinamica **obliqua** = «the output weights and the largest PCs are poorly correlated». E: «the magnitude of output weights… can serve as a control knob» tra i due regimi [C]. Pesi di uscita grandi → dinamica obliqua; pesi di uscita piccoli → dinamica allineata.

Il punto più rilevante per voi [C]: «lazy solutions were observed for large output weights in feedforward networks… these are unstable for recurrent networks and are replaced in a second phase of learning by oblique solutions». Cioè nelle RNN il regime lazy puro non è un punto fisso: se si lascia correre, una seconda fase di apprendimento lo sostituisce. Osservano anche che «learning only weakly changes the norm of the output weights» [C], quindi il regime è deciso all'inizializzazione e non si auto-corregge.

Questa è, in tutta la letteratura consultata, **l'unica base teorica seria per sperare in un apprendimento tardivo del nucleo**. Va però notato che nei loro esperimenti la "seconda fase" arriva su scale temporali paragonabili alla prima, non 10× più tardi, e che il loro readout non è seguito da una testa softmax 4096-dimensionale addestrabile. Vedi §E per la valutazione complessiva.

### A.5 Contorno: rich vs lazy nelle rappresentazioni neurali

Flesch, Saxe & Summerfield, *Rich and lazy learning of task representations in brains and neural networks* (bioRxiv 2021.04.23.441128) [P]: definiscono lazy = espansione della dimensionalità per proiezioni casuali sullo strato nascosto; rich = le unità nascoste acquisiscono rappresentazioni strutturate che privilegiano le feature rilevanti. Trovano evidenza di codifica coerente col regime rich in corteccia prefrontale di macaco e in fMRI umana. Utile come cornice concettuale: il vostro nucleo sta facendo *esattamente* il lavoro lazy (proiezione casuale non lineare ad alta dimensione), che è precisamente il ruolo di un reservoir.

---

## B. RNN allenate contro reservoir

### B.1 Schuessler et al. 2020 — perché un grande substrato casuale rende superflua la modifica

*The interplay between randomness and structure during learning in RNNs* (Schuessler, Mastrogiuseppe, Dubreuil, Ostojic, Barak; arXiv:2006.11036, NeurIPS 2020). Questo è il lavoro più direttamente applicabile alla vostra misura (1) e (2).

Risultati [C]:
- I cambiamenti di connettività dopo training con gradiente non vincolato sono **matrici a basso rango**. Il "rango funzionale" (rango a cui la loss scende sotto il 5% del valore iniziale) è 2 per il flip-flop, 4 per il task di Mante, 6 per il task di Romo; «For all three tasks, it drops to a value close to zero before rank 10».
- Troncare ΔW ai primi ranghi preserva la prestazione: «The loss after truncation indeed drops to zero at rank 2» per il flip-flop.
- **La norma di ΔW decresce al crescere dell'ampiezza g della connettività casuale iniziale**: «The norm of ΔW not only remains unchanged for increasing N…, but further decreases with increasing g». La spiegazione analitica passa per β = 1/(1 − g²): il coefficiente del primo ordine cresce con g, quindi serve una modifica totale più piccola.
- «The final connectivity W = W₀ + ΔW is dominated by the initial connectivity» quando g è grande, dato che ‖W₀‖ = √N · g.

**Implicazione diretta.** Il vostro nucleo ha N = 166.700 e una connettività iniziale con norma grande (pesi ~0,03 ma 2,75 M archi, con somma entrante normalizzata a 0,5 per neurone). La teoria dice che la modifica *funzionalmente necessaria* è (i) a basso rango, (ii) di norma piccola, e (iii) tanto più piccola quanto più è grande il substrato casuale. Uno 0,30% di variazione relativa in norma è quindi **coerente con un apprendimento reale a basso rango**, non necessariamente con "nessun apprendimento". Ma la vostra misura (2) — reset a init cambia la CE di 0,0006 nat — esclude anche questa lettura ottimistica: se ci fosse una componente a basso rango funzionalmente utile, azzerarla costerebbe. Quindi: nucleo = reservoir, confermato dai vostri dati e non contraddetto dalla teoria.

**Conseguenza per la diagnostica:** il modo corretto di misurare se il nucleo sta imparando *non* è la norma relativa di ΔW (che sarà sempre minuscola) ma (a) il rango effettivo di ΔW, (b) la CE dopo troncamento di ΔW a rango r, (c) la CE dopo reset a init. Voi avete già (c); (a) e (b) sono gratuiti e vanno aggiunti.

### B.2 Reservoir computing contro BPTT: quando conviene allenare i ricorrenti

- Vlachas et al., *Backpropagation algorithms and Reservoir Computing in RNNs for the forecasting of complex spatiotemporal dynamics* (Neural Networks 2020; arXiv:1910.05266) [P]: quando l'intero stato è osservabile in training, il reservoir computing **batte** le varianti BPTT in accuratezza predittiva e statistiche di lungo periodo, a costo di training molto inferiore; ma con dati ridotti (osservazione parziale) i reservoir grandi diventano instabili e divergono più facilmente delle reti BPTT. Morale: il reservoir non è una scorciatoia degradata, è competitivo — ed è probabilmente quello che state ottenendo. Il vostro GRU 2×256 a 3,43 contro 3,54 è compatibile con il reservoir che perde poco ma perde.
- DePasquale, Cueva, Rajan, Escola, Abbott, *full-FORCE: A target-based method for training recurrent networks* (PLoS ONE 2018; arXiv:1710.03070) [P]: modificando **l'intera matrice ricorrente** invece del solo readout si ottengono reti che svolgono il compito «with fewer neurons and greater noise robustness than traditional least-squares (FORCE) approaches». Il meccanismo è cruciale: si introduce una **seconda rete durante il training che fornisce le dinamiche "target" interne**. Cioè: allenare i ricorrenti *aiuta davvero*, ma serve un bersaglio per le dinamiche interne, non solo l'errore in uscita. Questa è la giustificazione dell'esperimento E12 (§Esperimenti).

### B.3 RNN vincolate dal connettoma: cosa si allena davvero

- **Lappalainen et al. 2024**, *Connectome-constrained networks predict neural activity across the fly visual system* (Nature 632, 2024; preprint bioRxiv 2023.03.11.532232; codice TuragaLab/flyvis). Dal preprint [C]: «Synapse counts, N_titj, and signs, σ_titj, from reconstruction and neurotransmitter and receptor profiling were kept fixed». Ciò che si allena sono **costanti di tempo e potenziali di riposo per tipo cellulare (130 parametri)** e **fattori di scala sinaptica per coppia di tipi cellulari (604 parametri)**, per un totale di **734 parametri liberi** su un modello di «45,669 neurons and 1,513,231 connections». Training: BPTT su predizione di flusso ottico (Sintel), ottimizzatore con momenti adattivi [C]. (I numeri della versione Nature finale non sono stati verificati in questa sessione: [P].)

  **Questo è il precedente più importante per voi.** Il lavoro di riferimento sulle reti vincolate dal connettoma **non allena le singole sinapsi**: allena parametri a livello di neurone e di tipo cellulare, in numero minuscolo. Il vostro tentativo di far imparare 2,75 M pesi sinaptici individuali non ha precedenti di successo in questa letteratura; il tentativo di far imparare ~10²–10⁵ parametri per neurone/gruppo sì.

- **Beiran & Litwin-Kumar 2025**, *Prediction of neural activity in connectome-constrained recurrent networks* (Nature Neuroscience 28, 2561–2574; preprint bioRxiv 2024.02.22.581667) [P]: un "allievo" con la stessa connettività del "maestro" ma parametri biofisici diversi. Risultato: «a connectome is often insufficient to constrain the dynamics of networks that perform a specific task», mentre registrazioni da un piccolo sottoinsieme di neuroni rimuovono la degenerazione. Cioè: **la topologia da sola lascia una degenerazione enorme nello spazio delle funzioni**. Questo è la spiegazione teorica della vostra misura (5): se la topologia non vincola la dinamica, un grafo null rimescolato con gli stessi gradi non ha ragione di comportarsi peggio.

---

## C. Gradient starvation, shortcut learning, squilibrio di velocità fra moduli

### C.1 Pezeshki et al. — il meccanismo formale

*Gradient Starvation: A Learning Proclivity in Neural Networks* (Pezeshki, Kaba, Bengio, Courville, Precup, Lajoie; arXiv:2011.09468, NeurIPS 2021).

Definizione [C]: «Feature i starves the gradient for feature j if dz_j*/d(s_i²) < 0» — aumentare la forza di una feature *danneggia* l'apprendimento di un'altra. Il meccanismo passa per il termine **U·S²·Uᵀ** nelle equazioni delle dinamiche di training: quando le feature hanno forze diverse (valori singolari diversi in S) si crea interdipendenza, e «when one feature is learned faster than the others, the gradient contribution of examples containing that feature is diminished» [C].

Rimedio proposto, **Spectral Decoupling** [C]: si sostituisce il weight decay ℓ₂ con una penalità sui *logit*,

    log[1 + exp(−Y·ŷ)] + (λ/2)·‖ŷ‖²

L'effetto è di eliminare il termine S² accoppiante. E il punto cruciale per voi: **il weight decay ℓ₂ standard non risolve il problema**, perché preserva la struttura di accoppiamento U·S²·Uᵀ; sperimentalmente «training longer or using different regularizers, including weight decay… do not encourage the network to learn» le feature trascurate [C].

Quel "training longer non basta" è una risposta diretta e negativa all'ipotesi "forse a 50.000 update succede qualcosa".

### C.2 Lo squilibrio fra moduli: l'analogia multimodale

Il vostro sistema è di fatto multi-percorso: attenzione esterna (percorso facile, denso, 4 teste, cache 128) + nucleo spiking (percorso difficile, sparso, con gradiente troncato). La letteratura multimodale ha studiato esattamente questa patologia:

- Wang, Tran, Feiszli, *What Makes Training Multi-modal Classification Networks Hard?* (CVPR 2020) [P]: la rete multimodale spesso fa **peggio** della migliore rete unimodale, perché «different modalities overfit and generalize at different rates, so training them jointly with a single optimization strategy is sub-optimal». Rimedio: **Gradient-Blending**, che calcola una miscela ottimale dei rami in base al loro comportamento di overfitting.
- Peng, Wei, Deng, Wang, Hu, *Balanced Multimodal Learning via On-the-fly Gradient Modulation* (CVPR 2022, oral) [P]: modulazione adattiva del gradiente per ciascuna modalità monitorando la discrepanza dei rispettivi contributi all'obiettivo, più iniezione dinamica di rumore.

Trasposizione: **modulare on-the-fly il guadagno del percorso attenzione rispetto al percorso nucleo, in base al loro contributo misurato alla loss**, è un rimedio pubblicato e direttamente applicabile.

### C.3 Congelare o rallentare la lettura: evidenza diretta

- Lan, Liu, Zhou, Yosinski, *LCA: Loss Change Allocation for Neural Network Training* (arXiv:1909.01440, NeurIPS 2019) [C]. Introduce la **Loss Change Allocation**: «The difference in loss produced by one training iteration may be decomposed into K individual Loss Change Allocation, or LCA, components», cioè un'attribuzione per-parametro del cambiamento di loss lungo la traiettoria. Risultati rilevanti:
  - «The first and last layers consistently hurt training» in CIFAR-ResNet con SGD (LCA totale positiva, p < 10⁻⁴);
  - «Freezing the last layer at its initialization improves training performance (and test performance…), with p-values < 0.001 for both train loss and test loss»;
  - «Decreasing the learning rate of the last layer by 10x results in similar behavior as freezing it»;
  - «The exact moments where learning peaks are curiously synchronized across layers» (p < 10⁻⁶).

  Questo dà **sia il rimedio** (congelare o rallentare 10× la testa) **sia la diagnostica** (LCA per modulo: il modulo aiuta, danneggia o non fa nulla?).

- Hoffer, Hubara, Soudry, *Fix your classifier: the marginal value of training the last weight layer* (arXiv:1801.04540, ICLR 2018) [P]: il classificatore può essere **fissato**, a meno di una costante globale di scala, con perdita di accuratezza nulla o marginale nella maggior parte dei task; implementano matrice ortogonale casuale o matrice di Hadamard, imparando solo una temperatura. Cioè: una testa 4096×256 completamente addestrabile **non serve**, e toglierla libera capacità di apprendimento (e gradiente) per il resto.

- Kosson, Messmer, Jaggi, *Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks* (ICML 2024; arXiv:2305.17212) [P]: il weight decay porta l'ampiezza attesa e la rotazione angolare del vettore di pesi di un neurone a uno stato stazionario ("rotational equilibrium") che **omogeneizza il learning rate efficace fra strati e neuroni**. Proxy pratico del learning rate efficace = rotazione media per step, cioè l'aggiornamento *relativo* ‖Δθ‖/‖θ‖. È la metrica giusta per confrontare la velocità di moduli con parametrizzazioni diverse (softplus vs denso, Adam vs Muon).

### C.4 Il fattore Muon/Adam

Il vostro schema usa Muon a lr 1e-3 sulle matrici dense di attenzione e lettura, Adam a lr 1e-4 sul resto. Muon (Jordan et al., 2024) ortogonalizza il momento con iterazioni di Newton–Schulz, il che equivale a discesa più ripida in **norma spettrale** [P]: l'update ha norma spettrale ~lr indipendentemente dalla magnitudine del gradiente. Conseguenza [NV]: le matrici dense si muovono a velocità *garantita* per step, mentre il nucleo si muove a velocità proporzionale al gradiente, ulteriormente compressa 34× dal softplus (§D.4). L'asimmetria di velocità relativa fra i due moduli è quindi di **almeno due ordini di grandezza già per costruzione dell'ottimizzatore**, prima ancora di considerare la sparsità dell'attività.

---

## D. Reti spiking: neuroni morti e flusso del gradiente surrogato

### D.1 Il problema dei neuroni morti / silenti

Eshraghian et al., *Training Spiking Neural Networks Using Lessons From Deep Learning* (Proceedings of the IEEE 111(9):1016–1054, 2023) [P]: il "dead neuron problem" è definito come la situazione in cui la soluzione analitica ∂S/∂U ∈ {0, ∞} produce un gradiente che non consente apprendimento; i gradienti surrogati sono «the most broadly adopted solution».

Ma il gradiente surrogato risolve solo metà del problema: risolve la derivata della funzione di spike rispetto al potenziale. **Non risolve il fattore presinaptico.** Nel vostro caso il gradiente del peso è g(bersaglio) × spike(sorgente): se il neurone sorgente non spara mai, il gradiente è zero *a prescindere dal surrogato*. È esattamente ciò che avete misurato (81% dei neuroni muti → solo 19% degli archi con gradiente potenziale).

Perez-Nieves & Goodman, *Sparse Spiking Gradient Descent* (NeurIPS 2021; arXiv:2105.08810) [P] quantificano la stessa cosa dal lato opposto (sfruttandola per accelerare): l'attività media «is never above 2% on any dataset and is as low as 1.06% in the F-MNIST dataset», quindi in media non serve calcolare più del 2% dei fattori di ∇W. Cioè: **la sparsità estrema del gradiente nelle SNN è la norma, non una vostra anomalia** — ma nelle loro reti l'apprendimento funziona lo stesso perché le reti sono piccole e ogni neurone *può* sparare. Nel vostro caso l'81% non può.

### D.2 Inizializzazione a fluttuazione — il rimedio con la teoria più pulita

Rossbroich, Gygax, Zenke, *Fluctuation-driven initialization for spiking neural network training* (Neuromorphic Computing and Engineering 2(4), 2022; arXiv:2206.10226).

Criterio [C]: si definisce

    ξ ≡ (θ − μ_U) / σ_U

con raccomandazione **1 ≤ ξ ≤ 3**; per distribuzioni di peso a media nulla questo si traduce in un'ampiezza di fluttuazione desiderata **1/3 ≤ σ_U ≤ 1**. La logica [C]: «the bulk of the Gaussian distribution has to lie below the firing threshold» mantenendo però «a non-vanishing probability to cross threshold to ensure some baseline levels of spiking activity».

Risultati quantitativi [C]:
- Rete a 7 strati su SHD: **4,5% ± 0,0** con inizializzazione Kaiming contro **80,9% ± 1,2** con init a fluttuazione (σ_U = 1).
- CIFAR-10 (4 strati): 10,0% contro 65,6%. DVS-Gesture (8 strati): 9,1% contro 86,4%.
- Diagnosi del fallimento: «networks initialized with σ_U = 0.2, in which all but the first hidden layer were quiescent», e «weight updates vanished in deeper layers, caused by the lack of presynaptic activity» [C].
- «initialization does not affect the magnitude of the gradient in the readout layer, but can change the magnitude of the gradient by two orders of magnitude in the hidden layer» [C].

Quest'ultima frase descrive **letteralmente la vostra situazione**: le interfacce ricevono gradiente pieno, il nucleo lo riceve attenuato di ordini di grandezza, e la causa è l'inizializzazione.

Nota sul surrogato [C]: usano h(x) = 1/(β|x| + 1)² con β = 20, e osservano che «the surrogate derivative for neurons at rest (0.023 for β = 20)» — cioè a riposo il surrogato vale ~2% del valore di picco. Il vostro surrogato è sigmoide 4(u−1): a u = 0 (riposo) vale σ'(−4) = σ(−4)(1−σ(−4)) ≈ 0,0180, moltiplicato per il fattore di scala 4 → ~0,0719 [NV, aritmetica]. Non è zero; il vostro collo di bottiglia primario resta il fattore **spike(sorgente)** binario, non la derivata surrogata.

**Osservazione sulla vostra inizializzazione.** Somma dei pesi entranti normalizzata a 0,5 con soglia 1 è un criterio sulla **media** del drive, non sulla **fluttuazione**. Secondo il criterio di Rossbroich, un'inizializzazione che controlla μ_U ma lascia σ_U libera (e verosimilmente ≪ 1/3, dato che con 2,75M/166,7k ≈ 16,5 archi entranti medi e pesi ~0,03 le fluttuazioni sono piccole) produce esattamente reti quiescenti. **Questo è il singolo intervento a più alta leva e a costo più basso disponibile.**

### D.3 Il surrogato: forma irrilevante, scala rilevante

Zenke & Vogels, *The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks* (Neural Computation 33(4):899–925, 2021; preprint bioRxiv 2020.06.29.176925) [C]:
- Hanno variato la forma (SuperSpike / derivata di sigmoide veloce, derivata di sigmoide standard, lineare a tratti alla Esser). «the surrogate gradient remains largely unaffected by how closely the function resembles the exact derivative as long as it is not a constant».
- Ma: «surrogate gradients are sensitive to the scale of the surrogate derivative. More specifically, when the scale of the surrogate derivative is too large, while either implicit or explicit recurrence are present in the network, the effect on learning can be detrimental». **Rilevante per voi: siete una rete ricorrente profonda (8 sottopassi × TBPTT 8 = fino a 64 passi ricorrenti), quindi allargare troppo il surrogato può essere dannoso, non solo inefficace.** Questo è coerente con la vostra misura (4): con gradiente "allargato" gli archi che si muovono passano dal 3% al 15–23% ma la CE non cambia.
- Usano due forme di regolarizzazione dell'attività per controllare i livelli di spiking negli strati nascosti, e «Activity regularized SNNs can perform with high accuracy down to some critical activity threshold at which their performance degrades rapidly» [C].

### D.4 Il vincolo di Dale e il passo effettivo — la spiegazione quantitativa dello 0,30%

**Il risultato pubblicato di riferimento.** Cornford, Kalajdzievski, Leite, Lamarquette, Kullmann, Richards, *Learning to live with Dale's principle: ANNs with separate excitatory and inhibitory units* (ICLR 2021; bioRxiv 2020.11.02.364968) [C]:
- La parametrizzazione usata è la **rettifica**, θ ← max(0, θ), non exp né softplus.
- Le reti sign-constrained ingenue imparano male: «Constrained weight matrix columns impair learning because they limit the potential solution space»; i modelli ColumnEi «failed to achieve 0% training error within the 50 epochs» su MNIST, mentre le DANN eguagliano gli MLP standard.
- I due accorgimenti che fanno funzionare le DANN [C]: (1) inizializzazione tale che «the inhibition centres and standardizes the excitatory activity» (imparentata con la normalizzazione), (2) «updates to inhibitory neuron parameters should be scaled using corrections based on the Fisher Information matrix», in pratica scalati di 1/n_e rispetto agli eccitatori [P].

Il messaggio: **vincolare il segno non è gratis; richiede una correzione esplicita del passo**. Voi usate softplus, che introduce una distorsione del passo che nessuno ha corretto.

**Derivazione [NV, aritmetica verificabile].** Con w = segno × softplus(raw) e w ≈ 0,03:

    softplus(raw) = 0,03  ⇒  raw = ln(e^{0,03} − 1) = ln(0,0304545) = −3,4914
    dw/d(raw) = sigmoid(raw) = 1/(1 + e^{3,4914}) = 0,029557

Cioè **il passo nello spazio dei pesi è ~1/33,8 del passo nello spazio raw**. Adam normalizza il passo nello spazio raw a ~lr = 1e-4, quindi per arco attivo:

    |Δw| per step ≈ 0,0296 × 1e-4 = 2,96·10⁻⁶   (relativo: 9,9·10⁻⁵)

Accumulo a random walk su T = 8000 update:

    |Δw| ≈ √8000 × 2,96·10⁻⁶ = 2,65·10⁻⁴   (relativo: 0,88%)

Ma solo il 19% degli archi riceve gradiente non nullo; sul totale degli archi la variazione relativa di norma attesa è

    √0,19 × 0,88% = 0,385%

**contro lo 0,30% misurato.** Accordo entro il 30% per un modello a una riga. Analogamente, perché un arco superi 1e-4 di spostamento assoluto servono T_eff ≥ (1e-4 / 2,96·10⁻⁶)² ≈ 1.141 step con gradiente non nullo: dato che gran parte dei neuroni "attivi" sparano raramente, solo gli archi con sorgente ad alta frequenza di scarica accumulano 1.141 eventi su 8000 update — il che spiega perché la frazione che supera 1e-4 sia 2,9% e non 19%.

**Predizione falsificabile e gratuita (nessuna GPU, nessun training):** il 2,9% di archi che si sono mossi sopra 1e-4 dev'essere quasi esattamente l'insieme degli archi la cui sorgente ha il firing rate più alto. Se la correlazione fra "spostamento dell'arco" e "firing rate della sorgente" non è fortissima, questa spiegazione è sbagliata e va cercata altrove.

**Conclusione di §D.4.** L'immobilità dei pesi (misura 1) è spiegata da softplus × sparsità × lr, ed è *rimediabile* (infatti la vostra misura (3) mostra che con lr 10–300× i pesi si muovono fino al 15%). L'inutilità dei pesi (misure 2 e 3) **non** è spiegata da questo ed è il problema vero: quella è §A e §C.

### D.5 Tracce di eleggibilità (e-prop): sostituire lo spike binario con una traccia filtrata

Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein, Maass, *A solution to the learning dilemma for recurrent networks of spiking neurons* (Nature Communications 11:3625, 2020; preprint arXiv:1901.09049) [C].

Fattorizzazione fondamentale [C]:

    dE/dθ_ji = Σ_t L_j^t · e_ji^t                                  (eq. 1)

con vettore di eleggibilità e traccia:

    ε_ji^t = D_j^{t−1} · ε_ji^{t−1} + ∂s_j^t/∂θ_ji                 (eq. 2)
    e_ji^t = (∂z_j^t/∂s_j^t) · ε_ji^t                              (eq. 3)

e, concretamente per neuroni LIF, e_ji^t è «the product of a postsynaptic pseudo derivative h_j^t with the trace ẑ_i^t of the presynaptic spikes» [C]. La proprietà chiave [C]: «a synapse remembers some of its activation history while ignoring inter neuron dependencies» — la traccia ẑ è una memoria decadente della storia presinaptica, **non** l'attività istantanea.

Prestazioni rispetto a BPTT [C]: su TIMIT, e-prop 1 raggiunge 0,651 contro 0,671 di BPTT; su store-recall converge in 50 iterazioni contro 28 di BPTT; su copy-repeat e-prop 3 risolve sequenze di 74 caratteri contro 39 di BPTT completo.

**Perché è rilevante per voi.** Il vostro problema è che g(peso) = δ × spike(sorgente) con spike ∈ {0,1} e quasi sempre 0. Con la traccia di eleggibilità il fattore presinaptico diventa ẑ_i (filtro passa-basso degli spike), quindi **un arco riceve gradiente ogni volta che la sorgente ha sparato di recente, non solo nell'esatto sottopasso corrente**. Con 8 sottopassi per token e TBPTT 8, una traccia con costante di tempo di ~10–30 sottopassi allarga enormemente il supporto del gradiente. Il vostro esperimento (4) — attività presinaptica morbida max(s, sigmoid(4(u−1))) — è la versione *spaziale/istantanea* di questa idea; la traccia è la versione *temporale* e ha una giustificazione teorica molto più solida (è un'approssimazione controllata del vero gradiente, non un ammorbidimento arbitrario).

---

## E. Grokking e apprendimento ritardato: è plausibile qui?

### E.1 I risultati di riferimento

**Power et al. 2022** (grokking originale su dataset algoritmici, piccoli, con molte epoche): non verificato direttamente in questa sessione [NV]; usato solo come riferimento storico.

**Nanda, Chan, Lieberum, Smith, Steinhardt**, *Progress measures for grokking via mechanistic interpretability* (arXiv:2301.05217, ICLR 2023) [C]:
- Misure di progresso: **restricted loss** (si tengono solo le cinque frequenze chiave nei logit, si azzerano le altre) e **excluded loss** (si rimuovono solo le frequenze chiave). Durante la formazione del circuito, «restricted loss starts to fall» e «excluded loss rises», **molto prima** del salto della test loss.
- Tre fasi: memorizzazione (epoche 0–1,4k), formazione del circuito (1,4k–9,4k), cleanup (9,4k–14k), con «each phase of grokking corresponded to an inflection point in the ℓ2-norm of the weights».
- Conclusione centrale [C]: «grokking, rather than being a sudden shift, arises from the gradual amplification of structured mechanisms encoded in the weights, followed by the later removal of memorizing components».
- Con λ (weight decay) = 0 non si ha grokking e la excluded loss non sale, cioè **il circuito generalizzante non si forma affatto** [P].

**Liu, Michaud, Tegmark**, *Omnigrok: Grokking Beyond Algorithmic Data* (arXiv:2210.01117, ICLR 2023) [C]:
- Meccanismo **LU**: «the (reduced) training loss and test loss are L-shaped and U-shaped against weight norm, respectively». Il grokking è il tempo che serve alla norma dei pesi per migrare dalla zona "L bassa, U alta" alla "Goldilocks zone".
- Inizializzazione grande induce grokking: moltiplicando i pesi iniziali per α > 1 **inducono grokking su MNIST** [C].
- Weight decay: «small γ leads to slow generalization (grokking), and large γ leads to faster generalization», con tempo ∝ γ⁻¹ [P].
- Dimensione del dataset: «larger datasets lead to de-grokking» perché la Goldilocks zone si allarga [C].
- Vincolando il modello su una sfera di norma piccola si **elimina** il grokking sui dataset algoritmici [C].

**Kumar, Bordelon, Gershman, Pehlevan**, *Grokking as the Transition from Lazy to Rich Training Dynamics* (arXiv:2310.06110, ICLR 2024) [C]:
- Tesi: «grokking can arise due to a neural network transitioning from lazy training dynamics to a rich, feature learning regime».
- Manopole: **scala dell'uscita α** («large α increases the timescale separation between the decrease in train loss and that of test loss»; «the laziness parameter α controls the timescale of the delay in grokking»); scala di inizializzazione (effetto identico a quello dell'output scaling); scala delle etichette; **weight decay**, che «move us out of the lazy regime».
- Si può **far sparire** il grokking: «we can make that grokking both vanish and more pronounced without touching their weight norm at initialization» agendo su α; su Transformer, «grokking has vanished for α=0.15».
- Statistiche sufficienti tracciate: w̄ = (1/N)Σ w_i e M = (1/N)Σ w_i w_iᵀ, più l'evoluzione dell'NTK dalla forma iniziale K(x,x′) = x·x′ + ε²(x·x′)² verso un kernel dipendente da w̄ e M [C].

**Barak, Edelman, Goel, Kakade, Malach, Zhang**, *Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit* (arXiv:2207.08799, NeurIPS 2022) [C]:
- «SGD gradually amplifies the sparse solution via a Fourier gap in the population gradient, making continual progress that is invisible to loss and error metrics».
- Non è ricerca casuale: il tempo di convergenza si adatta al parametro di sparsità k con scaling n^{O(k)}, incompatibile con una ricerca esaustiva.
- Misura di progresso nascosto proposta: **ρ(w_{0:t}) := ‖w_t − w_0‖_∞**, «motivated by the fact that w_t − w_0 is a linearized estimate for the initial population gradient» [C].

**Schaeffer, Miranda, Koyejo**, *Are Emergent Abilities of Large Language Models a Mirage?* (arXiv:2304.15004, NeurIPS 2023, outstanding paper) [P]: le capacità emergenti appaiono «due to the researcher's choice of metric rather than due to fundamental changes in model behavior with scale»; metriche non lineari o discontinue producono apparente emergenza, metriche lineari o continue producono cambiamenti lisci e prevedibili.

**Zhu, Fu, Zhou, Lin**, *Critical Data Size of Language Models from a Grokking Perspective* (arXiv:2401.10463) [C]: esiste una "critical data size" che segna il passaggio da memorizzazione rapida a generalizzazione lenta; per i dati di linguaggio si osservano «smoother phase transitions occurring at the critical dataset size», cioè transizioni **più lisce** che nei setting algoritmici.

### E.2 Verdetto per il caso connettoma

**Un apprendimento tardivo spontaneo del nucleo, nelle condizioni attuali, è poco plausibile.** Le ragioni, ciascuna con fonte:

1. **Manca il motore.** Nanda: senza weight decay il circuito generalizzante non si forma e la excluded loss non sale [P]; le tre fasi sono scandite dalla norma dei pesi [C]. Voi non avete weight decay dichiarato. Omnigrok: il tempo di grokking scala come γ⁻¹, e γ = 0 dà tempo infinito [P].
2. **Il dataset è troppo grande.** Omnigrok: «larger datasets lead to de-grokking» [C]. TinyStories in streaming è l'opposto del regime piccolo-dato-molte-epoche in cui il grokking vive.
3. **La metrica è continua.** Schaeffer [P]: la CE è una metrica liscia; le transizioni apparenti sono molto più probabili con metriche a soglia. Zhu et al. [C]: sui dati di linguaggio le transizioni sono comunque più lisce.
4. **Il test di progresso nascosto è già stato fatto e ha dato esito negativo.** Barak [C] dice che nei casi di apprendimento ritardato SGD sta **amplificando un segnale già presente**. La vostra misura (2) dice che dopo 8000 update il segnale accumulato nel nucleo vale 0,0006 nat. La vostra misura (3) dice che anche muovendo i pesi del 15% la CE non cambia di nulla a 300 e a 2000 update. Insieme: non c'è nulla da amplificare, la loss è localmente piatta in quelle direzioni.
5. **Gradient starvation non si cura con il tempo.** Pezeshki [C]: «training longer… do not encourage the network to learn» le feature affamate.

**La controevidenza da non ignorare.** Schuessler et al. 2024 [C] affermano che nelle RNN le soluzioni lazy con pesi di uscita grandi sono **instabili** e vengono sostituite da soluzioni oblique in una seconda fase. Questo è l'unico meccanismo pubblicato che prevederebbe un risveglio tardivo. Va però notato che il loro regime non ha (a) un readout sparso vincolato a 2.241 neuroni discendenti, (b) una testa softmax densa 4096×256 addestrabile che può assorbire qualsiasi cambiamento, (c) un percorso attenzione parallelo che bypassa il ricorrente. Ognuno di questi tre stabilizza la soluzione lazy.

**Condizioni che favorirebbero un apprendimento tardivo, se le create voi:** weight decay sulle interfacce (per costringere la funzione a migrare nel nucleo, meccanismo LU di Omnigrok); scala dell'uscita ridotta (Kumar, Schuessler 2024); dataset ridotto e ripetuto (Omnigrok: dataset piccolo favorisce grokking) — quest'ultimo è controintuitivo ma è il regime in cui il fenomeno esiste; bottleneck che rende il percorso facile insufficiente.

### E.3 Come riconoscerlo per tempo — misure di progresso nascosto per il vostro caso

Da monitorare a ogni ~100 update, tutte a costo trascurabile:

| Misura | Fonte | Cosa significa se sale |
|---|---|---|
| ΔCE da reset del nucleo a init (in nat) | vostra misura (2), generalizzata nel tempo | funzione realmente trasferita nel nucleo — **è la metrica ground-truth** |
| ‖W_t − W_0‖_∞ e ‖W_t − W_0‖_F | Barak et al. [C] | stima linearizzata del gradiente di popolazione: progresso nascosto |
| rango effettivo di ΔW e CE dopo troncamento a rango r | Schuessler 2020 [C] | se il rango funzionale scende sotto 10 e la CE tiene, è apprendimento vero a basso rango |
| quota della norma di update del nucleo ‖Δθ_core‖/‖θ_core‖ vs interfacce | Kosson [P], LCA [C] | riequilibrio della velocità fra moduli |
| LCA per modulo | Lan et al. [C] | il nucleo aiuta, danneggia o è neutro |
| rotazione dell'autostruttura dell'NTK ristretto al nucleo | Kumar [C], Atanasov [P] | uscita dal regime lazy (silent alignment) |
| frazione di neuroni con rate > 0, distribuzione di ξ = (θ−μ_U)/σ_U | Rossbroich [C] | il substrato è entrato nel regime a fluttuazione |
| norma ℓ₂ dei pesi delle interfacce | Nanda [C], Omnigrok [C] | i punti di flesso marcano le fasi del grokking |

---

## F. Diagnostiche: come misurare se un modulo "riceve segnale"

### F.1 Quota di norma del gradiente per gruppo di parametri

La misura ingenua ‖g_core‖/‖g_tot‖ è necessaria ma **non sufficiente e potenzialmente fuorviante** in presenza di Adam e Muon, che rinormalizzano. Le tre versioni da registrare insieme:

1. **Quota di gradiente grezzo:** ‖g_m‖₂ / Σ_m ‖g_m‖₂ per ciascun gruppo m ∈ {embedding, iniettore, sinapsi, lettura, testa, attenzione}.
2. **Quota di update effettivo:** ‖Δθ_m‖₂ per step (post-ottimizzatore). Con Muon questa è ~costante per costruzione [P].
3. **Update relativo / "rotazione":** ‖Δθ_m‖₂ / ‖θ_m‖₂, che Kosson et al. identificano come proxy del learning rate efficace [P]. È l'unica delle tre confrontabile fra moduli con parametrizzazioni diverse (softplus vs denso).

Per il nucleo va inoltre distinta la quota nello spazio *raw* da quella nello spazio *pesi*, che differiscono del fattore sigmoid(raw) ≈ 0,0296 (§D.4) [NV].

### F.2 Rapporto segnale/rumore del gradiente fra batch

Metrica standard: dividere il batch in due metà disgiunte, calcolare g₁ e g₂, e registrare per modulo
- **coseno(g₁, g₂)**: se ≈ 0, il gradiente è puro rumore;
- **GSNR** ≈ ‖E[g]‖² / Var[g], stimato con più microbatch.

La letteratura sul GSNR lo usa per limitare gli update dei parametri privi di stima affidabile del gradiente e lo collega al gap di generalizzazione [P]. Nel vostro caso, con batch 2 e TBPTT 8, mi aspetto coseno(g₁,g₂) per il gruppo sinapsi vicinissimo a zero [NV]: se lo è, aumentare il learning rate sinaptico non può funzionare per costruzione (state amplificando rumore), e l'unica leva è aumentare il batch efficace o cambiare il segnale.

### F.3 Rango effettivo degli aggiornamenti

Da Schuessler et al. [C]: calcolare lo spettro singolare di ΔW = W_t − W_0 (per il grafo sparso, sulla matrice sparsa o su proiezioni casuali), riportare il **rango funzionale** = rango minimo r tale che la loss usando W_0 + trunc_r(ΔW) scende sotto una soglia rispetto al valore con ΔW completo. Nei loro task il rango funzionale è 2–6 e la loss è quasi nulla prima di rango 10. Se il vostro ΔW ha rango effettivo alto e diffuso, è rumore; se ha rango basso ma ΔCE ≈ 0, è rumore strutturato ma inutile.

### F.4 CKA fra stato all'init e stato allenato

Kornblith, Norouzi, Lee, Hinton, *Similarity of Neural Network Representations Revisited* (ICML 2019) [P]: l'indice di similarità proposto è equivalente al **centered kernel alignment (CKA)**, e a differenza di altri metodi «can reliably identify correspondences between representations of layers in networks trained from different initializations» e «can reveal network pathology that is not evident from test accuracy alone».

Uso concreto: fissare un set di sonda di ~256 sequenze; raccogliere la matrice di attività del nucleo (spike o potenziali, campionati sui 2.241 discendenti o su un sottoinsieme casuale di 5.000 neuroni) al checkpoint init e al checkpoint t; calcolare CKA(X_init, X_t). **CKA ≈ 1 ⇒ il nucleo è un reservoir immobile**; ogni discesa significativa di CKA è la prova che il nucleo sta cambiando rappresentazione. Questa misura è più sensibile della norma di ΔW perché guarda la funzione, non i parametri.

### F.5 Tangent kernel alignment / silent alignment

- Atanasov, Bordelon, Pehlevan, *Neural Networks as Kernel Learners: The Silent Alignment Effect* (arXiv:2111.00034, ICLR 2022) [P]: il "silent alignment" richiede che il tangent kernel evolva nella sua **autostruttura** mentre è ancora piccolo e prima che la loss scenda apprezzabilmente, crescendo poi solo in scala complessiva; avviene in reti omogenee con inizializzazione piccola e dati sbiancati; il kernel sviluppa un contributo a basso rango nella fase iniziale.
- Baratin, George, Laurent, Hjelm, Lajoie, Vincent, Lacoste-Julien, *Implicit Regularization via Neural Feature Alignment* (AISTATS 2021; codice github.com/tfjgeorge/ntk_alignment) [P]: l'allineamento dinamico delle tangent feature lungo poche direzioni rilevanti per il task è interpretabile come selezione + compressione delle feature.

Uso concreto per voi: costruire l'NTK **ristretto ai parametri sinaptici** su un set di sonda piccolo (es. 64 sequenze × 8 token), tramite prodotti Jacobiano-vettore, e tracciare (a) l'allineamento kernel-bersaglio yᵀKy/(‖K‖‖y‖²), (b) la rotazione dei primi autovettori fra t e t+Δ. Se K_core resta costante in autostruttura, siete in regime kernel e il nucleo non imparerà mai senza un intervento strutturale.

### F.6 Loss Change Allocation

Lan et al. [C]: decomposizione per-parametro (aggregabile per modulo) del cambiamento di loss lungo la traiettoria. È la diagnostica che risponde direttamente alla domanda del committente — *quale modulo riceve segnale e cosa ne fa* — distinguendo tre casi che la norma del gradiente confonde: modulo che aiuta (LCA negativa), modulo che danneggia (LCA positiva, il caso di primo e ultimo strato nelle loro ResNet), modulo inerte (LCA ≈ 0). La mia previsione per il vostro caso [NV]: LCA del nucleo ≈ 0 con varianza alta, LCA fortemente negativa per embedding + testa + attenzione.

### F.7 Il test definitivo, che avete già

Il reset a init con misura del ΔCE (vostra misura 2) è **la diagnostica migliore che esista** per questa domanda, perché è causale e non correlazionale. Va solo (a) resa periodica invece che finale, (b) fatta per modulo e non solo per il nucleo, (c) affiancata dalla versione "troncata a rango r" (§F.3). Un pannello `ΔCE_reset(modulo, t)` in nat, aggiornato ogni 500 update, risponde alla domanda del committente meglio di qualsiasi proxy.

---

## Esperimenti proposti per il caso connettoma

Ordinati per costo GPU crescente. Riferimento di costo dichiarato dal committente: run da 2000 update ≈ 15–27 min su GTX 1080. I primi tre non richiedono training.

---

### E0 — Pannello diagnostico su checkpoint esistenti (0 min di training, ~10 min di calcolo)

**Cosa si cambia:** niente. Si analizzano i checkpoint già presenti (init, 2000, 8000).

**Cosa si misura:**
- distribuzione dei firing rate per neurone; frazione esattamente a zero (attesa: 81%);
- distribuzione di ξ = (θ − μ_U)/σ_U per neurone, con θ = 1: quanti neuroni cadono nella finestra 1 ≤ ξ ≤ 3 di Rossbroich;
- correlazione fra |Δw| dell'arco e firing rate della sorgente presinaptica;
- rango effettivo di ΔW e CE con ΔW troncato a rango 1, 2, 5, 10, 50;
- CKA fra attività del nucleo a init e a 8000 su un set di sonda fisso;
- ΔCE da reset per **ciascun** modulo (non solo il nucleo): embedding, iniettore, sinapsi, lettura, testa, attenzione.

**Quale risultato direbbe cosa:**
- se < 5% dei neuroni ha ξ ∈ [1,3] ⇒ l'inizializzazione è la causa prima del silenzio: E1 è obbligatorio e prioritario;
- se la correlazione |Δw| ↔ firing rate della sorgente è forte ⇒ la derivazione §D.4 è corretta e l'immobilità è un artefatto del passo effettivo, non della loss;
- se CKA(init, 8000) > 0,98 ⇒ regime lazy confermato a livello di rappresentazione, non solo di parametri;
- se ΔCE da reset della testa o dell'attenzione è grande e quello del nucleo è ~0 ⇒ quantificazione diretta di "chi fa il lavoro".

**Fonti:** Rossbroich et al. 2022 [C]; Schuessler et al. 2020 [C]; Kornblith et al. 2019 [P]; derivazione §D.4 [NV].

---

### E1 — Re-inizializzazione a fluttuazione del nucleo (1 run, 15–27 min)

**Cosa si cambia:** si riscalano i pesi entranti per neurone in modo che la deviazione standard del potenziale di membrana σ_U cada in [1/3, 1] e ξ = (1 − μ_U)/σ_U in [1, 3], invece di normalizzare la **somma** dei pesi entranti a 0,5. In pratica: normalizzazione sulla varianza (√Σw²) invece che sulla somma, con bilanciamento E/I esplicito per tenere μ_U sotto soglia. Segni di Dale invariati, topologia invariata.

**Cosa si misura:** frazione di neuroni silenti, frazione di archi con gradiente non nullo, CE a 2000 update, quota di norma di gradiente del nucleo.

**Quale risultato direbbe cosa:** se la frazione di silenti crolla dall'81% a < 30% e la quota di gradiente del nucleo sale di un ordine di grandezza **ma la CE resta invariata**, allora il problema non è il gradiente ma la piattezza della loss (§A/§C) e si passa a E2–E4. Se invece la CE migliora, l'inizializzazione era la causa dominante.

**Fonte:** Rossbroich, Gygax, Zenke 2022 [C] — su reti a 7 strati la differenza fra init a fluttuazione e Kaiming è 80,9% contro 4,5%.

**Nota:** questo è, a mio giudizio, l'esperimento con il miglior rapporto informazione/costo dell'intera lista.

---

### E2 — Scala dell'uscita ÷10 e ÷100 (2 run, 30–55 min)

**Cosa si cambia:** si moltiplica l'uscita della lettura (o equivalentemente si riduce l'inizializzazione della matrice 77-gruppi→256 e della testa) per 1/10 e per 1/100, compensando il learning rate per mantenere la stessa velocità di apprendimento in uscita. In alternativa μP-style: moltiplicatore 1/√d o 1/d sullo strato di uscita.

**Cosa si misura:** CE a 2000; ‖Δw‖ relativo del nucleo; ΔCE da reset del nucleo; CKA(init, 2000); rotazione dell'autostruttura dell'NTK ristretto al nucleo.

**Quale risultato direbbe cosa:** secondo Chizat il movimento dei parametri scala come 1/α, quindi ridurre α di 100× dovrebbe aumentare il movimento relativo di ~100×. Se il movimento aumenta **e** ΔCE da reset diventa significativo (> 0,05 nat), siete usciti dal regime lazy e la strada è quella. Se il movimento aumenta ma ΔCE da reset resta ~0, il nucleo si muove senza imparare: il problema è §C (starvation) e serve E5/E9.

**Attenzione:** Woodworth avverte che «α must become exponentially small» per il regime rich vero, mentre «polynomially large α suffices» per quello kernel [C]: ÷10 potrebbe non bastare; ÷100 è il minimo sindacale, valutare ÷1000.

**Fonti:** Chizat & Bach 2019 [C]; Woodworth et al. 2020 [C]; Kumar et al. 2024 [C]; Schuessler et al. 2024 [C] (nelle RNN la manopola è proprio la magnitudine dei pesi di uscita).

---

### E3 — Interfacce lente (lr ÷10, ÷100), nucleo veloce (2 run, 30–55 min) — *richiesto (ii)*

**Cosa si cambia:** lr di embedding, iniettore, lettura, testa e attenzione diviso per 10 e per 100 (Muon incluso: 1e-3 → 1e-4 → 1e-5); lr sinaptico invariato o alzato a 1e-3.

**Cosa si misura:** CE a 2000; quota di update relativo ‖Δθ_m‖/‖θ_m‖ per modulo; ΔCE da reset del nucleo; LCA per modulo.

**Quale risultato direbbe cosa:** se rallentare le interfacce fa salire la CE senza far salire ΔCE_reset(nucleo), il nucleo semplicemente non può compensare e la conclusione "reservoir" è robusta. Se invece la CE regge quasi altrettanto bene e ΔCE_reset(nucleo) cresce, c'è trasferimento di funzione verso il nucleo e il rapporto di lr è la manopola giusta.

**Fonti:** Lan et al. 2019 [C] (lr della testa ÷10 ≈ congelamento della testa, e migliora il training); Yang & Hu 2021 [C] (lr per strato è parte della definizione di regime); Peng et al. 2022 [P] (modulazione del gradiente per ramo); Kosson et al. 2024 [P].

---

### E4 — Congelare tutte le interfacce, allenare solo il nucleo da checkpoint allenato (1–2 run, 15–55 min) — *richiesto (i)*

**Cosa si cambia:** si parte dal checkpoint a 8000 update, si congelano embedding, iniettore, lettura, testa e attenzione, e si allenano **solo** i pesi sinaptici, con lr sinaptico 1e-4 (run A) e 1e-2 (run B, cioè ×100 per compensare il fattore softplus di §D.4).

**Cosa si misura:** curva di CE su 2000 update; ‖Δw‖ relativo; rango effettivo di ΔW; ΔCE da reset periodico; coseno fra gradienti di microbatch disgiunti (SNR) per il gruppo sinapsi.

**Quale risultato direbbe cosa:** questo è **il test decisivo sulla capacità residua del nucleo**. Tre esiti possibili:
- (a) la CE scende in modo apprezzabile ⇒ il nucleo *può* imparare, e nella configurazione normale è solo affamato dalle interfacce: la cura è E3/E5/E6;
- (b) la CE non scende e il coseno fra microbatch è ~0 ⇒ il gradiente è rumore: la cura è aumentare il batch efficace e/o le tracce di eleggibilità (E9, E11);
- (c) la CE non scende ma il coseno è alto (gradiente coerente) e i pesi si muovono ⇒ la loss è genuinamente piatta nelle direzioni del nucleo con interfacce fisse: serve un bersaglio interno (E12, full-FORCE) o un bottleneck (E6).

**Fonti:** full-FORCE, DePasquale et al. 2018 [P] (allenare la matrice ricorrente completa dà reti migliori, ma serve un target); Pezeshki et al. 2021 [C] (la starvation si rimuove disaccoppiando, non allenando più a lungo).

---

### E5 — Weight decay (3 run, 45–80 min) — *richiesto (iv)*

**Cosa si cambia:** tre configurazioni: (a) weight decay solo sulle interfacce (γ = 0,01 e 0,1), (b) weight decay su tutto, (c) spectral decoupling λ/2·‖logit‖² al posto del weight decay (λ = 0,01–0,1).

**Cosa si misura:** norma ℓ₂ per modulo nel tempo (cercando i punti di flesso che Nanda associa alle fasi); CE train e valid; ΔCE da reset del nucleo; excluded/restricted-loss analogo (CE con il contributo del nucleo azzerato vs CE con solo il nucleo).

**Quale risultato direbbe cosa:** il meccanismo LU di Omnigrok prevede che ridurre la norma delle interfacce sposti il modello verso la Goldilocks zone e forzi la funzione a migrare. Se dopo il decay delle interfacce il ΔCE da reset del nucleo **cresce**, avete innescato la migrazione e il grokking diventa plausibile: allora conviene un run lungo. Se la norma delle interfacce scende e la CE peggiora senza che il nucleo raccolga nulla, il grokking è escluso anche a lungo termine.

**Nota importante:** Pezeshki mostra che il weight decay ℓ₂ **non** cura la gradient starvation [C] mentre lo spectral decoupling sì; quindi la variante (c) va eseguita, non è un extra.

**Fonti:** Liu et al. 2023 Omnigrok [C]; Nanda et al. 2023 [C/P]; Kumar et al. 2024 [C]; Pezeshki et al. 2021 [C].

---

### E6 — Bottleneck sulle interfacce (3 run, 45–80 min) — *richiesto (v)*

**Cosa si cambia, tre varianti isolate:**
- (a) **meno porte:** 17.937 neuroni porta → 2.000 (mantenendo fan-in 8);
- (b) **lettura più stretta:** 77 gruppi → 16 gruppi, e/o 2.241 discendenti → 256;
- (c) **embedding più piccolo:** d = 256 → 64, con testa corrispondente.
- (d, opzionale e potente) **testa fissa:** matrice ortogonale casuale non addestrabile + sola temperatura appresa (Hoffer et al.).

**Cosa si misura:** CE; quota di gradiente e di update relativo del nucleo; ΔCE da reset del nucleo; frazione di neuroni raggiunti dall'iniezione che spara.

**Quale risultato direbbe cosa:** se restringere l'interfaccia **non** peggiora la CE, l'interfaccia era sovradimensionata e stava assorbendo capacità inutilmente; se la CE peggiora ma ΔCE_reset(nucleo) sale, la funzione si è spostata nel nucleo — è il risultato che si cerca. Se la CE peggiora e il nucleo non raccoglie niente, il nucleo non è in grado di sostituire l'interfaccia e la conclusione "reservoir" si rafforza.

**Fonti:** Hoffer et al. 2018 [P] (classificatore fisso a costo quasi nullo); Lappalainen et al. 2024 [C] (734 parametri liberi su 1,5 M connessioni: il modello vincolato dal connettoma di riferimento ha interfacce minuscole); Yang & Hu 2021 [C] (moltiplicatore 1/larghezza in uscita).

---

### E7 — Tracciamento continuo della quota di gradiente del nucleo (costo marginale ~0) — *richiesto (vi)*

**Cosa si cambia:** niente nell'addestramento; si aggiunge strumentazione da eseguire **dentro ogni run** della lista.

**Cosa si misura, ogni 50–100 update:**
- ‖g_m‖₂ per modulo (spazio raw **e** spazio pesi per il nucleo);
- ‖Δθ_m‖₂ e ‖Δθ_m‖₂/‖θ_m‖₂ per modulo;
- coseno(g₁, g₂) fra due microbatch disgiunti, per modulo;
- frazione di archi con |g| > 0;
- ogni 500 update: ΔCE da reset per modulo, ‖W_t − W_0‖_∞, CKA.

**Quale risultato direbbe cosa:** la firma da cercare è la **traiettoria** della quota: se la quota del nucleo parte bassa e **scende ulteriormente** nei primi 500 update, è gradient starvation in atto (le interfacce catturano la loss e spengono il resto), esattamente la dinamica descritta da Pezeshki. Se la quota è bassa e **piatta** dall'update 0, è un problema strutturale di init/sparsità (E1). Le due diagnosi portano a cure diverse, e solo questa misura le distingue.

**Fonti:** Pezeshki et al. 2021 [C]; Lan et al. 2019 [C]; Kosson et al. 2024 [P]; letteratura GSNR [P].

---

### E8 — Parametri appresi per neurone invece che per sinapsi (1–2 run, 15–55 min)

**Cosa si cambia:** si congelano i pesi sinaptici (che tanto non imparano) e si rendono addestrabili parametri **per neurone**: soglia θ_i, bias/corrente di riposo b_i, costante di leak λ_i (166.700 × 3 ≈ 500k parametri, molto meno dei 2,75 M archi ma con gradiente non nullo anche per i neuroni muti). Variante economica: parametri per **gruppo** (es. per tipo/neuropilo), alla flyvis, con 10²–10³ parametri.

**Cosa si misura:** CE; frazione di neuroni che passano da muti ad attivi; ΔCE da reset dei soli parametri neuronali; distribuzione finale di θ_i e b_i.

**Quale risultato direbbe cosa:** se questi ~500k (o ~10³) parametri producono un ΔCE da reset molto più grande dei 2,75 M pesi sinaptici, la conclusione operativa è netta: **nel vostro sistema il canale di apprendimento utile è il neurone, non la sinapsi**, e va ridisegnata la parametrizzazione. È esattamente ciò che fa il lavoro di riferimento sul connettoma della mosca.

**Fonte:** Lappalainen et al. 2024/flyvis [C]: segni e conteggi sinaptici **fissi**, si allenano 130 parametri di singolo neurone (costanti di tempo, potenziali di riposo) + 604 fattori di scala per coppia di tipi cellulari = 734 parametri liberi totali.

---

### E9 — Tracce di eleggibilità al posto dello spike binario (1–2 run, 20–60 min)

**Cosa si cambia:** nel calcolo del gradiente del peso si sostituisce spike(sorgente) ∈ {0,1} con la traccia filtrata ẑ_i (passa-basso con costante di tempo di 5, 15 e 30 sottopassi), alla e-prop: e_ji = h_j · ẑ_i. Costo: una variabile di stato in più per neurone presinaptico, non per arco. Da provare sia da solo sia in combinazione con E1.

**Cosa si misura:** frazione di archi con gradiente non nullo; coseno fra microbatch (SNR); CE; ΔCE da reset.

**Quale risultato direbbe cosa:** se la frazione di archi con gradiente sale al 60–90% e il coseno fra microbatch sale, il canale di credito è stato riaperto. Se la CE comunque non cambia (come è successo con l'ammorbidimento spaziale della vostra misura 4), la diagnosi "loss piatta nelle direzioni sinaptiche" diventa molto solida e bisogna spostarsi su E8, E11, E12.

**Fonte:** Bellec et al. 2020 [C]: e_ji^t = prodotto della pseudo-derivata postsinaptica h_j^t per la traccia ẑ_i^t degli spike presinaptici; «a synapse remembers some of its activation history». Nota di cautela da Zenke & Vogels [C]: in reti ricorrenti allargare troppo il surrogato può essere **dannoso** — la traccia temporale è meno rischiosa dell'allargamento della derivata.

---

### E10 — Regolarizzazione omeostatica del firing rate (1 run, 15–27 min)

**Cosa si cambia:** si aggiunge alla loss un termine che penalizza la deviazione del rate di ciascun neurone da un target (es. 2–5% di probabilità di spike per sottopasso), con peso piccolo (0,001–0,01). Il gradiente di questo termine **non dipende dal task** e raggiunge anche i neuroni muti tramite la derivata surrogata.

**Cosa si misura:** frazione di neuroni muti nel tempo; CE del task (deve non peggiorare); quota di gradiente del nucleo; frazione di archi mossi.

**Quale risultato direbbe cosa:** se il regolarizzatore accende i neuroni ma la CE non migliora, si è dimostrato che il silenzio dell'81% **non era la causa** della piattezza della loss — informazione preziosa che restringe il campo a §A/§C. Se accende i neuroni e la CE migliora, è una vittoria a costo bassissimo.

**Fonti:** Zenke & Vogels 2021 [C] (due forme di regolarizzazione dell'attività; le SNN regolarizzate reggono fino a una soglia critica di attività); letteratura sulla regolarizzazione omeostatica nelle SNN [P].

---

### E11 — Rimozione/limitazione della scorciatoia + spectral decoupling (2–3 run, 30–80 min)

**Cosa si cambia:**
- (a) attenzione esterna disattivata del tutto (il nucleo è l'unica memoria);
- (b) attenzione con cache ridotta da 128 a 16 e 1 sola testa;
- (c) attenzione attiva ma con dropout alto (0,5) sul percorso attenzione → la corrente rientrante è affidabile solo in media, il che penalizza la scorciatoia;
- in tutte: spectral decoupling λ/2·‖logit‖².

**Cosa si misura:** CE; quota di gradiente del nucleo; ΔCE da reset del nucleo; confronto con il baseline GRU (3,43) e col vostro migliore (3,54).

**Quale risultato direbbe cosa:** se togliendo l'attenzione la CE degrada catastroficamente, l'attenzione stava facendo quasi tutto il lavoro e il "modello connettomico" è in realtà un modello ad attenzione con un reservoir attaccato. Se la CE degrada moderatamente e il nucleo comincia a raccogliere ΔCE da reset, il percorso facile era il colpevole e la cura è il bottleneck sull'attenzione.

**Fonti:** Pezeshki et al. 2021 [C] (spectral decoupling; il weight decay non basta); Wang et al. 2020 [P] e Peng et al. 2022 [P] (nei sistemi multi-percorso il ramo dominante sopprime l'altro, e la cura è modulare i rami).

---

### E12 — μP + TBPTT lungo + batch grande (2 run, ~1–2 h ciascuno)

**Cosa si cambia:** riparametrizzazione μP dell'intero modello (moltiplicatore 1/larghezza in uscita, init n^{−1/2}, lr O(1) sugli strati nascosti), TBPTT da 8 a 32 token, batch da 2 a 16 con accumulo di gradiente. Obiettivo: alzare il rapporto segnale/rumore del gradiente sinaptico, che con 16 bersagli per update è verosimilmente sotto la soglia di utilità.

**Cosa si misura:** coseno fra microbatch per il gruppo sinapsi (deve salire), CE, ΔCE da reset, rango effettivo di ΔW.

**Quale risultato direbbe cosa:** se il coseno fra microbatch passa da ~0 a > 0,2 e solo allora i pesi cominciano a spostarsi in modo coerente (rango effettivo basso invece che diffuso), il problema era il rumore e non la piattezza. È l'ipotesi alternativa più credibile a quella lazy e va falsificata esplicitamente.

**Fonti:** Yang & Hu 2021 [C]; letteratura GSNR [P]; Schuessler et al. 2020 [C] per la lettura del rango effettivo.

---

### E13 — Bersagli interni stile full-FORCE (2+ h, il più costoso)

**Cosa si cambia:** si costruisce una rete "maestro" (il GRU 2×256 già addestrato, oppure il modello con interfacce addestrate) e si definisce un bersaglio per le **correnti interne** del nucleo: si addestrano i pesi sinaptici a riprodurre quelle dinamiche target, oltre che a minimizzare la CE. È la trasposizione diretta del metodo full-FORCE, che usa una seconda rete durante il training per fornire dinamiche target.

**Cosa si misura:** errore sulle correnti interne; CE; ΔCE da reset del nucleo; rango effettivo di ΔW.

**Quale risultato direbbe cosa:** se con un bersaglio interno i pesi sinaptici imparano (errore interno che scende, ΔCE da reset che sale) mentre con la sola CE non imparano, la diagnosi finale è: **il nucleo ha capacità, ma la sola loss di uscita non produce credito assegnabile alle sinapsi attraverso 8 sottopassi × 8 token di TBPTT**. È il risultato più informativo dell'intera lista, ed è anche il più costoso.

**Fonte:** DePasquale, Cueva, Rajan, Escola, Abbott 2018, full-FORCE [P]: allenando l'intera connettività ricorrente con dinamiche target si ottengono reti che svolgono il compito «with fewer neurons and greater noise robustness» rispetto al solo readout.

---

### Ordine consigliato di esecuzione

1. **E0** (gratis) → decide se E1 è obbligatorio.
2. **E7** (gratis, da attivare in tutti i run successivi).
3. **E1** (1 run) — massima leva attesa.
4. **E4** (1–2 run) — test decisivo sulla capacità residua del nucleo, e risponde direttamente alla domanda (i) del committente.
5. **E3** e **E2** (4 run) — le due manopole di regime.
6. **E10**, **E9** (2–3 run) — riapertura del canale di credito.
7. **E6**, **E5** (6 run) — bottleneck e weight decay.
8. **E8** — riparametrizzazione neurone-centrica (il precedente flyvis suggerisce che sia la strada giusta a lungo termine).
9. **E11**, **E12**, **E13** — solo se i precedenti non hanno chiuso la domanda.

Costo totale dei punti 1–5: circa 2–3 ore di GTX 1080, che è il budget entro cui la domanda "il nucleo può imparare qualcosa?" è rispondibile in modo conclusivo.

---

## Fonti

### A — Regime lazy / rich
1. Chizat, Oyallon, Bach — *On Lazy Training in Differentiable Programming*, arXiv:1812.07956, NeurIPS 2019. https://arxiv.org/abs/1812.07956 — abstract **[C]**; formule κ_h, O(1/α), O(1/α²) e risultato VGG-11 da ar5iv https://ar5iv.labs.arxiv.org/html/1812.07956 **[C]**; valori numerici 61,7% / 89,7% **[P]**.
2. Woodworth, Gunasekar, Lee, Moroshko, Savarese, Golan, Soudry, Srebro — *Kernel and Rich Regimes in Overparametrized Models*, arXiv:2002.09277 (v. anche 1906.05827), COLT 2020. https://arxiv.org/abs/2002.09277 — α→∞ ℓ₂ / α→0 ℓ₁ e "exponentially small vs polynomially large" da ar5iv **[C]**.
3. Yang, Hu — *Feature Learning in Infinite-Width Neural Networks*, arXiv:2011.14522, ICML 2021. https://arxiv.org/abs/2011.14522 — abstract **[C]**; abc-parametrizzazione, r = 0 / r > 0, tabella μP da ar5iv **[C]**.
4. Schuessler, Mastrogiuseppe, Ostojic, Barak — *Aligned and oblique dynamics in recurrent neural networks*, eLife 2024, 10.7554/eLife.93060. https://elifesciences.org/articles/93060 — manopola = magnitudine dei pesi di uscita; instabilità delle soluzioni lazy nelle RNN **[C]**.
5. Flesch, Saxe, Summerfield — *Rich and lazy learning of task representations in brains and neural networks*, bioRxiv 2021.04.23.441128. https://www.biorxiv.org/content/10.1101/2021.04.23.441128v1 — **[P]**.

### B — RNN, reservoir, connettoma
6. Schuessler, Mastrogiuseppe, Dubreuil, Ostojic, Barak — *The interplay between randomness and structure during learning in RNNs*, arXiv:2006.11036, NeurIPS 2020. https://arxiv.org/abs/2006.11036 — ranghi funzionali 2/4/6, ‖ΔW‖ decresce con g, ‖W₀‖=√N g, troncamento preserva la prestazione **[C]** (via ar5iv).
7. Vlachas et al. — *Backpropagation algorithms and Reservoir Computing in RNNs for the forecasting of complex spatiotemporal dynamics*, Neural Networks 126 (2020), arXiv:1910.05266. https://arxiv.org/abs/1910.05266 — **[P]**.
8. DePasquale, Cueva, Rajan, Escola, Abbott — *full-FORCE: A target-based method for training recurrent networks*, PLoS ONE 13(2):e0191527, arXiv:1710.03070. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0191527 — **[P]**.
9. Lappalainen et al. — *Connectome-constrained networks predict neural activity across the fly visual system*, Nature 632 (2024). https://www.nature.com/articles/s41586-024-07939-3 — versione Nature **[P]**; preprint bioRxiv 2023.03.11.532232 (segni e conteggi fissi, 130 + 604 = 734 parametri liberi, 45.669 neuroni / 1.513.231 connessioni, BPTT su flusso ottico) https://www.biorxiv.org/content/10.1101/2023.03.11.532232v1.full — **[C]**. Codice: https://github.com/TuragaLab/flyvis
10. Beiran, Litwin-Kumar — *Prediction of neural activity in connectome-constrained recurrent networks*, Nature Neuroscience 28:2561–2574 (2025). https://www.nature.com/articles/s41593-025-02080-4 — «a connectome is often insufficient to constrain the dynamics» **[P]**.

### C — Gradient starvation, squilibrio fra moduli
11. Pezeshki, Kaba, Bengio, Courville, Precup, Lajoie — *Gradient Starvation: A Learning Proclivity in Neural Networks*, arXiv:2011.09468, NeurIPS 2021. https://arxiv.org/abs/2011.09468 — definizione, U·S²·Uᵀ, spectral decoupling λ/2‖ŷ‖², "weight decay non basta" **[C]** (via ar5iv).
12. Lan, Liu, Zhou, Yosinski — *LCA: Loss Change Allocation for Neural Network Training*, arXiv:1909.01440, NeurIPS 2019. https://arxiv.org/abs/1909.01440 — primo/ultimo strato danneggiano; congelare l'ultimo strato migliora (p<0,001); lr ÷10 equivalente **[C]** (via ar5iv).
13. Hoffer, Hubara, Soudry — *Fix your classifier: the marginal value of training the last weight layer*, arXiv:1801.04540, ICLR 2018. https://arxiv.org/abs/1801.04540 — **[P]**.
14. Wang, Tran, Feiszli — *What Makes Training Multi-modal Classification Networks Hard?*, CVPR 2020. https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.pdf — **[P]**.
15. Peng, Wei, Deng, Wang, Hu — *Balanced Multimodal Learning via On-the-fly Gradient Modulation*, CVPR 2022. https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.pdf — **[P]**.
16. Kosson, Messmer, Jaggi — *Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks*, ICML 2024, arXiv:2305.17212. https://proceedings.mlr.press/v235/kosson24a.html — **[P]**.
17. Jordan et al. — *Muon: An optimizer for hidden layers in neural networks* (2024). https://kellerjordan.github.io/posts/muon/ — discesa più ripida in norma spettrale **[P]** (blog dell'autore, non peer-reviewed).

### D — Reti spiking
18. Rossbroich, Gygax, Zenke — *Fluctuation-driven initialization for spiking neural network training*, Neuromorphic Comput. Eng. 2(4) 044016 (2022), arXiv:2206.10226. https://arxiv.org/abs/2206.10226 — ξ=(θ−μ_U)/σ_U ∈ [1,3], σ_U ∈ [1/3,1], SHD 7 strati 80,9% vs 4,5%, gradiente nascosto alterato di due ordini di grandezza **[C]** (via ar5iv). Codice: https://github.com/fmi-basel/stork
19. Zenke, Vogels — *The Remarkable Robustness of Surrogate Gradient Learning…*, Neural Computation 33(4):899–925 (2021); preprint bioRxiv 2020.06.29.176925. https://direct.mit.edu/neco/article/33/4/899/97482/ — robustezza alla forma, sensibilità alla scala (dannosa con ricorrenza), regolarizzazione dell'attività **[C]** (dal preprint).
20. Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein, Maass — *A solution to the learning dilemma for recurrent networks of spiking neurons*, Nature Communications 11:3625 (2020); preprint arXiv:1901.09049. https://www.nature.com/articles/s41467-020-17236-y — fattorizzazione dE/dθ = Σ_t L_j^t e_ji^t, traccia = pseudo-derivata × traccia filtrata degli spike presinaptici, confronto con BPTT **[C]** (via ar5iv del preprint; la pagina Nature richiede redirect di autenticazione).
21. Eshraghian et al. — *Training Spiking Neural Networks Using Lessons From Deep Learning*, Proceedings of the IEEE 111(9):1016–1054 (2023). https://proceedingsoftheieee.ieee.org/training-spiking-neural-networks-using-lessons-from-deep-learning/ — dead neuron problem **[P]**.
22. Perez-Nieves, Goodman — *Sparse Spiking Gradient Descent*, NeurIPS 2021, arXiv:2105.08810. https://arxiv.org/abs/2105.08810 — attività media mai sopra il 2% **[P]**.
23. Cornford, Kalajdzievski, Leite, Lamarquette, Kullmann, Richards — *Learning to live with Dale's principle: ANNs with separate excitatory and inhibitory units*, ICLR 2021; bioRxiv 2020.11.02.364968. https://www.biorxiv.org/content/10.1101/2020.11.02.364968v2.full — rettifica max(0,θ), fallimento di ColumnEi, correzione tramite Fisher **[C]**; dettaglio del fattore 1/n_e **[P]**.

### E — Grokking ed emergenza
24. Nanda, Chan, Lieberum, Smith, Steinhardt — *Progress measures for grokking via mechanistic interpretability*, arXiv:2301.05217, ICLR 2023. https://arxiv.org/abs/2301.05217 — restricted/excluded loss, tre fasi, flessi della norma ℓ₂ **[C]** (via ar5iv); "λ=0 ⇒ nessun grokking" **[P]**.
25. Liu, Michaud, Tegmark — *Omnigrok: Grokking Beyond Algorithmic Data*, arXiv:2210.01117, ICLR 2023. https://arxiv.org/abs/2210.01117 — meccanismo LU, grokking indotto su MNIST con init grande, de-grokking con dataset più grandi **[C]** (via ar5iv); tempo ∝ γ⁻¹ **[P]**.
26. Kumar, Bordelon, Gershman, Pehlevan — *Grokking as the Transition from Lazy to Rich Training Dynamics*, arXiv:2310.06110, ICLR 2024. https://arxiv.org/abs/2310.06110 — α controlla il ritardo, weight decay esce dal regime lazy, grokking annullabile (α=0,15) **[C]** (via ar5iv).
27. Barak, Edelman, Goel, Kakade, Malach, Zhang — *Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit*, arXiv:2207.08799, NeurIPS 2022. https://arxiv.org/abs/2207.08799 — Fourier gap, progresso invisibile alle metriche, ρ = ‖w_t − w_0‖_∞ **[C]** (via ar5iv).
28. Schaeffer, Miranda, Koyejo — *Are Emergent Abilities of Large Language Models a Mirage?*, arXiv:2304.15004, NeurIPS 2023. https://arxiv.org/abs/2304.15004 — **[P]**.
29. Zhu, Fu, Zhou, Lin — *Critical Data Size of Language Models from a Grokking Perspective*, arXiv:2401.10463. https://arxiv.org/abs/2401.10463 — transizioni più lisce sui dati di linguaggio **[C]** (abstract).
30. Power, Burda, Edwards, Babuschkin, Misra — *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*, arXiv:2201.02177 — citato come riferimento storico, **[NV]** in questa sessione.

### F — Diagnostiche
31. Kornblith, Norouzi, Lee, Hinton — *Similarity of Neural Network Representations Revisited*, ICML 2019. http://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf — CKA, rileva patologie non visibili nell'accuratezza **[P]**.
32. Atanasov, Bordelon, Pehlevan — *Neural Networks as Kernel Learners: The Silent Alignment Effect*, arXiv:2111.00034, ICLR 2022. https://arxiv.org/abs/2111.00034 — **[P]**.
33. Baratin, George, Laurent, Hjelm, Lajoie, Vincent, Lacoste-Julien — *Implicit Regularization via Neural Feature Alignment*, AISTATS 2021. https://proceedings.mlr.press/v130/baratin21a.html — codice https://github.com/tfjgeorge/ntk_alignment — **[P]**.
34. Letteratura GSNR (gradient signal-to-noise ratio) come diagnostica per parametro/strato — **[P]**, nessuna fonte singola canonica verificata in questa sessione.

### Derivazioni proprie (non bibliografiche)
35. Fattore softplus sul passo effettivo: dw/d(raw) = sigmoid(softplus⁻¹(0,03)) = 0,0296 (riduzione ≈ 33,8×); predizione a random walk della variazione relativa di norma su 8000 update con frazione attiva 19% = 0,385% contro 0,30% misurato; soglia di 1.141 step con gradiente non nullo per superare 1e-4 di spostamento assoluto — **[NV]**, §D.4.
36. Valore della derivata surrogata sigmoide 4(u−1) a riposo (u=0): 4·σ(−4)·(1−σ(−4)) ≈ 0,0719 — **[NV]**, §D.2.
37. Asimmetria di velocità Muon/Adam + softplus ≈ due ordini di grandezza — **[NV]**, §C.4.
