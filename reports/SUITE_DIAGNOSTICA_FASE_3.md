# Fase3 — suite diagnostica proposta

Stato: **T0 passato; T1b d1 calibrato; T2–T5 eseguiti**. GRU d1 passa 30,80%/200update→100%/1000update; d4/d8 controllo non calibrati. Mosca d0 head 18,85% DEV, probe finale 84,77%. T4: K/V informativi, uso locale debole su testo. T5 update uguali: d1 L8/L16 DEV 27,82%/26,26%; d8 12,57%/12,63%, nessun recupero. [Resoconto aggiornato](RESOCONTO_T1B_T4_T5.md). T6/T7 non eseguiti; fase3 aperta. Criteri e architettura invariati. Nuova [suite soluzioni a grafo intero](SUITE_SOLUZIONI_FASE_3.md).

## Gli8 dubbi, ordinati per dipendenza

| Ordine | Dubbio | Numero nella ricerca | Perché qui |
|---|---|---:|---|
| 1 | Cosa deve dimostrare fase3? | 8 | Definire decisioni prima di raccogliere numeri |
| 2 | Dati/budget e controlli adeguati? | 1 | Separare task mal calibrato da modello insufficiente |
| 3 | Token distinguibile al readout? | 2 | Senza info in uscita, memoria/LM non diagnosticabili |
| 4 | Dinamica LIF e gradienti usabili? | 6 | Localizzare attenuazione, saturazione, feedback dominante |
| 5 | Attenzione conserva/usa info? | 3 | Studiare memoria dopo aver verificato cosa entra |
| 6 | TBPTT e frequenza update separati? | 4 | Attribuire differenze al gradiente, non a più update |
| 7 | LM impara distribuzioni condizionate e genera? | 5 | Passare da diagnosi a pilot su testo nuovo |
| 8 | Soglia5 migliora qualità/costo? | 7 | Confrontare archi su pipeline diagnosticata |

## Regole comuni

- Riferimento: soglia10, T8=4pre+4post, BPE4096/d256, Adam0,0001, B2; L8 salvo confronto T5. GPU FP32/CUDA Graph. CPU solo dati/I/O/statistiche/supervisione.
- Una prova GPU per volta; guardie paging autorizzate, NaN/Inf, progressi ogni32 update o più spesso, cleanup misurato. Nessun nuovo grid optimizer.
- Riutilizzare regressioni2.1–2.5; ripetere solo dove nuova strumentazione può alterare risultati. Registrare hash pesi/config/dati, target validi, update, token nuovi/ripetuti, tempo, picco allocator.
- Le16 storie validation precedenti sono DEV già consultato, incluse ultime12: non chiamarle audit nuovo. Riservare 64 altre storie validation, IDs fissati senza leggere risultati. Aprire audit una volta dopo selezione; test finale separato resta chiuso.
- Baseline/probe sono strumenti diagnostici separati, non bypass nel modello. Modifiche a grafo, porte, gain, reset, leak, attenzione: proposta prima dell'applicazione. Soglia5 e TBPTT16 autorizzati come esperimenti, non adottati.

## Blocco A — protocollo affidabile

### T0 — Contratti del task e criteri

**Domanda:** stiamo misurando ciò che pensiamo?

- Oracle a buffer per copia d0/1/4/8: confrontare target, maschere, reset, confini finestre. Atteso100% su target validi; nessuna previsione PAD/BOS valutata.
- Separare correttezza, capacità appresa, qualità linguistica. 80% ritardo resta traguardo diagnostico storico, non prova universale di validità LM.
- Blocchi obbligatori: target errati, fuga dal futuro, reset errato, NaN/Inf, cleanup fallito. Altri test: esito → decisione, non inseguimento illimitato di percentuale.
- Budget: lettura/check deterministici; eventuale smoke GPU entro60s. Nessun training grande.

### T1 — Controlli e calibrazione del budget

- Stessi generatori/task: oracle + predittore costante/chance12,5% + piccola GRU controllo. Unigramma e bigramma con smoothing/backoff, stimati SOLO da train, per testo.
- Prima calibrare controllo su d0/d1; d4/d8 solo quando funziona. Stream nuovi generati deterministicamente per training; DEV fisso separato. Per confronti riprodurre identico elenco stream.
- Train e DEV sempre riportati entrambi. Se anche controllo non impara, rivedere task/budget prima di attribuire fallimento alla mosca. Se controllo riesce, avere costo di riferimento.
- Budget iniziale: massimo200 update per controllo, niente griglia LR. Budget insufficiente dichiarato, non fallo della mosca.

## Blocco B — informazione nel grafo

### T2 — Identità del token e localizzazione della perdita

- Sottoinsieme8 simboli bilanciati, identità d0; training/DEV con contesti distinti. Sia reset fra esempi sia contesto variabile, per distinguere risposta al token e storia dello stato.
- Registrare: embedding normalizzato → correnti sensoriali → letture pre-pooling →154 feature pooled → rappresentazione256 pre/post →logits.
- Misurare separazione fra token rispetto a variabilità stesso token fra contesti; probe lineari congelando modello, fit su train e punteggio DEV. Controlli: etichette permutate, rappresentazione costante, embedding come controllo positivo.
- Probe negativo significa info non linearmente accessibile al probe, non assenza matematica. In parallelo, breve apprendimento d0 con head LM originale, accuracy8-classi e4096-classi entrambe.
- Budget: raccolta massima128 stream×32 posizioni; apprendimento d0 massimo200 update; niente ripetizioni per distanza/seme finché manca diagnosi.
- Decisione: segnale presente prima e debole dopo interfaccia → indagare quella; segnale leggibile ma head LM non impara → indagare ottimizzazione/head. Non aumentare archi automaticamente.

### T3 — Dinamica/gradienti per regione

- Usare stesse sequenze e checkpoint di T2: evitare altro training.
- Per ingressi, intermedi, uscite: potenziale rispetto a soglia, firing valido senza PAD, distribuzione gradienti, rapporto aggiornamento/peso, RMS token/feedback. Valutare stato anche durante generazione.
- Gradienti rispetto a correnti dei singoli token, non solo alla matrice embedding condivisa: quest'ultima riceve anche gradiente diretto dal readout.
- Decisione: identificare dove segnale/gradiente diminuiscono o esplodono. Firing basso da solo non equivale a neuroni inutili; finitezza da sola non equivale a buona calibrazione.

### T4 — Contenuto e uso della memoria

- Prima verificare diversità e decodificabilità di K/V sugli stessi contesti T2. Poi confrontare inferenza con cache reale, temporalmente permutata e vuota, preservando validità/causalità.
- Stessi pesi/prompt, nessun update. Misurare variazione logits/CE, non solo mappe attenzione. Perturbazioni sono interventi diagnostici temporanei, non architettura candidata.
- Nessun effetto → modello poco dipendente dalla cache in quel test. Effetto grande → dipendenza, non necessariamente utilità. Perturbazioni possono creare stati fuori distribuzione.
- Per dichiarare vantaggio attenzione serve poi controllo riaddestrato con pari budget; non necessario subito per localizzare il problema.

### T5 — Memoria e TBPTT senza confondere gli update

- In entrambi i casi update ogni16 token; stessi16 token, pesi iniziali, reset, LR e numero update.
- L8: due segmenti8, detach fra segmenti, accumulo gradienti; nessun optimizer step intermedio. L16: singolo segmento16 senza detach interno. Loss pesata sul totale target validi, non media ingenua delle medie dei segmenti.
- Partire da d1 e d8,200 update/config massimo; d4 solo se aiuta a localizzare la transizione. Stessi stream nuovi dell'elenco fissato in T1. Misurare train/DEV e gradiente verso corrente del token sorgente.
- Diagnostica breve prima; secondo seme solo per confermare differenza utile. 80% riportato, non abbassato a posteriori.
- Decisione: miglioramento a L16 con update uguali → evidenza utile per scegliere troncamento; nessuna differenza → non raddoppiare L senza nuova ipotesi.

## Blocco C — pilot e decisione sulla fase3

### T6 — Testo nuovo, stabilità, generazione

- Solo dopo T0 e diagnosi T2/T3: pilot iniziale100.000 target validi da storie train nuove rispetto0..81, un passaggio. Valutazioni DEV prefissate a10k/30k/100k; stessi prompt, nessuna selezione estetica.
- Confrontare unigramma/bigramma e, se serve, GRU sullo stesso tokenizer e dati. CE per storia, accuracy, entropia, frequenza token dominante, sensibilità al prompt. Baseline contestuale distingue frequenze apprese da uso del contesto.
- Greedy e sampling con temperatura/top-p/semi fissati prima: riportare entrambi. Sampling meno ripetitivo non basta a dichiarare linguaggio coerente.
- Affiancare stabilità su sequenze vere e generazione: quantili/max dei potenziali, gradienti, correnti nel tempo. Non usare solo tre punti monotoni finali come diagnostica di deriva.
-100k target: stima formazione ~35–40min a soglia10 dai run brevi; misurare prima e concordare tetto complessivo. Non è benchmark di throughput garantito. Nessun avvio implicito del pilot.
- Se primo pilot promettente, secondo seme a pari budget; poi unico audit sulle64 storie riservate. Report differenze per storia e variabilità, niente vittoria da ultima loss train.
- Proposta chiusura3: correttezza/stabilità valide nel pilot, apprendimento replicato su dati nuovi con vantaggio sul baseline senza contesto e uso del contesto documentato; generazioni ispezionate con limiti espliciti. Nessuna richiesta di storie perfette prima del pretraining. Se manca, indicare quale capacità resta non dimostrata.

### T7 — Soglia10/5 dopo la diagnosi

- Se T2 indica strozzatura strutturale, anticipare sola misura percorsi/separabilità10/5. Altrimenti rinviare confronto completo alla fase4.
- Stessi target nuovi, update, stato iniziale artificiale e LR; registrare costo e pesi rinormalizzati. Non confondere aggiunta archi con semplice aggiunta di capacità a pesi invariati.
- Confronto memoria già eseguito riutilizzabile come evidenza negativa locale. Confronto CE5 interrotto resta incompleto; non ripartire per completare una casella prima di avere test informativo.

## Cosa abbiamo già provato

| Test | Esito osservato | Cosa NON dimostra |
|---|---|---|
| API/reset/padding/causalità/cache128/gradienti/round-trip checkpoint | Pass nelle fasi2.1–2.5 | Capacità linguistica/memoria appresa |
| Overfit32 target | CE8,24→2,02;19/32 corretti | Generalizzazione |
| Stress inferenza256 token | Finito, cache128 corretta | Stabilità durante pretraining |
| Copia1/4/8, TBPTT8,30 epoche | Dev72,58%/14,73%/9,375%; sotto80% | Impossibilità del task con altro training |
| TBPTT8/16, dati uguali | Nessun recupero; update diversi | Confronto isolato della lunghezza gradiente |
|24 run Adam/AdamW/Muon/ROOT | Vantaggio Muon/ROOT sotto soglia2% fissata | Migliore optimizer dopo vero pretraining |
| LM64 storie, Adam0,0001/512 update | CE8,376→5,918; greedy punti | Linguaggio utile |
| LM64 storie, Adam0,0003/1024 update | CE6,149 vs unigramma6,012; greedy virgole | Che basti modificare decoding |
| Soglia10/5 memoria,L16 | Dev24,19/10,71/9,375% vs18,55/11,16/9,375%; costo~2× | Superiorità generale di una soglia |
| Soglia10/5 linguaggio,512 update |10:CE6,035;5:interrotta durante validation | Confronto CE concluso |
| Oracle task d0/1/4/8 |100%, reset/PAD/chunk/futuro passati | Apprendimento del modello |
| GRU controllo200 update d0/d1 | DEV100%/30,80%; d4/d8 non avviati | Budget memoria calibrato |
| Unigramma/bigramma solo train | DEV CE6,012/4,803 | Qualità della mosca |
| Probe token→readout + T3 dinamica/gradienti | Identità leggibile post92,29% grezzo/84,77% finale; head18,85%; gradienti presenti | Qualità linguistica, memoria lunga, causa unica del limite |
| GRU T1b d1,1000update | Train/DEV100%,CE0,00242 | Controllo d4/d8 o capacità mosca |
| T4 cache reale/vuota/invertita + controllo invariante | K/V decodificabili; ΔCE testo molto piccolo | Attenzione inutile in generale |
| T5 L8/L16,200update uguali | Nessun recupero d1/d8; gradiente sorgente d8 presente soloL16 nel test | Impossibilità con altro training |
| Pilot100k | NON eseguito | Nessun esito anticipato |

## Mappatura sul punto3

| Parte | Suite proposta | Stato |
|---|---|---|
|3.1 Protocollo/telemetria | T0–T1b; protocollo versionato | T0 passato; d1 controllo calibrato |
|3.2 Apprendimento elementare/interfacce | T2–T3 + overfit storico | Overfit passato; diagnosi eseguita, lettura da calibrare |
|3.3 Memoria/attenzione/TBPTT | T4–T5 | Isolamento eseguito; capacità non recuperata |
|3.4 Ottimizzatore/configurazione | Riutilizzare griglia; decidere dopo T3/T5 | Nessun nuovo grid ora |
|3.5 Pilot/validation/stabilità | T6 + audit finale | Pilot proposto; qualità ancora non dimostrata |
|Ponte verso4 | T7 | Archi5 sperimentali, non adottati |

Ordine operativo minimo: T0/T1 → T2/T3. Poi decidere: riparazione mirata se manca segnale; T4/T5 se problema memoria; T6 se percorso funziona. Non eseguire tutto come matrice combinatoria.

Fonti e motivazioni: [ricerca8 dubbi](DUBBI_FASE_3_RICERCA.md). Evidenze: [fase3](RESOCONTO_FASE_3.md), [recupero](RESOCONTO_RECUPERO_FASE_3.md), [confronti](results/phase3d_summary.json).
