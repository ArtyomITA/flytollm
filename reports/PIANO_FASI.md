# Mosca LM — piano operativo in 6 fasi

Specifica: [ARCHITETTURA_MOSCA_ATTENZIONE.md](ARCHITETTURA_MOSCA_ATTENZIONE.md), v0.6, profilo H1 separato.
Revisione: [REVISIONE_OTTIMIZZAZIONI.md](REVISIONE_OTTIMIZZAZIONI.md).
Stato13settembre: **3.5 verifiche concluse; fase4 confronto operativo10/5 concluso; H1-10 scelta. Fase3.3 e qualità linguistica restano aperte. Pretraining fermato su richiesta a26.247update; checkpoint finale salvato. Bilancio completo: [stop e MoE](RESOCONTO_STOP_NOTTE_MOE.md).** [Risultati aggiornati](RESOCONTO_3_5_4_NOTTE.md), [piano notte](PRETRAINING_NOTTE.md). Le sezioni storiche sotto conservano evidenze e candidati futuri, non costituiscono dichiarazione di qualità superata.
Vincolo esecuzione aggiornato: **calcolo modello solo GPU FP32 con CUDA Graph; nessun offload CPU**. CPU per dati, inizializzazione e supervisione. Eager GPU solo riferimento numerico.

Modifiche architetturali: domanda all’utente prima di applicare. Valutazioni nel piano ≠ autorizzazione varianti.

Priorità confermata: **grafo intero, migliore assegnazione porte/letture/percorsi; nessun pruning predefinito**. Calibrare lettura non significa rimuovere nodi. Soglia sinaptica10/5 distinta da firing LIF1; MHA4/KV128 già presenti.

Vincoli permanenti: 166.700 nodi, identità/direzioni conservate, nessun LFM, moduli artificiali contabilizzati, guardie RAM/VRAM esistenti.

## 1 — Dati, tokenizer, criteri di successo — COMPLETATA

- Verificare TinyStories V2: split, confini storie, duplicazioni rilevanti, hash.
- BPE byte-level solo train; confronto offline 2k/4k/8k: byte/token, lunghezze, costo. Baseline4k, d256.
- Fissare EOS/BOS/PAD, round-trip, tokenizer immutabile/run; pretokenizzare su disco, corpus fuori RAM.
- Prima dei risultati: fissare benchmark, piccoli modelli confronto, split valutazione, prompt/decoding, budget.
- Metriche: CE stesso tokenizer, bit/byte con convenzioni comuni, qualità completamenti, testo/s e risorse. Confronti a tempo uguale e dati uguali distinti.

**Output:** tokenizer + manifest dati + protocollo valutazione.
**Passaggio:** tokenizzazione reversibile, nessun test usato per apprendimento/selezione, budget e confronti espliciti.

Evidenza: `dataset/prepared_v1/manifest.json`, `verification.json`, `tokenizer_comparison.json`; [PROTOCOLLO_VALUTAZIONE.md](PROTOCOLLO_VALUTAZIONE.md). Train intero pretokenizzato; BPE4k confermato.

## 2 — Modello completo: testo → grafo → attenzione → testo

### 2.1 — API del nucleo e stato — COMPLETATA

- Separare sottopassi dalla loss sintetica: stato iniziale, avanzamento, rate della fase, reset slot, detach esplicito.
- Comporre fase pre/post con correnti diverse; nessuna mutazione dello stato del chiamante.
- Preservare vecchio benchmark; confrontare equazioni/gradienti, composizione fasi, reset, slot inattivi e checkpoint.
- Smoke CPU + CUDA sotto guardie, incluso CNS soglia10 a batch piccolo.

**Output:** nucleo riutilizzabile da LM; API documentata; test/regressioni passati.

Evidenza: [CORE_API.md](CORE_API.md), `results/phase21_cpu_tests.json`, `results/phase21_cns_t10.json`, `results/phase21_toy_graph.json`.

### 2.2 — Embedding e interfacce anatomiche — COMPLETATA

- Embedding4k×256 da zero; porte sparse bilanciate; scelta fan-in/gain/bias e seme.
- Letture compatte potenziale+rate; scegliere popolazioni/pooling senza cambiare nodi.
- Readout256→vocabolario con E condivisa; definire porta di rientro attenzione e gate.

**Output:** testo→correnti e stato→logits; dimensioni/parametri/gradienti verificati, nessun bypass.

Evidenza: [RESOCONTO_FASE_2_2.md](RESOCONTO_FASE_2_2.md), [INTERFACCE_TESTO.md](INTERFACCE_TESTO.md), `results/phase22_toy.json`, `results/phase22_cns_t10.json`. Cinque test GPU; forward/backward/Adam dentro CUDA Graph; cleanup0MiB.

### 2.3 — Attenzione e memoria causale — COMPLETATA

- MHA4×64, RoPE, memoria128; Q stato provvisorio, K/V stati finali precedenti.
- SDPA matematico FP32; maschere validità, memoria vuota, cache funzionale e posizioni per slot.
- CUDA Graph: operazioni catturabili, forme/buffer statici, maschere dinamiche nei tensori.
- Test riferimento attenzione, nessun accesso futuro, eviction, reset e gradienti Q/K/V.

**Output:** modulo attenzione sostituibile, corretto anche con cache e padding.

Evidenza: [ATTENZIONE_API.md](ATTENZIONE_API.md), [RESOCONTO_FASE_2_3.md](RESOCONTO_FASE_2_3.md), `results/phase23_attention.json`. Sette test GPU; tre update Adam e134 replay inferenza in CUDA Graph, reset/padding/eviction; cleanup0MiB.

### 2.4 — Integrazione LM e generazione — COMPLETATA

- Catturare LM completo in CUDA Graph: training e generazione separati, forme statiche, nessun offload CPU.
- Collegare 2.1–2.3: pre→attenzione→reiniezione→post→logits→append memoria.
- T8 approvato:4pre+4post, corrente post token+feedback; forward sequenze, teacher forcing, CE mascherata.
- Generazione greedy/campionata, BOS/EOS, reset storie; contratto TBPTT completo.

**Output:** modello completo che produce token; checkpoint di struttura/configurazione, nessuna qualità presunta.

Evidenza: [RESOCONTO_FASE_2_4.md](RESOCONTO_FASE_2_4.md), [LM_API.md](LM_API.md), `results/phase24_toy.json`, `results/phase24_cns_t10.json`. Training/generazione CUDA Graph, checkpoint e gradienti verificati; cleanup0MiB.

### 2.5 — Verifica integrata e risorse — COMPLETATA

- Estendere test end-to-end causalità, equivalenza sequenza/token singolo, padding/reset, gradienti per modulo oltre smoke2.4; eviction128 sul LM completo, batch misti e carry TBPTT dopo update.
- Confrontare checkpoint funzionale; smoke training LM CNS sotto guardie; profilo iniziale memoria/tempo.
- Profilare separatamente core, attenzione, interfacce e copie/cache nel LM completo; attribuire il costo prima di scegliere ottimizzazioni. Riportare tempi GPU misurati, non stime energetiche Spikformer.
- Misurare firing/saturazione, scale token/feedback, gate, gradienti Q/K/V, raggiungibilità readout; distinguere memoria128/finestra TBPTT.
- Congelare configurazione diagnostica e passare a verifica apprendimento fase3.

**Output:** forward/backward/optimizer/generazione validi sul sistema completo; limiti risorse misurati.

Evidenza2.5: [RESOCONTO_FASE_2_5.md](RESOCONTO_FASE_2_5.md), `configs/diagnostic_v1.json`. Causalità/cache128/reset, carry6 update e checkpoint2/4/8 verificati. Confronto diagnostico AdamW/Muon/ROOT: [CONFRONTO_OTTIMIZZATORI.md](CONFRONTO_OTTIMIZZATORI.md); nessuna adozione.

**Output fase2:** forward LM + CE + generazione autoregressiva eseguibili.
**Passaggio:** forme, causalità, reset, padding e gradienti verificati; smoke CNS sotto guardie. Nessuna qualità ancora dichiarata.

## 3 — Dimostrare apprendimento e stabilità

Dettagli3.3–3.5: [PROTOCOLLO_FASE_3_B.md](PROTOCOLLO_FASE_3_B.md). Profilo paging opzionale autorizzato: [PROFILO_PAGING_FASE3.md](PROFILO_PAGING_FASE3.md); guardie originali restano default.

Sequenza: **3.1 preparazione → 3.2 overfit → 3.3 memoria → 3.4 ottimizzatori → 3.5 validation/stabilità**. Calcolo modello GPU FP32 con CUDA Graph, guardie RAM/VRAM; nessun offload CPU. Modifiche architetturali sempre da discutere prima dell'applicazione.

### 3.1 — Protocollo e telemetria — COMPLETATA

- Prima dei run: campione train, validation separata, semi, budget, criteri numerici successo/stop. Test finale escluso.
- Preparare runner diagnostico: CE mascherata, skip optimizer se zero target validi, confini storie/BOS, carry detached dopo update, log e salvataggio config/pesi diagnostici.
- Baseline Adam FP32, foreach=False, decay raw=0; gruppi core/interfacce espliciti. Fissare B/L iniziali sostenibili dalle prove2.5.
- Registrare firing, silenti/saturi, potenziale/soglia, gradienti e aggiornamenti core/porte/QKV/readout; scale token/feedback e gate. Seguirli nel tempo: silenzio iniziale ~90% non equivale a neuroni morti.

**Output:** protocollo breve + runner/log verificati con smoke catturato.
**Passaggio:** dati separati, metriche interpretabili, reset/skip corretti, guardie e salvataggio funzionanti; criteri dei sottopassi successivi fissati.

Evidenza3.1: [PROTOCOLLO_FASE_3.md](PROTOCOLLO_FASE_3.md), [RESOCONTO_FASE_3_1.md](RESOCONTO_FASE_3_1.md), `results/phase31_smoke.json`.

### 3.2 — Overfit controllato — COMPLETATA

- Ripetere piccolo campione train, baseline Adam; confrontare loss iniziale/finale, completamenti stessi prompt.
- Verificare aggiornamenti utili nucleo/interfacce/attenzione; loss in calo ≠ attività spike.
- Se necessario calibrare LR e clipping su gradienti finiti, una variabile alla volta. Variazioni gain/gate o dinamica da presentare prima quando cambiano la configurazione architetturale.

**Output:** curva di apprendimento + checkpoint diagnostico + generazioni prima/dopo.
**Passaggio:** riduzione loss secondo criterio3.1, aggiornamenti finiti, assenza di instabilità persistente. Memorizzazione del campione non prova generalizzazione; se fallisce, diagnosticare prima di aumentare scala.

Evidenza3.2: [RESOCONTO_FASE_3_2.md](RESOCONTO_FASE_3_2.md), `results/phase32_adam_lr1e4.json`. CE8,24→2,02, accuracy19/32, stop a360 update; criteri diagnostici superati, non memorizzazione perfetta o generalizzazione.

### 3.3 — Memoria, attenzione e TBPTT — ESEGUITA, CRITERI NON SUPERATI

- Verificare crescita feedback/potenziali osservata in3.2 su sequenze oltre16 token, con guardie invariate; stabilità lunga non ancora dimostrata.
- Task sintetici copia/ritardo con distanze prefissate e controlli semplici; separare successo del task da comprensione linguistica.
- Verificare gradiente/aggiornamenti Q/K/V/O e feedback; presenza ≠ vantaggio attenzione.
- Confrontare coerenza cache/stato fra training e generazione; valutare B/L e politica TBPTT, dichiarando stato prodotto dai pesi precedenti e detach tra finestre. Cache128 non significa retropropagazione lunga128.
- Eventuale confronto token+memoria contro sola memoria nel post: proposta da discutere prima, stesso grafo/T8/dati/seme e controllo della scala di corrente. Nessuna variante applicata automaticamente.

**Output:** risultati per distanza, diagnostica memoria e limiti TBPTT.
**Passaggio:** criteri copia/ritardo3.1 verificati, reset corretti, comportamento memoria spiegato. Fallimenti registrati e analizzati, nessuna utilità dell'attenzione presunta.

Evidenza3.3: [RESOCONTO_FASE_3_3.md](RESOCONTO_FASE_3_3.md). Stress256 passato; dev ritardi1/4/8:32,66%/9,82%/9,375%, sotto80%. Fase3 rimane aperta; altre diagnosi proseguono su richiesta utente.

### 3.4 — Confronto Adam/AdamW, Muon e ROOT — COMPLETATA

- Estendere lo smoke2.5 a confronto di apprendimento: inizializzazione/dati comparabili, LR adeguati per famiglia, stesso budget di ricerca iperparametri e più semi entro budget3.1.
- Distinguere confronto a dati uguali da confronto a tempo uguale; misurare loss train/validation, stabilità, tempo e memoria. Ultima loss train insufficiente per scegliere.
- Adam/AdamW: esplicitare beta2 e decay; a decay0 non attribuire differenze al nome dell'optimizer.
- Muon/ROOT sulle cinque matrici dense; AdamW su embedding, sinapsi e porte sparse. ROOT attuale usa coefficienti di ripiego: calibrazione specifica separata, non beneficio già verificato.
- Baseline: nessuna sostituzione automatica. Riportare risultato + motivo proposta.

**Output:** tabella qualità/costo/stabilità + configurazione candidata, oppure baseline confermata se non emerge vantaggio affidabile.
**Passaggio:** confronto documentato entro budget; incertezza dichiarata, niente vincitore dedotto dai sei update dello smoke.

Evidenza3.4: [RESOCONTO_FASE_3_4.md](RESOCONTO_FASE_3_4.md). Griglia24/24; Muon/ROOT migliorano CE ma non superano soglia2% in entrambi i semi. Baseline invariata.

### 3.5 — Validation, stabilità prolungata e chiusura — ESEGUITA

- Candidata: validation esclusa dal training; confronto inizializzazione, prompt fissi. Selezione su validation; test finale chiuso.
- Eseguire run diagnostico più lungo entro budget3.1: loss, attività, potenziali, gradienti, aggiornamenti, RAM/VRAM e velocità nel tempo.
- Verificare generazione: cache, BOS/EOS, storie indipendenti. Registrare capacità/fallimenti; qualità pretraining non presunta.
- Congelare config, checkpoint diagnostico, log; priorità efficienza/recupero archi fase4. Ripresa esatta completa: requisito fase5, assente nel checkpoint diagnostico.

**Output:** resoconto fase3 + configurazione stabile + curve/checkpoint.
**Passaggio fase3:** apprendimento e miglioramento validation secondo criteri3.1, generazione funzionante, nessuna instabilità persistente nel run prestabilito. Se mancano, tornare al sottopasso pertinente e lasciare fase3 aperta.

Evidenza3.5: [RESOCONTO_FASE_3_5.md](RESOCONTO_FASE_3_5.md). CE8,376→5,918;512 update; nessun allarme drift. Greedy:24 punti su entrambi i prompt. Resoconto complessivo: [RESOCONTO_FASE_3.md](RESOCONTO_FASE_3.md). Fase3 aperta; priorità diagnosi memoria/generazione.

## 4 — Efficienza e recupero connessioni

- Priorità dal profilo2.5: nucleo/backward; attenzione isolata0,177 ms contro nucleo pre+post11,57 ms.
- Verificare checkpoint attivazioni dentro CUDA Graph prima di adottarlo: segmento2 riduce picco incrementale42% nel probe eager, aumenta tempo34%.
- Recupero archi: controllare i37/2.241 nodi letti strutturalmente irraggiungibili alla soglia10; non rimuoverli automaticamente.
- Se utile dopo fase3: costo Muon/ROOT e calibrazione coefficienti per forme256; nessun beneficio del paper presunto.

- Profilare LM completo: forward/backward/optimizer, attivazioni, VRAM/RAM, token/s e byte/s.
- Provare CSR contro custom: stesso grafo, forward/gradienti equivalenti; misurare anche storage FP32 vs bool e workspace.
- Checkpoint puro segmenti2/4/8; correnti locali; indici int32 dove supportati, body ID originali int64.
- B/L/T separati: cambiare T modifica dinamica. T8 resta riferimento; ablation4+4/6+2 e T2/4/8 solo se utile.
- Recuperare archi 10→5→soglie inferiori secondo risorse; definire trasferimento pesi e nuovi archi. Tutti nodi sempre mantenuti.
- Valutare T ridotto, leak e detach_reset come varianti: cambiano dinamica/gradiente, richiedono confronto e discussione preventiva.
- SSA di Spikformer solo come variante separata se profilo utile: softmax non equivalente, Q/K binari incompatibili direttamente con RoPE, causalità/eviction da progettare. Query1×128: nessun vantaggio quadratico presunto.
- Non copiare BatchNorm lungo i token senza verificare causalità, né introdurre bypass del connettoma.
- BPE8k solo se motivato: probe attuale −2,96% token contro raddoppio embedding; confrontare byte/s e qualità a budget uguale, non sola compressione.
- GQA/8bit/custom kernel solo se profilo ne mostra utilità; nessuna installazione automatica.

**Output:** backend/configurazione misurati + massimo grafo sostenibile nel budget + regressioni di correttezza.
**Passaggio:** guadagno documentato o backend originale confermato; apprendimento fase3 ancora valido; headroom risorse verificato. Full graph non requisito già garantito.

## 5 — Pretraining controllato

- Congelare anche corrente post, T/leak/reset-gradient, backend attenzione e maschere; registrare eventuali varianti approvate.
- Congelare configurazione/tokenizer per run; fissare budget tempo/dati e frequenza valutazioni.
- Allenare su TinyStories; checkpoint riprendibili con optimizer, stato necessario, RNG e manifest.
- Monitorare validation, stabilità, throughput; stop secondo criteri prefissati, non solo loss train.
- Curriculum contesto/dati eventuale: esperimento esplicito; non cambiare simultaneamente archi/T/gain/tokenizer.
- Scegliere checkpoint tramite validation; test finale ancora escluso dalla selezione.

**Output:** checkpoint candidato + curve + costo effettivo + generazioni riproducibili.
**Passaggio:** training entro budget, valutazione validation completa, nessun problema numerico irrisolto. Qualità insufficiente → revisione mirata, nessuna promessa di vittoria.

## 6 — Benchmark e contributo della mosca

Evidenza6 (14 settembre): [RESOCONTO_FASE_6_CONTROLLI.md](RESOCONTO_FASE_6_CONTROLLI.md), `results/phase6_controls_summary.json`. CE DEV16 a pari target (52.673 a 2000 / 208.271 a 8000): mosca reale H1-10 5,087 / 3,959; rimescolata a gradi conservati (seed 41, init identica) 4,931 / 3,814; GRU 1e-3 4,213 / 3,429; Transformer 1e-4 4,410 / 3,563; GRU 1e-4 4,175 e Transformer 1e-3 4,323 a 8000; unigramma 5,843, bigramma 4,283. Lettura da protocollo: nessuna evidenza che il cablaggio reale aiuti (rimescolato meglio di 0,15-0,20 nat lungo tutta la curva, un seed); nessun vantaggio assoluto della mosca a pari dati. Nessuna modifica architetturale decisa. Ricerca bibliografica collegata: [RICERCA_CONNETTOMA_LM.md](RICERCA_CONNETTOMA_LM.md).

Evidenza6b (15-16 settembre): suite corta pre-registrata completa in [SUITE_TEST_FASE_6B.md](SUITE_TEST_FASE_6B.md) (A-F, S: 40 test tra run a 2000, inferenza e analisi) e promozioni a 8000 in [RESOCONTO_FASE_6_CONTROLLI.md](RESOCONTO_FASE_6_CONTROLLI.md) §6c. Esiti: tetto della pipeline con i null 3,65-3,70 a 8000 contro 3,959 del reale; il freno del cablaggio reale sta nell'ingresso (porte random sul grafo intero 3,689) e nell'inibizione (segni permutati 3,723); scelte biologiche che reggono a 8000: soglia sinaptica relativa (+0,06); falso positivo: lettura per gruppi anatomici (+0,24 a 2000, +0,015 a 8000). Kernel fuso cupy ×1,73 equivalente, adottato per i controlli.

- Eseguire protocollo fase1: piccoli modelli storici selezionati, baseline semplici, metriche comparabili.
- Ablazioni: senza attenzione; core congelato; grafo rimescolato controllando gradi/segni; nucleo rimosso/sostituito.
- Confronto equo → riaddestrare controlli dove necessario. Rimozione a inferenza misura dipendenza; baseline allenata resta necessaria.
- Dichiarare parametri extra, archi/sinapsi conservati, raggiungibilità, testo visto e tempo. Più semi dove sostenibile.
- Se sperimentate, includere varianti token/feedback, T/leak/reset-gradient e SSA nella matrice confronti; riportare differenze architetturali e hardware reale, non speedup dedotti da energia teorica.
- Report vittorie/sconfitte per prova; niente cherry-picking test o confronto PPL fra tokenizer diversi.

**Output:** tabella confronti + report limiti + modello/configurazione riproducibili.
**Completamento:** risultati verificabili e contributo del connettoma quantificato. Battere baseline è obiettivo sperimentale, non criterio da ottenere alterando il test.

## Sequenza / ritorni

`1 dati/protocollo → 2 LM → 3 apprendimento → 4 efficienza/archi → 5 pretraining → 6 confronto`

- OOM/stallo già in fase2–3: anticipare solo ottimizzazione necessaria della fase4.
- Modifica numerica/architetturale fase4 → ripetere verifiche pertinenti fase3.
- Baseline/ablazioni possono partire dopo fase3 per stimarne costo; protocollo fissato in fase1.
- Fallimento test finale → nuova ipotesi/versione; test già consultato dichiarato, nessun tuning nascosto.

## Pausa e ricerca

Test fermati su richiesta utente. Recupero3.3 e confronto memoria soglia10/5 eseguiti senza raggiungere80%; confronto linguaggio5 interrotto durante validation. Nessuna variante adottata. [Risultati recupero](RESOCONTO_RECUPERO_FASE_3.md), [dubbi/fonti e proposte](DUBBI_FASE_3_RICERCA.md). Criteri storici conservati; revisione proposta, non applicata.

## Suite diagnostica proposta per fase3

[SUITE_DIAGNOSTICA_FASE_3.md](SUITE_DIAGNOSTICA_FASE_3.md): T0 passato; T1b d1 controllo calibrato100%DEV. **T2/T3 eseguiti:** mosca d0 head18,85% DEV, probe uscite grezze92,29%, rappresentazione finale84,77%; lettura/ottimizzazione prioritarie. [T2/T3](RESOCONTO_T2_T3.md). Aggiornamento2000update: L16d1 75,756% controL8 68,75%; precedente mancato recupero limitato al budget corto. T6 confronto testo2000/replica/audit eseguiti; T7 topologia10/5 eseguita. Audit64 consultato una volta: CE5,161; test finale chiuso.

## Tabella avanzamento

Suite soluzioni aggiornata: [SUITE_SOLUZIONI_FASE_3.md](SUITE_SOLUZIONI_FASE_3.md). R1 lettura originale75,59%, reinserita R2 74,61%; calibrazione eseguita senza pruning. P1/P2/D1/D2/H1 testate; H1 scelta sul testo. Confronto2000target budget effettivo52.673: tied10 CE5,762; H1-10 5,087; H1-5 5,211. Replica H1-10 seed23 CE5,239. Audit64 CE5,161: supera unigramma, non bigramma. Fase3 resta aperta per memoria/generazione, non per mancanza di questi test. d8 lungo e ottimizzazioni ulteriori rinviati esplicitamente alla fase successiva.

Resoconto finale: sempre tabella aggiornata.

| Fase | Stato |
|---|---|
| 1 — Dati/tokenizer/protocollo | ✅ Completata |
| 2.1 — API nucleo/stato | ✅ Completata |
| 2.2 — Embedding/interfacce/readout | ✅ Completata |
| 2.3 — Attenzione/memoria | ✅ Completata |
| 2.4 — Integrazione LM/generazione | ✅ Completata |
| 2.5 — Verifica completa | ✅ Completata |
| 3.1 — Protocollo e telemetria | ✅ T0; T1 d1 calibrato,100%DEV |
| 3.2 — Overfit controllato | ✅ Overfit, T2/T3 e calibrazione R1/R2 eseguiti |
| 3.3 — Memoria/attenzione/TBPTT | ⚠️ T4/T5 eseguiti; criteri non superati |
| 3.4 — Confronto ottimizzatori | ✅ Completata |
| 3.5 — Validation/stabilità finale | ✅ Confronto/replica/audit/ripresa; qualità aperta |
| 4 — Ottimizzazione/più archi | ✅ Confronto operativo10/5; H1-10 scelta; ulteriori backend futuri |
| 5 — Pretraining | ⏸️ Fermato su richiesta a26.247; checkpoint salvato |
| 6 — Benchmark/ablazioni | ⚠️ Controlli 14/9: rimescolato 3,814 < reale 3,959 a 8000; GRU 3,429 e Transformer 3,563 sotto la mosca. Suite 6b 15-16/9 completa (`SUITE_TEST_FASE_6B.md`, `RESOCONTO_FASE_6_CONTROLLI.md` §6c): ogni null batte il reale; porte random sul grafo intero 3,689 a 8000; soglia relativa 3,903; occhi→fru/dsx unica coppia anatomica che funziona; kernel fuso ×1,73 equivalente. Ablazioni attention/nucleo sostituito da fare; decisioni di adozione all'utente |
