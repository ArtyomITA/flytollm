# Mosca + attenzione — specifica v0.6

Data: 2026-09-13. Stato: **fasi1/2.1–2.5 fatte; confronto2000, replica/audit, riprese verificati; pretraining notturno avviato**.

Profilo notturno da confronto autorizzato: **H1, head separata, soglia sinaptica10**. CEDEV5,087 vs5,762 tied10 e5,211 H1-5, stessi52.673 target. Totale5.459.469 parametri,166.700nodi,2.753.975archi. Confronto H1-10/H1-5 esteso a8000 in notte; no router MoE. API base compatibili con head tied; variante H1 applicata esplicita dal trainer, serializzata nel checkpoint.

Sequenza operativa: [PIANO_FASI.md](PIANO_FASI.md) — 6 fasi, output e criteri passaggio.

Revisione ricerca: `REVISIONE_OTTIMIZZAZIONI.md`. Architettura confermata; rafforzati vincoli autograd, stabilità, ordine esperimenti. Report ricerca = proposte, non piano operativo.

## Obiettivo / confini

- Modello linguistico da connettoma; futuro confronto con piccoli modelli storici su benchmark scelti prima del training.
- Massima conservazione praticabile di neuroni, archi diretti, identità anatomiche. Moduli artificiali piccoli, contributo misurabile.
- Architettura: **nucleo LIF ricorrente + attenzione temporale causale**. No Transformer standard; no SpikeGPT/RWKV.
- No LFM, encoder preaddestrato, distillazione in prima versione. Pesi appresi da zero.
- Esecuzione: modello solo GPU + CUDA Graph; CPU per caricamento, costruzione dati, supervisione. No offload né fallback CPU. Eager GPU ammesso come riferimento test.
- Hardware verificato: GTX 1080, 8 GiB VRAM; Windows; PyTorch 2.14.0+cu126.

## Configurazione candidata

| Componente | Profilo corrente |
|---|---|
| Tokenizer | Byte-level BPE; vocabolario totale 4.096, speciali inclusi |
| Embedding | `E[V=4096, d=256]`; inizializzazione casuale; trainabile |
| Nucleo | 166.700 body ID fissi; archi filtrati, direzioni conservate |
| Dinamica | LIF ricorrente; stato persistente fra token |
| Aggiornamenti | Budget iniziale T=8/token; 4pre+4post approvati; post con corrente token+feedback |
| Attenzione | Un modulo MHA softmax; 4 teste × 64 dimensioni |
| Posizioni | RoPE su Q/K; indici coerenti durante scorrimento cache |
| Memoria | Ultimi 128 token; K/V ricavati dagli stati del grafo |
| Uscita | Stato finale → 256 dimensioni → `Hᵀ` → logits; `H[4096,256]` separata da E |
| Precisione | FP32 iniziale; altre precisioni solo dopo verifica |

Valori iniziali, **non ottimi dimostrati**. Contesto 128 ≠ finestra backprop 128. Batch e lunghezza backprop da dimensionare sul LM completo.

## Tokenizer ≠ embedding

1. Allena BPE solo su training set; alfabeto byte completo; conserva maiuscole e punteggiatura.
2. Definisci EOS e politica BOS/PAD; salva vocabolario, regole, ID speciali, hash.
3. Testo → frammenti → ID. No conoscenza linguistica preaddestrata.
4. Lookup `e_t = E[id_t]`: vettore di 256 numeri. Stesso token → stessa riga; contesto dal nucleo.
5. Aggiorna E via loss linguistica con altri pesi. No pretraining separato embedding.
6. Congela tokenizer durante esperimento; cambiarlo = rimappatura/reinizializzazione e nuovo training.

E: 1.048.576 parametri; 4 MiB FP32; ~16 MiB con gradiente + due momenti Adam, temporanei esclusi.
H1: altri1.048.576 parametri; inizio `H = E.clone()`, poi gradienti indipendenti. Baseline storica tied: `H = E`. Separazione non aggiunge bypass: z sempre dal grafo dopo attenzione e fase post.
BPE scelto per ridurre aggiornamenti costosi grafo per frase; vantaggio da misurare. Caratteri: baseline diagnostica.

## Flusso causale per token t

```text
id_t → E[id_t] → ingresso sparso → correnti sensoriali
     → grafo: T_pre aggiornamenti → stato provvisorio
     → lettura compatta + normalizzazione → Q_t
Q_t + memoria K/V dei token < t → MHA + proiezione → segnale recuperato
     → rientro sparso nel grafo → T_post aggiornamenti
     → stato finale → lettura compatta → z_t[256]
     ├→ logits = z_t @ H.T → previsione id_(t+1)
     └→ K_t/V_t → append cache; espulsione posizione più vecchia se >128
```

- `T_pre + T_post = 8` budget candidato; no raddoppio implicito. Baseline approvata4+4; alternative solo dopo discussione.
- Primo token: memoria vuota → contributo attenzione zero.
- Ingresso token mantenuto in tutti gli8 sottopassi, come approvato; rientro attenzione in fase post.
- Q da stato provvisorio; K/V da stati finali precedenti. No dipendenza circolare; no token futuro.
- Normalizzazione/proiezioni Q/K/V/O trainabili; partenza semplice: LayerNorm per rappresentazioni compatte.
- Output attenzione passa nel grafo prima del readout; no collegamento diretto embedding/attenzione → logits.
- Stato, cache, posizioni separati per esempio; reset al confine storia. Packing: no contaminazione fra storie.

## Interfacce artificiali ↔ connettoma

- **Ingresso:** ogni neurone sensoriale riceve poche componenti del vettore 256, con pesi appresi. Mappa sparsa fissa e riproducibile; fan-in8, seme17, bias iniziale0.6, gain0.6. Conteggio pesi ~`N_sensory × fan_in`.
- **Rientro attenzione:** proiezione verso porte esplicite del grafo; porte sensoriali, pesi separati, seme18, gate0.1.
- **Lettura:** aggregazione popolazioni → piccola proiezione trainabile → 256 componenti. 2.241 nodi motori/discendenti/efferenti →77 gruppi (classe/lato, blocchi max32 per bodyId), medie potenziale+rate →154 segnali. Dettagli: `INTERFACCE_TESTO.md`.
- Segnali possibili: potenziale, spike/rate della fase. Scelta e scala da verificare; evita perdita d'informazione nascosta dal pooling.
- Porte: copertura bilanciata delle 256 componenti; scala coerente col fan-in; traccia seme, gain, frazione neuroni raggiungibili/attivi.
- Lettura candidata: potenziale + spike/rate separati; sola tensione post-reset perde traccia dei neuroni appena attivati. Pooling da confrontare, non scegliere per sola intuizione.
- Reiniezione: gate trainabile inizializzato piccolo ma non nullo; verifica gradiente Q/K/V. Ampiezza da calibrare, non valore universale.
- Evita matrice densa `256 × 166700`: 42.675.200 pesi solo per un adattatore.
- No equivalenza «token/parola = regione cerebrale». Porte/aggregazioni sono convenzioni artificiali.
- Attenzione = memoria/comunicazione aggiunta; mantenere archi anatomici ≠ conservare tutta la fisiologia.

## Nucleo esistente / conservazione

- Codice: `fly_core.py`, `fly_graph.py`. Stato operativo: `BENCHMARK_STATO.md`; misure: `RISULTATI_BENCHMARK.md`.
- Insieme nodi fisso; no ritaglio anatomico. Anche nodi isolati mantenuti.
- Archi verificati: soglia 10 → 2.753.975; soglia 5 → 6.242.118. Totale filtrato senza soglie: 25.582.938.
- Soglie = compromesso computazionale; recupera più archi progressivamente. Nodi completi ≠ connessioni complete.
- Peso ricorrente: `sign[src] × softplus(raw_weight)`. Segni fissi, magnitudini trainabili, topologia fissa.
- Segni da neurotrasmettitori: approssimazione esplicita. Conteggi sinaptici: prior d'inizializzazione, non efficacia biologica esatta.
- LIF: accumulo + perdita + soglia + reset; derivata surrogata nel backward. Parametri dinamici artificiali.
- Stato attuale: obiettivo sintetico di firing rate; **no qualità linguistica dimostrata**.
- API fase2.1 implementata: `initial_state`, `advance`, `reset_state`, `detach_state`; stato/rate separati dalla loss sintetica; backward primo ordine. Contratto: `CORE_API.md`. Embedding e interfacce 2.2 implementati, verificati su GPU con CUDA Graph; attenzione 2.3 implementata e verificata separatamente (`ATTENZIONE_API.md`). Integrazione2.4 completata (`fly_lm.py`, `LM_API.md`); verifica completa2.5 da fare.

## Attenzione sostituibile

Separa: `GraphEncoder` → `QKVProjection` → `PositionEncoding` → `MemoryCache` → `AttentionBackend` → `GraphInjector`.

- Implementata2.3: `fly_attention.py`, MHA4×64/RoPE/cache128 funzionale, backend SDPA matematico forzato;262.656 parametri. Riferimento softmax/matmul e CUDA Graph verificati su GPU; contratto `ATTENZIONE_API.md`.
- GTX 1080: backend matematico FP32 come base; non richiedere FlashAttention-2 né installare dipendenze CUDA aggiuntive senza necessità.
- Query singola + cache del passato: tutti gli elementi validi della cache sono leggibili. Non applicare alla cieca `is_causal=True` su Q/K di lunghezze diverse; maschera su posizioni reali.
- Caso v1 Q lungo 1/cache solo passato: `is_causal=False` + maschera validità; booleano SDPA True=ammesso. Primo token senza memoria: contributo zero.
- Cache con padding/batch diversi: maschere per esempio. Dropout iniziale zero; in eval sempre zero.
- Backend equivalente: pesi riutilizzabili, equivalenza numerica verificata.
- GQA: modifica teste K/V e cache; conversione + eventuale ulteriore training.
- Attenzione lineare: modifica memoria/matematica; nuova variante architetturale, non semplice cambio kernel.
- RoPE/contesto: aggiornamenti da validare; no estensione illimitata garantita.
- Ricorrenza grafo → elaborazione token sequenziale anche nel training. SDPA non rende il nucleo parallelizzabile lungo il testo.

## Training / inferenza

- TinyStories: prima palestra; dati ulteriori scelti secondo benchmark futuri. Split e test separati.
- Teacher forcing: input vero `id_t`, target vero `id_(t+1)`; cross-entropy sui logits, padding escluso.
- Gradienti: embedding + adattatori + magnitudini grafo + attenzione + readout + H1. Run corrente: Adam FP32 capturable, `foreach=False`, decay0, LR1e-4 e clip1; unico gruppo ottimizzatore, update ogni16posizioni con TBPTT8.
- **No decay automatico su raw:** raw negativo spinto verso zero aumenta `softplus(raw)`. Eventuale penalità futura su magnitudini o deviazione dal prior, esplicita e separata.
- Baseline tied: stesso parametro E per lookup e uscita, registrato una volta. H1 corrente: E lookup e H uscita distinti, entrambi trainabili; gradiente H denso.
- Backprop troncata: conserva valori stato/cache, interrompi storia autograd ai confini espliciti. Distanza consultabile ≠ distanza attraversata dal gradiente.
- Non staccare K/V indiscriminatamente dentro finestra training: preserva apprendimento proiezioni.
- Cache training funzionale: niente ring buffer in-place che sovrascriva tensori richiesti dal backward. Checkpoint: segmenti puri, stato/cache restituiti; append/eviction mai su oggetti globali durante ricomputazione.
- Genera correnti per token dentro i segmenti; evita tensore denso permanente `[L,B,N]`. Checkpoint `use_reentrant=False`; confronta output/gradienti e costo prima dell'adozione.
- Continuazione dopo optimizer.step: stato/KV conservati vengono da pesi precedenti; approssimazione TBPTT dichiarata. No `retain_graph=True` fra update, no riuso di pesi trasformati obsoleti.
- Padding: non evolvere stato/cache degli slot terminati. Gradient accumulation: normalizza sul totale token validi; non estende finestra temporale dei gradienti.
- Inferenza: pesi fissi; campionamento/argmax → token successivo; stato/cache aggiornati con lo stesso ordine del training.

## Verifiche prima del training lungo

1. Tokenizer: round-trip testo; EOS; assenza dati test nel training tokenizer.
2. Causalità: cambiare suffisso futuro non cambia logits precedenti; reset/padding/cache corretti.
3. Gradiente: finito e non nullo nei componenti attivi; core con spike non degeneri.
4. Piccolo campione: apprendimento effettivo; poi validazione mai vista e generazioni.
5. Misura LM completo: VRAM/RAM, token/s, testo processato/s, stabilità; rispetta supervisore e guardie esistenti.
6. Confronti: senza attenzione; grafo rimescolato con proprietà controllate; nucleo rimosso/sostituito; piccoli modelli di riferimento.
7. Budget, tokenizzazione e metriche dichiarati. Perplexity con tokenizer diversi non direttamente confrontabile: usa metriche comuni, es. bit/byte o prove identiche.

## Ottimizzazioni — ordine operativo

1. **Correttezza LM:** API sottopassi, CE, causalità, gradienti, piccoli task copia/ritardo e overfit. Telemetria potenziale/soglia, neuroni silenti/saturi, gradienti e aggiornamenti per gruppo; clipping solo su gradienti finiti.
2. **Profilo completo:** forward/backward/optimizer separati; B1/L4 iniziale poi crescita sotto guardie. KV grezzi B8/W128/d256 FP32 = 2 MiB; attivazioni LIF multi-GiB nei vecchi test. No priorità GQA per risparmio VRAM v1.
3. **Backend sparse candidato:** CSR `A[dst,src] @ s.T`, senza densificare. PyTorch 2.14 documenta backward CSR×dense; verifica build/device, gradiente valori→raw, duplicati, isolati e permutazione archi. Backend attuale resta riferimento. Misura anche memoria salvata: autograd CSR potrebbe conservare spike FP32 invece dei bool attuali. Adotta solo dopo equivalenza e vantaggio complessivo sotto budget.
4. **Memoria:** checkpoint puro segmenti 2/4/8 token; correnti per token; poi indici compatti int32 se API compatibili, ID biologici originali int64. Mai creare `[B,E]` completo o `[N,N]` denso.
5. **Tokenizer:** BPE4k baseline; confronto offline 2k/4k/8k su stesso campione train, byte/token e distribuzioni lunghezze. Eventuale confronto LM dei candidati migliori a tempo uguale; riporta anche testo visto.
6. **Dinamica:** confronto 4+4 vs 6+2 a T8; poi T2/4/8 solo come varianti, poiché cambiano leak/token, integrazione e percorsi. Non modificare insieme T, gain e soglia.
7. **Più archi:** preserva ID; esplicita trasferimento pesi comuni/inizializzazione nuovi quando cambia soglia. Distingui espansione checkpoint da riaddestramento da zero; misura raggiungibilità e conteggi sinaptici conservati.

Rinviati fino a evidenza: GQA/attenzione lineare, optimizer8bit, ALIF/e-prop, eventi, kernel custom. Ottimizzazioni event-driven devono preservare anche gradiente delle sorgenti silenti. No cambio stack come ottimizzazione automatica. CUDA Graph richiesto già dalle interfacce 2.2; ricattura e verifica sul LM completo 2.4–2.5. Offload CPU escluso.

**Da chiudere prima del run:** ripartizione T, calibrazione inizializzazioni, batch, finestra gradienti, learning rate, quantità dati, benchmark e budget confrontabili.
Nessuna promessa di accuratezza, velocità o capacità emergenti. Componenti aggiunte conteggiate; contributo del connettoma verificato sperimentalmente.

## Fonti

- [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
- [PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.sparse.Embedding.html)
- [Tokenizers / BPE](https://huggingface.co/docs/tokenizers/quicktour)
- [RoPE](https://arxiv.org/abs/2104.09864)
- [GQA](https://arxiv.org/abs/2305.13245)
- [FlashAttention: requisiti](https://github.com/Dao-AILab/flash-attention)
- [TinyStories](https://arxiv.org/abs/2305.07759)
- [PyTorch 2.14 sparse.mm](https://docs.pytorch.org/docs/2.14/generated/torch.sparse.mm.html)
- [AdamW: aggiornamento parametri](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)
- [Checkpoint](https://docs.pytorch.org/docs/2.14/checkpoint.html)
