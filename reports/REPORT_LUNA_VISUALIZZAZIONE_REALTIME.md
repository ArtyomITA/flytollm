# Luna — visualizzazione attivazioni realtime

## Esito

Architettura minima: **tap GPU fisso → snapshot campionato → ring buffer latest-wins → endpoint locale binario → WebGL2 in Worker/OffscreenCanvas → LOD**.

Telemetria osserva modello intero; no rimuove/disattiva nodi/arcs. Campionare vista ≠ pruning modello.

## Stato locale

- Mosca maschio: 166.700 nodi; grafo soglia ≥10: 2.753.975 archi.
- `fly_graph.py`: `body_ids` ordinati; `src/dst` = indici contigui, non `bodyId`.
- `body-annotations...feather`: campi `bodyId`, `superclass`, `somaLocation`; locale mostra `somaLocation=[x,y,z]`.
- `fly_core.py`: stato `(voltage, spike)`, `rate` per nodo; `Propagate` sparse chunked. `new.voltage` = voltage post-reset. `phase3_extended.py` già ha hook GPU + statistiche aggregate in capture/replay.
- HTML esistente (`connettoma-parlante-recuperato.html`): Canvas2D, 2.400 nodi/archi sintetici; fallback narrativo, non vista live/anatomica.

## Pipeline proposta

### 1. Telemetria GPU

Prima capture, preallocare buffer device a indirizzi/formati fissi:

- `scalar_dev[S]`: step, token, substep, loss, firing medio, `%zero/%sat`, voltage RMS/max, gradient norm/max per `core/interfaces/attention/readout`, gate, tempo GPU.
- `node_dev[R,K,C]`: campione deterministico stratificato `K=8.192` nodi; iniziare `R=2–4` slot, `C={voltage_post, rate, token_current, feedback}` FP16. Includere sempre sensoriali, discendenti/motori, quota intrinseci; `node_index`/`bodyId` = metadata statico.
- `edge_event_dev`: solo eventi attivi/top-k, `(edge_index, substep, amplitude, sign)`. Mai copiare 2,75M archi ogni frame.
- Gradienti: scalari ogni replay; gradiente nodo campionato solo replay diagnostico bassa frequenza, via tap autograd su tensori intermedi. `raw.grad` = gradiente archi/pesi, non gradiente nodo.

Tap = operazione tensoriale catturata, forme/puntatori invarianti; niente `.item()`, `.cpu()`, `synchronize()` o alloc Python nel percorso hot. Leggere `new.voltage`/`rate` già esposti; token/feedback dai tap già usati dall’harness. Pre-reset `u` non esposto dall’API: richiede strumentazione separata + verifica equivalenza, quindi partire da `voltage_post`.

### 2. GPU → CPU senza fermare training

Dopo replay, stream copia separata aspetta evento GPU, esegue `copy_(non_blocking=True)` verso buffer host **pinned** preallocati, poi registra evento host-consumer. Ring host latest-wins da solo insufficiente: serve ownership anche del buffer device sorgente. Ogni slot ha stato `FREE → GPU_WRITTEN → D2H_INFLIGHT → HOST_READY → CONSUMED/FREE`, generation counter, `cudaEvent` `copy_done`. Consumer CPU legge solo dopo `copy_done.query()`. Prima nuovo replay, slot device riutilizzabile solo se evento precedente completato; se tutti occupati, telemetria salta campione (`dropped_busy++`) e training prosegue. Nessun overwrite device mentre D2H in volo. Implementazione: ring device+host preallocato con slot-id GPU, oppure capture separata per slot; puntatori fissi per capture. Ring iniziale: 8 slot, 256 snapshot storici max; ultimo frame completo disponibile. Persistenza disco opzionale, fuori hot path.

Formato frame binario little-endian, versionato: header `{step, token, substep, count, channels, model_hash}` + record `{node_index, values[]}` + eventi edge. Trasmettere `bodyId`/coordinate una volta come manifest statico. FP16 per attivazioni; FP32 solo se precisione diagnostica richiesta.

Ordini grandezza: vettore FP32 su 166.700 nodi ≈0,67 MB; voltage+rate ≈1,33 MB/frame, quindi 10 Hz ≈13,3 MB/s prima altri canali. Campione K=8.192, 4 canali FP16 = **64 KiB payload** (`8192×4×2`); con `node_index` uint32 per record = 96 KiB/frame, mentre indici statici nel manifest lasciano 64 KiB; 2 Hz ≈128 KiB/s payload. Header/eventi esclusi. Numeri guidano rate/canale; misurare sul sistema reale.

### 3. Frontend

**Scelta minima: WebGL2 + `drawArraysInstanced` + Worker/OffscreenCanvas.** Buffer statici: posizioni, sign/role, node index; buffer dinamico: activation/rate. Disegnare punti/sprite istanziati. Archi: solo aggregati a zoom basso; ROI/edge attivi/top-k a zoom alto. Static edge buffer completo resta caricabile per query, non necessariamente renderizzato tutto.

**WebGPU opzionale:** storage buffers + compute culling/LOD + indirect draw, `GPUQueue.writeBuffer` per upload snapshot. Usarlo se browser/driver lo espongono e profiling dimostra vantaggio; evitare `mapAsync` nel render loop: buffer mappato non usabile da GPU e `MAP_READ` ha vincoli usage.

Canvas2D/DOM/SVG/D3 per 2,75M linee: demo/fallback, non renderer principale. Three.js/deck.gl aggiungono comodità ma non eliminano bisogno di buffer GPU, culling, LOD; non installare per MVP.

## Anatomia, layout, mapping

Usare `somaLocation` come coordinate anatomiche raw, preservando ordine assi nel manifest. NeuPrint definisce `somaLocation` come x,y,z centro cellula; Feather locale non porta unità esplicite. Pagina MaleCNS dichiara **8 nm/unità per skeleton SWC**, ma non dimostra stessa unità per `somaLocation`: non convertire automaticamente in nm. Fino a verifica fonte, manifest deve dichiarare `coordinate_space="male_cns_dataset"`, `units="to_verify"`; normalizzare solo lato renderer. Soma point = centro cellula; morfologia completa richiede skeleton, rinviata.

Manifest minimo:

```text
nodes: node_index, bodyId, soma_xyz_i64, coordinate_space, units, superclass/role, sign
edges: src_index_u32, dst_index_u32, weight_f32, edge_sign
```

`bodyId` = identificatore esterno; array indicizzano `node_index`. Generare `bodyId → node_index` da ordinamento già in `load_graph`; verificare unicità, cardinalità, round-trip edge. Vista anatomica (3D/proiezione x-y/x-z) e vista artificiale per ruolo (sensory → interneuroni → descending/motor) = due layout selezionabili: layout artificiale non sovrascrive coordinate né cambia modello.

## Librerie/API: scelta

| Opzione | Pro | Costo/rischio | Decisione |
|---|---|---|---|
| WebGL2 raw + OffscreenCanvas | supporto maturo; instancing; integrazione diretta con HTML attuale; nessun pacchetto | culling/LOD da scrivere | **MVP** |
| WebGPU raw | storage/compute/indirect draw; LOD GPU naturale | disponibilità browser/driver; API più nuova; testare fallback | fase2 |
| Canvas2D attuale | zero setup; narrativa | CPU draw, archi sintetici, scala insufficiente | fallback |
| viewer DOM/SVG | ispezione singolo neurone facile | impossibile per grafo intero | escluso |

Worker/OffscreenCanvas separa render e DOM. `EXT_disjoint_timer_query_webgl2` misura GPU render senza confonderlo col tempo CPU; lato training usare CUDA Events per tempo GPU, host wall-time solo attorno a batch misurato.

## Playback storico

Ogni frame porta `step/token/substep`, `model_hash`, `sample_id`, `dropped_before`. Frontend mantiene coda bounded; play/pause/scrub seleziona frame già ricevuti. Per storia lunga, writer asincrono salva chunk binari compressi; playback mostra timestamp e segnala buchi/drop. Snapshot non ricevuto non va interpolato come dato scientifico; interpolazione solo effetto visivo esplicitamente marcato.

## Overhead e criteri

Confrontare stesso seed/input/shape in quattro run: baseline; scalar-only; K=8.192 a 2 Hz; K=32.768 a 5 Hz + eventi. Misurare mediana/p95 replay, token/s, CUDA Event GPU time, `max_memory_allocated/reserved`, NVML used/free, RAM/commit, bytes/s, ring depth/drop, frame time, timer-query WebGL.

Gate iniziale (da confermare con benchmark): slowdown mediano ≤5%, p95 ≤10%; incremento VRAM ≤256 MiB, RAM ring ≤512 MiB; zero errori CUDA Graph; logits/stato entro tolleranza già verificata (max logit ≈1,43e−6); drop ammesso solo frontend/telemetria e sempre contato. Se gate fallisce: ridurre frequenza/K/canali, mai alterare grafo per ottenere visualizzazione.

## Piano MVP, senza cambio modello

1. Creare manifest statico da Feather/NPZ: bodyID, soma xyz, sign/role, src/dst/weight; test cardinalità/round-trip.
2. Aggiungere tap buffer statici al runner già catturato; scalar always-on, K=8.192 campionato; copia su ring pinned con stream/eventi separati.
3. Servire manifest/frame via HTTP locale binario con endpoint latest/history; niente dipendenze nuove.
4. Sostituire solo renderer HTML con WebGL2 instanced + OffscreenCanvas; aggiungere anatomico/artificiale toggle, ROI, rate/voltage/color scale, playback.
5. Eseguire matrice overhead/equivalenza; pubblicare numeri e drop. Gradiente nodo e pre-reset solo modalità diagnostica esplicita.

## Fonti primarie

- NVIDIA, [NVML API](https://docs.nvidia.com/deploy/nvml-api/): query GPU, memoria, utilizzo, temperatura, clock, processi; campioni utilization hanno periodo prodotto-dipendente.
- NVIDIA, [CUDA Graphs Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html): graph replay riduce launch overhead; nodi kernel/memcpy/event/semaphore; forme/indirizzi fissi = vincolo operativo.
- PyTorch, [CUDA semantics — graphs](https://docs.pytorch.org/docs/main/notes/cuda.html): replay riusa stessi indirizzi; input aggiornati copiando in buffer statici; `.item()`/sync CPU e dynamic shapes vietati in capture.
- NVIDIA, [CUPTI Activity API e overhead](https://docs.nvidia.com/cupti/main/main.html): raccolta asincrona buffered/callback; abilitare solo attività interessate, flush meno frequente; tracing meno invasivo del metrics profiling.
- NVIDIA, [CUDA API sync behavior](https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html) e [Best Practices — pinned memory](https://docs.nvidia.com/cuda/archive/12.6.1/pdf/CUDA_C_Best_Practices_Guide.pdf): trasferimenti host/device dipendono da memoria pinned/pageable; `cudaMemcpyAsync` richiede pinned per asincronia utile.
- W3C, [WebGPU Candidate Recommendation](https://www.w3.org/TR/2026/CRD-webgpu-20260109/): `writeBuffer`, mapAsync e vincoli usage/mapping; buffer mappato non usabile da GPU.
- Khronos, [WebGL 2.0 spec](https://registry.khronos.org/webgl/specs/latest/2.0/) e [instanced draw](https://developer.mozilla.org/en-US/docs/Web/API/WebGL2RenderingContext/drawArraysInstanced): buffer/typed arrays e draw istanziato.
- Khronos, [EXT_disjoint_timer_query_webgl2](https://registry.khronos.org/webgl/extensions/EXT_disjoint_timer_query_webgl2/): query GPU elapsed/disjoint, risultato letto quando disponibile.
- W3C, [WebGPU canvas su OffscreenCanvas/Worker](https://www.w3.org/TR/2022/WD-webgpu-20220608/) e MDN, [OffscreenCanvas](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas): render separato dal main thread.
- Janelia, [MaleCNS download](https://male-cns.janelia.org/download/): Feather annotations, connectome, soma/skeleton; pagina specifica 8 nm/unità per skeleton SWC, non assunto per `somaLocation`; dataset CC-BY.
- NeuPrint, [data model](https://neuprint.janelia.org/public/neuprintuserguide.pdf): `bodyId` unico/random; `somaLocation` centro cellula x,y,z.

Fonti consultate 2026-09-13. Nessuna installazione/download pesante, nessuna modifica al modello eseguita.