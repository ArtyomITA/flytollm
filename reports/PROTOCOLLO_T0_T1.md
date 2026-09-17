# T0/T1 — protocollo fissato prima dei run

Autorizzazione: «vai t0 e t1 caveman ultra». Nessuna modifica mosca.

- T0: oracle FIFO indipendente; d0/1/4/8; PAD congela, BOS reset; chunk8/16 conservano stato; perturba futuro; confronto generatore storico d1/4/8; testo next-token senza attraversare storie.
- Audit:64 IDs validation sorteggiati seed93013, esclusi intero elenco evaluation_indices validation, DEV16, primi82 IDs. Solo metadata; testi audit non aperti.
- Stream: Python Random seed93014 train400×32; seed93015 DEV64×32. Stesso elenco ogni ritardo. B2, BOS+32 simboli400..407+7PAD. Target validi32−d/stream. Train eval primi64 stream già allenati; DEV separato. Manifest conserva stream esatti.
- Controllo GRU: embedding4096×32, GRUCell32→64, head64→4096, FP32 GPU; Adam0,003, clip1, seed89, max200 update. Sequenza40 intera, reset ogni coppia; no TBPTT interno. CUDA Graph train/eval, equivalenza GPU eager pre-training, warmup ripristinato.
- Differenza deliberata da mosca: BPTT40, embedding/head separati, architettura standard. Calibra fattibilità; NON confronto causale architetture o TBPTT. Budget400 stream nuovi =12.800 simboli; target200×2×(32−d). No griglia LR.
- Aprire d4/d8 solo se d0 E d1 DEV accuracy4096≥80%. Criterio operativo diagnostico, non revisione criteri storici.
- Baseline simboli: oracle100%, chance attesa12,5% fra8, costante400 misurata. Riportare CE4096, accuracy4096, ristretta8.
- Testo: train18..81, max128 target/storia; DEV16 storico stesso limite. Unigram add-one su4096; bigram p(y|x)=(count(x,y)+10*p_unigram(y))/(count(x)+10). Count solo train, no fit DEV. CE naturale, accuracy, target totali/per storia. Score GPU catturato; CPU conteggi/preparazione.
- Guardie paging approvate; un worker alla volta, max600s, heartbeat32update, abort5update consecutivi>2s. Finitezza loss/gradienti/pesi e cleanup allocator0 obbligatori.
- Hash sorgenti/protocollo/stream/dataset; tempi training, picco VRAM, train/DEV finali. Nessun T2 successivo implicito.
