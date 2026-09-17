# Fase3.1 — protocollo e telemetria

**Completata.** Protocollo fissato prima overfit: [PROTOCOLLO_FASE_3.md](PROTOCOLLO_FASE_3.md), config `configs/phase3_protocol_v1.json`.

- Train: storie0/1, primi16 target ciascuna;32 target distinti, B2,L8,2 finestre/epoca. Reset ogni epoca; carry detached fra finestre, stato da pesi precedenti. No attraverso storie.
- Validation:16 indici riservati dal manifest esistente; test finale non valutato. No validation per dimostrare sola memorizzazione3.2.
- Adam FP32, gruppi core/interfacce/attenzione, decay0, clipping globale1; forward/backward/optimizer catturati CUDA Graph. No cambio dinamica modello.
- Obiettivo3.2: dimezzare CE, accuracy>=50% stesso campione. Max400 update LR1e-4; solo se insufficiente,400 update LR1e-3 da init identica. Max10min/caso, guardie preesistenti.

## Smoke eseguito

Due epoche,4 update. CE8,2407→8,0336, accuracy0%; non prova ancora overfit. Verificati:

1. Batch vuoto saltato **dopo Adam con momenti non nulli**: parametri, momenti, stato invariati. Evita update spurii anche con loss=0.
2. Posizioni cache8/16 nelle due finestre; reset prima epoca successiva.
3. Gradienti/parametri finiti; delta pesi nei tre gruppi; logging loss, potenziali, firing, gate, correnti.
4. Salvataggio/ricaricamento: tutti tensori state_dict identici su GPU. Checkpoint diagnostico, non ripresa esatta training.
5. Picco allocato388,12 MiB; cleanup0 MiB. Supervisore senza abort. Avvio lento, RAM variabile; nessuna soglia allentata.

Telemetria firing = ultima fase post ultima finestra; gradienti dopo clipping, norma globale prima. Non media intero corpus. Hook osservano operazioni esistenti, senza cambiare equazioni. Prima3.2 misura corrente feedback allineata al gate precedente update, senza alterare update pesi; hash runner registrati per run.

Entrambi prompt fissati = `Once upon a`: continuazioni greedy uguali attese, non due prove indipendenti. Output integrali nei JSON, senza selezione estetica.

Codice: `train_diagnostic.py`. Evidenze: `results/phase31_smoke.json`, `.worker.json`, `.jsonl`, `.diagnostic.pt`.