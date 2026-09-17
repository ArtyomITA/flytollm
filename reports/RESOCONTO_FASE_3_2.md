# Fase3.2 — overfit controllato

**Criterio diagnostico raggiunto al primo tentativo Adam LR1e-4.** Stop a180 epoche/360 update, prima del limite200/400. Secondo tentativo LR1e-3 non eseguito perché non necessario. Architettura invariata.

## Risultato

| Misura | Inizio | Fine |
|---|---:|---:|
| CE media, stesso campione | 8,2407 | 2,0216 |
| Accuracy teacher-forced | 0/32 | 19/32 =59,375% |
| Posizioni target distinte | 32 | 32 |
| Esposizioni target nel training | 0 | 5.760 |

CE giù75,47%. Criteri pre-fissati: CE almeno dimezzata, accuracy>=50%. Entrambi superati. **Non memorizzazione perfetta, non generalizzazione:**32 posizioni di due prefissi TinyStories ripetute180 volte, non5.760 token nuovi, non due storie intere. Accuracy teacher-forced = prevedere prossimo token dato prefisso corretto; non misura generazione libera.

| Epoche | CE | Accuracy |
|---:|---:|---:|
| 0 | 8,241 | 0% |
| 20 | 6,212 | 18,75% |
| 40 | 4,517 | 12,50% |
| 60 | 3,466 | 18,75% |
| 80 | 3,063 | 18,75% |
| 100 | 2,887 | 25,00% |
| 120 | 2,740 | 28,125% |
| 140 | 2,556 | 34,375% |
| 160 | 2,346 | 37,50% |
| 180 | 2,022 | 59,375% |

Loss e accuracy non si muovono insieme a ogni checkpoint: più probabilità al token corretto abbassa CE prima che diventi il token più probabile.

## Generazione libera, senza scegliere esempi favorevoli

I due prompt prefissati sono identici: `Once upon a`. Greedy dà quindi la stessa continuazione in entrambi gli slot.

- Prima: `ached film film Susie film cleached testachedached test test`
- Dopo: ` time there a was a a Ben Ben Ben Ben named Ben`

Parole del campione riconoscibili, ordine scorretto, ripetizioni. La generazione libera accumula errori: coerente con apprendimento iniziale, **non prova che sappia scrivere una storia**. Entrambi gli output e tutti gli ID salvati interi nel JSON.

## Che cosa ha imparato ad aggiornare

Delta RMS vs pesi iniziali: raw sinapsi0,00453; embedding0,02053; proiezione lettura0,00792; Q0,00985, K0,01086, V0,01642, O0,01114. Tutti i tensori registrati hanno gradienti e delta non nulli al controllo finale; tutti finiti. Non vuol dire che ogni sinapsi cambi o che l'attenzione sia indispensabile: servono task memoria e controlli dopo.

Raw sinapsi passa ancora per softplus e segni fissi: nessun cambio di identità, direzione, topologia o segno del connettoma. Gate appreso da0,1 a0,14595, senza modifica manuale. Clipping norma globale1; norma pre-clipping finale ~3,39. Stessi LR per i tre gruppi, decay0, Adam betas(0,9;0,999).

## Stabilità osservata e limiti

Nel campione ultima fase post: firing medio5,22%, silenti91,44%, firing a tutti i sottopassi2,95%. Potenziale RMS0,744, max assoluto25,82; feedback RMS0,641 contro token0,706 (~91%). Valori precedenti all'ultimo update, non medie del corpus.

Feedback e potenziali salgono durante l'apprendimento. Nessun NaN/Inf né perdita divergente, ma **la crescita va verificata su sequenze più lunghe in3.3/3.5**. Ogni epoca qui azzera lo stato dopo16 token: stabilità ricorrente lunga non dimostrata. Potenziali negativi possono superare1 in valore assoluto per l'inibizione; il solo picco non identifica un'esplosione numerica. Neuroni silenti in questo campione non per forza morti.

## Esecuzione e risorse

- GPU GTX1080, FP32, forward/backward/clipping/Adam/carry in CUDA Graph; inferenza decoder catturata. CPU solo dati, I/O, controllo; nessun offload del modello.
- B2,L8,2 finestre per epoca. Memoria e cache detached fra finestre; valori da pesi precedenti conservati. CE di valutazione senza update, su tutto il prefisso con pesi fissi.
- Passo medio0,316s per16 target, ~50,6 target/s del solo passo misurato; runtime supervisore138s inclusi init, warmup, valutazioni, checkpoint, cleanup. Non previsione di throughput su corpus/context maggiori.
- Picco tensori allocati388,12 MiB; GPU totale campionata1.771 MiB. RAM sistema fino90%, libera1,366GiB al cleanup: vicina alla guardia1,25GiB; niente aumento auto di batch/contesto. Dopo uscita worker RAM libera2,743GiB, GPU totale866MiB; include altri processi/driver.
- Allocatore PyTorch dopo cleanup:0 MiB allocati e riservati; nessun abort delle guardie. Salvataggio/ricaricamento verificato con uguaglianza di tutti i tensori.

## Artefatti e stato

- Runner: `train_diagnostic.py`; protocollo e hash: `PROTOCOLLO_FASE_3.md`, `configs/phase3_protocol_v1.json`.
- Evidenza: `results/phase32_adam_lr1e4.json`, `.worker.json`, `.jsonl`; curva machine-readable `results/phase32_learning_curve.csv`.
- Pesi: `results/phase32_adam_lr1e4.diagnostic.pt`; config, archi/segni e tokenizer identificato nei metadati. Checkpoint diagnostico senza optimizer/RNG/stato per ripresa esatta; non modello preaddestrato.
- Manifest artefatti: `results/phase31_32_evidence_manifest.json`. Hash reali per run, niente sostituzione dei risultati del tentativo precedente.

**3.1 e3.2 concluse; prossima3.3 memoria/attenzione/TBPTT.** Validation non valutata in3.2; nessun risultato test finale consultato. Muon/ROOT restano confronto3.4, non adottati per questo overfit.

Riproduzione protetta (un caso alla volta):

```powershell
$env:TEMP = (Resolve-Path '.runtime-tmp').Path
$env:TMP = $env:TEMP
.venv/Scripts/python.exe train_diagnostic.py --smoke --epochs 2 --output results/phase31_smoke.json
.venv/Scripts/python.exe train_diagnostic.py --output results/phase32_adam_lr1e4.json
```
