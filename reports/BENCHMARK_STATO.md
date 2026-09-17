# Benchmark — stato operativo

Piano storico `PIANO_OPERA_FACTCHECKED.md` non vigente. Audit: `VERIFICA_PASSAGGIO_0.md`. Benchmark originale in `bench_throughput.pre_step1.py`.

## Vincoli

- CNS completo: insieme fisso 166700 body ID con `superclass`; no ritaglio anatomico.
- Soglie 10 e 5: solo archi cambiano. ID ordinati identici, inclusi nodi isolati.
- Quantità testo: da scegliere dopo misure.

## Codice

- `fly_graph.py`: lettura Feather per record batch. Cache NPZ con percorsi, dimensioni, mtime, versione schema; no nuovo download. Verifiche provenienza servono checksum ufficiali.
- `fly_core.py`: propagazione a blocchi, magnitudine softplus, segno presinaptico fisso. LIF custom conserva un potenziale; backward = reset differenziabile riferimento. Gradienti primo ordine; spike presinaptici binari.
- `test_fly_core.py`: riferimento forward/backward, reset vicino soglia, checkpoint, conservazione stato, segni e ID stabili, storage autograd.
- Fase2.1: `Core.advance` separa sottopassi/stato/rate da loss; `window` resta wrapper sintetico compatibile. `test_core_api.py`: composizione pre/post, riferimento indipendente, reset slot, freeze, detach. Contratto: `CORE_API.md`.
- `bench_runtime.py`: seed/input/stato iniziale uguali, warmup, Adam capturable, azzeramento gradienti catturato. Confronto multi-step di loss, attività, parametri, momenti Adam. Memoria/tempi JSON, log JSONL. Supervisore avvia Python base con `-S` e soli package venv: un worker diretto, no figli, arrestabile via handle posseduto. `taskkill` negato dal sistema: non più usato.

## Cosa misura

**Solo nucleo sintetico attivo.** Corrente esterna nei nodi con etichetta contenente `sensory`, per T sottopassi. Valori uniformi 0.4–1.6; magnitudini iniziali proporzionali a log(1+sinapsi), normalizzate per somma entrante, moltiplicate per `--gain`. Gain 4 ha dato gradienti NaN su L=64: scartato, log conservato. Gain 0.5 è config successiva di verifica. Valori per diagnosi, non parametri biologici calibrati.

Loss sintetica: errore quadratico del rate vs 0.15. No cross-entropy linguistica, embedding, readout. Finestre benchmark partono da stato zero per confrontare eager e graph; il core supporta stato esterno. Ogni metrica `synthetic_positions_per_sec` conta B×L posizioni sintetiche. Non equivale a token/s di un language model.

Segni da NT: semplificazione esplicita dell'audit. GABA/glutammato/istamina negativi; altri positivi, inclusi modulatori e sconosciuti per questo microbenchmark. Non nuove certezze biologiche.

## Esecuzione Windows

```powershell
.\.venv\Scripts\python.exe -u test_fly_core.py
.\.venv\Scripts\python.exe -u bench_throughput.py --toy --output results/toy_cuda.json
.\.venv\Scripts\python.exe -u bench_throughput.py --soglia 10 --output results/cns_t10_b1_l4.json
.\.venv\Scripts\python.exe -u bench_throughput.py --soglia 10 --batch 8 --tronca 64 --gain 0.5 --output results/cns_t10_b8_l64_gain05.json
```

Default diagnostici: B=1, L=4, T=8, gain=0.5, tre aggiornamenti misurati dopo warmup. Anatomia completa. B e L si aumentano solo dopo misura breve. Config lunga: `--batch 8 --tronca 64`; checkpoint opzionale: `--checkpoint-chars 4`. Prime prove archiviate senza parametro `gain` nel JSON usavano gain=4, come da loro note.

Guardie: RAM libera iniziale ≥1.75 GiB; stop sotto 1.25 GiB; GPU totale osservata ≤7000 MiB; allocatore PyTorch limitato al 75% GPU. Timeout totale 600 s, silenzio di fase 90 s. Controllo risorse circa ogni 2 s: polling non garantisce assenza picchi fra campioni. OOM allocatore e stallo terminano il worker; il processo principale registra l'esito. Nessun benchmark da lanciare col flag interno `--worker` direttamente.

## Interpretazione

Prima correttezza, poi tempi. Graph fallito/divergente non è risultato di velocità valido. Atomiche CUDA possono causare differenze numeriche: il confronto applica tolleranze esplicite e le registra nel codice. Nessun limite inferiore promesso al guadagno CUDA graph. Nessuna stima di epoche TinyStories da questo test.
