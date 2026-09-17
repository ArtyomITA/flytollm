# Risultati — 12 settembre 2026

## Verdetto

Nucleo corretto gira su GTX 1080 con CNS intero. Test sintetico; qualità linguistica non ancora dimostrata. Ogni run tiene 166700 nodi. Test CPU: 6 passati.

## Correzioni

- LIF: 1 potenziale salvato; derivata reset invariata vs riferimento. Test anche vicino soglia.
- Propagazione: temporanei per blocco; magnitudini softplus, segni fissi per sorgente.
- Caricatore: ID fissi, nodi isolati tenuti, cache per soglia.
- Graph: gradienti azzerati nel grafo; confronto dopo tre aggiornamenti da stato identico, momenti Adam inclusi.
- Supervisore: RAM/VRAM, timeout, log immediati, arresta solo albero avviato, memoria dopo uscita.

Originale conservato: `bench_throughput.pre_step1.py`. Piano storico marcato non vigente. Istruzioni: `BENCHMARK_STATO.md`.

## Misure

GTX 1080, torch 2.14.0+cu126. T=8. Tempi mediani 3 aggiornamenti dopo warmup. Posizioni = B×L, sintetiche. No embedding/readout/tokenizzatore/CE; no estrapolazione a epoche.

| Soglia | B | L | Gain | Eager s | Graph s | Incremento throughput graph |
|---|---:|---:|---:|---:|---:|---:|
| 10 | 1 | 4 | 4 | 0.237 | 0.084 | +183% |
| 10 | 8 | 8 | 4 | 1.076 | 1.008 | +6.8% |
| 10 | 8 | 64 | 4 | — | — | scartato: gradienti NaN |
| 10 | 8 | 64 | 0.5 | 8.647 | 8.310 | +4.1% |
| 5 | 8 | 64 | 0.5 | 17.929 | 17.074 | +5.0% |

Gain 4 instabile su finestra lunga: rientrare in memoria non basta. Gain 0.5 a soglia 10: attività ~8.05%, gradienti finiti/non nulli, loss sintetica 0.061884→0.061870. Prova aggiornamenti funzionanti su tre passi, non convergenza a lungo termine.

Soglia 10, B8/L64/gain0.5: **61.6 posizioni sintetiche/s** con graph. Differenza massima parametri eager/graph: 4.77e-7; primo momento Adam: 4.55e-13. Picco allocatore eager 4241 MiB; picco GPU totale campionato dal supervisore 5742 MiB (5.61 GiB, desktop/runtime inclusi). Allocatore dopo cleanup: 0 MiB allocati e riservati; memoria fisica riverificata dopo uscita.

Soglia 5, stessa B8/L64/T8/gain0.5: **30.0 posizioni sintetiche/s**. Parametri eager/graph: differenza massima 9.54e-7; primo momento Adam: 5.68e-14. Attività ~8.28%, gradienti finiti, loss 0.063799→0.063779. Picco GPU totale campionato **6414 MiB (6.26 GiB)**; RAM sistema max 78%, processo ~2703 MiB. Cleanup allocatore 0/0 MiB; dopo uscita GPU 666 MiB, RAM libera 5.52 GiB.

**Nota sui primi JSON:** il campo `graph.peak_vram_mb` nei run senza `source_sha256` è stato azzerato dopo cattura e sottostima la memoria del graph. Per quei run usare `peak_gpu_used_mb` del supervisore; il codice finale conserva il picco dalla cattura e registra anche `peak_reserved_mb`. Tempi ed equivalenza non toccati da questo errore di telemetria.

## Limiti

- Esperimento di throughput, non allenamento su TinyStories.
- Ingresso sintetico tenuto per sottopasso; gain e target rate sono parametri diagnostici.
- Normalizzazione entrante cambia con soglia: nessun confronto qualità 10→5 da queste loss.
- Tre campioni, eager prima di graph: nessuna pretesa di caratterizzare rumore termico o guadagni piccoli con precisione statistica. +4.1% è l'osservazione di questo run.
- Picco fisico campionato ogni ~2 s; può perdere transitori. Contatori CUDA e limite allocatore completano il controllo.
- Checkpoint verificato su CPU; supporto cattura CUDA da verificare separatamente prima dell'uso reale.
- Cache usa size/mtime, non checksum remoto; provenienza ufficiale dei dati resta separata da correttezza numerica del benchmark.

Log/metriche: directory `results/`. Nessun checkpoint di language model creato.

## Arresto e verifica finale

Primo test timeout ha scoperto che `taskkill` restituisce `Access denied` in questo ambiente. Supervisore corretto: Python base avviato direttamente con `-S`, percorsi package venv anteposti, nessun processo figlio del worker. Arresto tramite handle del processo, senza ricerca PID.

`guard_timeout_final.json`: timeout intenzionale, worker terminato con exit 1, nessun risultato di training completato, memoria GPU identica prima/dopo (672 MiB). Durata totale supervisore 4.375 s inclusa pausa di verifica finale; il timeout è soggetto al polling, non un timer hard real-time.

`toy_direct_worker.json`: ultimo codice eseguito via worker diretto, CUDA attiva, tre aggiornamenti eager/graph. Verifica esplicita anche dei gradienti finali; hash sorgenti nei risultati. Copre le ultime modifiche a supervisore e telemetria senza ripetere inutilmente i benchmark lunghi.
