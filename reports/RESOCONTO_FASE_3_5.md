# Fase3.5 — validation e stabilità

**Criterio numerico superato; generazione greedy degenerata. Fase3 complessiva aperta.**

| Misura | Prima | Dopo |
|---|---:|---:|
| CE validation16 storie / 2048 target | 8,375530 | 5,918187 |
| Accuracy next-token | 0% | 7,324% |
| CE ultime12 storie / 1536 target | 8,372318 | 5,910470 |

- CE: −29,34%, oltre soglia5%. Ultime12 storie fuori selezione optimizer: miglioramento simile.
- Train:64 storie nuove rispetto3.2/3.4, primi128 target/storia;8126 target effettivi,512 update,179,05s. Adam LR0,0001, seed17, B2/L8, clip1, decay0. Un solo passaggio; diagnostica breve.
- GPU FP32/CUDA Graph; valutazione catturata verificata vs riferimento GPU eager. Nessuna modifica architetturale.
- Nessun NaN/Inf; aggiornamenti finiti core/interfacce/attenzione. RMS variazione pesi:0,001601 /0,019140 /0,005217.
- RMS potenziali nei16 log:0,4223–0,4292; allarme drift assente. Gate finale0,12245. Non prova stabilità pretraining lungo.
- Picco allocator CUDA378,81MiB; non coincide con memoria totale driver/processo/GPU. Cleanup finale:allocated0, reserved0. Generazione:cleanup allocated0.
- Paging autorizzato, guardie attive, nessun abort nel run valido. Pagefile configurato ≠ traffico paging misurato.

## Generazione: fallimento qualitativo

Due prompt validation fissi, fuori selezione optimizer;24 token greedy prima/dopo:

| Prompt | Dopo training |
|---|---|
| There was a boy named Tim. | `........................` |
| Once upon a time, there was | `........................` |

Prima: frammenti incoerenti (`celebr`, `lcano`, ecc.). Dopo:24 punti in entrambi i casi. Decoding eseguito correttamente, testo inutilizzabile. Nessuna ricerca prompt/semi favorevoli.

CE teacher forcing migliore = probabilità target reali aumentata. Non garantisce buona scelta argmax né stabilità autoregressiva quando modello rilegge propri output. Qui: miglioramento statistico, collasso qualitativo osservato. Causa non isolata.

## Artefatti

- [Run e guardie](results/phase35_adam_s17.json), [metriche](results/phase35_adam_s17.worker.json).
- [Generazioni complete prima/dopo](results/phase35_generation.json).
- [Checkpoint diagnostico](results/phase35_adam_s17.diagnostic.pt): pesi/config, non ripresa completa optimizer/RNG/carry.
- [Protocollo prefissato](PROTOCOLLO_FASE_3_B.md). Test finale mai consultato.