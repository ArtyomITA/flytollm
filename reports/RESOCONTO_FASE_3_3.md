# Fase3.3 — memoria e TBPTT

**Prove fatte; criterio apprendimento NON superato.** Nessuna modifica architettura. Protocollo in `PROTOCOLLO_FASE_3_B.md`.

## Stress ricorrente

Checkpoint overfit3.2,256 token payload train0 ripetuto, BOS solo iniziale, nessun reset intermedio. Cache satura a128; posizioni a256. Nessun NaN/Inf. RMS potenziale 0,659–0,740 nei campioni32..256; max assoluto 24,18–27,73. Nessuna crescita continua in questo stress inferenza. Non prova testo naturale lungo né stabilità training lungo.

Evidenza: `results/phase33_long_attempt2.json`. Primo avvio `phase33_long.json` interrotto da guardia RAM prima del caricamento.

## Copia ritardata, da inizializzazione nuova

Otto simboli ordinari, 32 simboli/stream, train8 stream/dev8 stream disgiunti. Target posizione corrente = simbolo visto d passi prima. Loss vocabolario4096, accuracy primaria su8 simboli; a fine training accuracy completa = ristretta. Test sintetico seed173 non usato.

| Ritardo | Accuracy iniziale dev | Accuracy finale dev | Target dev | Update | Criterio80% |
|---:|---:|---:|---:|---:|---|
| 1 | 12,90% | **32,66% (81/248)** | 248 | 200 | Non superato |
| 4 | 12,50% | **9,82% (22/224)** | 224 | 200 | Non superato |
| 8 | 11,98% | **9,375% (18/192)** | 192 | 160 | Non superato |

Chance nominale12,5%. Ritardo1 migliora sul dev; ritardi4/8 restano vicini al caso nel campione piccolo. Non dedurre incapacità teorica o superiorità/inferiorità statistica dal solo scarto vs12,5%.

Adam LR1e-3, 10 epoche, B2/L8, clipping1. Ritardo8 ha160 update anziché200: prime finestre senza target validi. Consumate in inferenza per formare contesto; optimizer non aggiornato. Non eliminate dalla memoria. Pesi overfit3.2 non riusati per questi tre task.

## Interpretazione

- Cache128 = capacità di conservare valori; **TBPTT8 limita percorso gradienti**, non lunghezza valori conservati. Componente causa limite non misurata.
- Stato ricorrente mosca + attenzione operano insieme: miglioramento ritardo1 non attribuisce merito ad attenzione sola. Ablazioni ancora da discutere.
- Fattori da testare in revisione futura: finestra gradiente, durata apprendimento, scala correnti, informazione al readout. Ipotesi, non diagnosi provate.
- Telemetria di alcune righe training campiona ultimo token PAD: rate0/silenti100% in quelle righe deriva da maschera, non da neuroni morti. Potenziali restano quelli conservati. Non usare queste righe per dichiarare collasso attività.

## Risorse e correttezza

Valutazione un token alla volta vs riferimento GPU eager su finestra completa: stato/loss coerenti. Nessun calcolo modello su CPU. Tutti i run validi con cleanup allocatore0 MiB.

Tentativi ritardo1 iniziali interrotti da soglie fisiche conservati (`phase33_memory_d1.json`, `_attempt2.json`). Run valido `_attempt3.json` con profilo paging autorizzato; altri risultati `phase33_memory_d4.json`, `phase33_memory_d8.json`. Score parziali abortiti non usati. Profilo e rettifiche soglie in `PROFILO_PAGING_FASE3.md`.

**Stato:** 3.3 resta da risolvere sul piano apprendimento. Proseguite3.4/3.5 come diagnosi richieste dall'utente; non chiudere tutta la fase3 sulla sola esecuzione test.