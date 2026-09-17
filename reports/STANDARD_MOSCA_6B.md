# Configurazione standard "Mosca 6b" — fissata dall'utente il 16 settembre 2026

Configurazione di riferimento per i prossimi test, scelta sui risultati della suite 6b e delle promozioni a 8000 (`SUITE_TEST_FASE_6B.md`, `RESOCONTO_FASE_6_CONTROLLI.md` §6c). Il modello principale (`fly_core.py`, checkpoint a 26.247 update) resta invariato: questo è lo standard per i run di controllo e per le fasi successive, non una modifica retroattiva.

## Scelte e prove che le sostengono

| Componente | Scelta standard | Prova |
|---|---|---|
| Grafo | male-cns v1.0 intero, 166.700 nodi, 2.753.975 archi | vincolo utente: mai pruning |
| Selezione degli archi | soglia sinaptica relativa a pari archi (`--rewire-kind relthr --rewire-seed 41`): un arco entra se pesa almeno l'1,31% dell'input totale del bersaglio, stesso numero di archi della soglia 10 | D8: 4,961 a 2000 (+0,13), 3,903 a 8000 (+0,06), sopra il reale lungo tutta la curva |
| Segni | dal neurotrasmettitore di consenso: gaba / glutammato / istamina = −1, tutto il resto = +1 (64,4% eccitatori) | D9: il 6% di segni sbagliati non costa (3,897 a 8000); B4: permutarli aiuta, quindi non si tocca l'anatomia ma non serve nemmeno correggerla |
| Pesi | conteggio sinapsi normalizzato come init, allenabili | B3: permutarli non cambia nulla (5,103) |
| Ingresso del testo | 17.937 porte scelte a caso tra i nodi non di lettura, seed 17 (`--ports random_matched`), fan-in 8 | B7a: 4,639 a 2000, 3,689 a 8000 (+0,27 sul reale; recupera l'87% del divario col tetto 3,648) |
| Lettura | 2.241 discendenti / motori / efferenti a blocchi di 32 → 77 gruppi → 256 → testa H1 separata (`--readout chunks`) | D3 (gruppi anatomici) non regge a 8000 (+0,015); F2/F4/F5 (fru, hub, CX) peggiori |
| Dinamica | LIF, leak 0,95, soglia 1, T = 4 + 4 sottopassi, TBPTT 8 (`--core-variant lif`) | D4/D5/D6 neutre; D7 (potenziali di inversione) +0,04 a 2000 con costo ×1,95: opzione, non standard |
| Attention | MHA 4×64 esterna, KV cache 128, feedback nelle porte, invariata | A1: vale 0,24 nat a 8000 |
| Core di calcolo | kernel fuso cupy (`--fast-mode fused`, `fly_core_fast.FusedCore`), ordine dei nodi originale | S6/S8b: equivalente al core PyTorch a 100 e 2000 update, ×1,73 |
| Ottimizzazione | Adam 1e-4, batch 2, seed 17, CUDA Graph FP32 | pipeline del pilota |

## Comando

```bash
./.venv/Scripts/python.exe pretrain_control.py --threshold 10 --head separate --seed 17 --eval-every 500 --checkpoint-every 2000 --fast-mode fused --rewire-kind relthr --rewire-seed 41 --ports random_matched --readout chunks --core-variant lif --updates 8000 --output results/phase7_standard_8000.json --allow-paging --timeout 14400
```

Opzione D7 (sinapsi a conduttanza): aggiungere `--core-variant reversal` (costo ×1,95 in tempo, VRAM di picco 2,9 GB).

## Stato di validazione

La combinazione porte random + soglia relativa NON è mai stata eseguita insieme: i due pezzi sono provati uno per uno. Attesa a 8000: 3,65–3,69 (tetto della pipeline 3,648, configuration model). I due freni grandi del cablaggio reale (ingresso e inibizione) non sono additivi. Primo passo quando l'utente dà il via: un run a 8000 col comando sopra, confronto con reale 3,959, porte random 3,689, config model 3,648.

## Cosa resta fuori dallo standard e perché

- Porte anatomiche (sensoriali veri): costano 0,27 nat a 8000. Da riprendere solo per fedeltà, con le vie che hanno indizi positivi: udito/tatto (F6, 5,182 a 2000) e occhi → fru/dsx (F3, 5,127 a 2000), entrambe da confermare a 8000.
- Riordino dei nodi: non equivalente (gruppi e porte definiti sull'indice); da rifare con permutazione dei buffer.
- Costanti di tempo per tipo (D6): richiede learning rate dedicato, non provato.
- Segni permutati (B4, 3,723): stesso guadagno delle porte random, ma butta via un dato biologico misurato; le porte random cambiano solo l'interfaccia artificiale.
