# AdamW, Muon, ROOT — curiosità verificata

**Tutti eseguibili nel nostro LM su GTX1080, FP32, training dentro CUDA Graph.** Sei update ciascuno, stessi pesi iniziali, stesse48 posizioni TinyStories (B2×L4,6 finestre con carry). Smoke compatibilità, non confronto capacità linguistiche.

| Variante diagnostica | Optimizer GPU medio | Passo completo medio | Picco allocato |
|---|---:|---:|---:|
| AdamW | 2,06 ms | 171,9 ms | 387,85 MiB |
| Muon FP32 + AdamW | 3,35 ms | 172,7 ms | 385,54 MiB |
| ROOT fallback + AdamW | 5,71 ms | 175,1 ms | 385,54 MiB |

Tempi: media6 replay, run distinti, nessuna stima speedup. Passo completo = forward/backward + copia stato. Picchi includono dispositivo verifica e riferimenti: non sono memoria solo optimizer. Risparmio piccolo: quasi tutti i parametri restano AdamW.

**Verifiche:** perdita/stato ogni finestra, parametri e momenti finali = GPU eager; max errore parametri4,77e-7 ciascuno. Gradienti finiti, cleanup allocatore0 MiB. ROOT prima su toy; percentile verificato vs `torch.quantile`, componente robusta vs clamp.

## Che cosa cambia matematicamente

AdamW: medie mobili gradiente e quadrato; normalizza ogni coordinata. Muon: tratta update matrice come matrice — momentum, Nesterov, cinque iterazioni Newton–Schulz per modificare valori singolari. Qui variante Moonlight, FP32 invece del BF16 originale. [Codice Moonlight](https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py).

ROOT: filtro prima dell'ortogonalizzazione — soglia al90° percentile di |momentum Nesterov|; sottrae componente eccedente. Coefficienti per forma sono l'altra proposta del paper. **Codice ufficiale non li fornisce per256×256 e256×154:** usa coefficienti Muon standard. Prova eseguita con quel fallback; non riproduce beneficio calibrazione paper. [Paper ROOT](https://arxiv.org/abs/2511.20626), [implementazione degli autori](https://github.com/huawei-noah/noah-research/blob/master/ROOT/examples/example_train.py).

Solo5 matrici dense: Q/K/V/O e proiezione lettura, **301.568 parametri,6,84% del totale**. Restanti4.109.325 con AdamW: embedding condiviso, sinapsi, porte sparse, bias/norm/gate. Porte = liste connessioni con indici diversi per neurone: storage2D non le rende proiezione densa per Muon. Non abbiamo rimodellato sinapsi in matrici arbitrarie.

## Scelte esatte e limiti

- LR base1e-4, decay0, AdamW betas(0,9;0,95), eps1e-8 in tutti i rami AdamW. Muon/ROOT momentum0,95, Nesterov,5 iterazioni, scala LR0,2√max(shape): per queste matrici LR effettivo3,2e-4. LR non ottimizzati.
- Baseline precedente resta Adam betas(0,9;0,999), decay0. A decay0 Adam/AdamW hanno stesso principio update; cambiare beta2 è scelta distinta, qui esplicitata.
- Implementazione locale diagnostica: rank percentile statici, valori CUDA; contatore AdamW catturabile. Ramo ausiliario usa AdamW PyTorch standard, non la diversa collocazione epsilon dell'esempio autori. Nessun `torch.compile`, installazione o modifica librerie.
- Coefficienti ROOT standard(3,4445;−4,775;2,0315), non calibrati per nostre forme. `fallback_count=0` = nessun fallback **di esecuzione su CPU**; fallback coefficienti invece presente e dichiarato a parte.
- Loss finale: AdamW8,2011; Muon8,2324; ROOT8,2325. **Non classificano gli ottimizzatori:** sei batch diversi lungo il run,48 token, un seme, nessun tuning LR, nessuna validation. Muon/ROOT quasi coincidenti in questo probe.

**Decisione:** nessuna sostituzione baseline. Fase3: eventuale confronto con overfit/validation, stesso budget, LR adeguati, più semi. Fase4: misurare se ottimizzazioni percentile/ortogonalizzazione servano davvero. Calibrare coefficienti ROOT richiede esperimento separato; non copiare quelli di matrici2048 su matrici256.

Codice: `optimizer_variants.py`, `verify_optimizers.py`. Risultati: `results/phase25_optimizer_{adamw,muon,root}.worker.json` e supervisori/log. Riproduzione sequenziale sotto watchdog:

```powershell
$env:TEMP = (Resolve-Path '.runtime-tmp').Path
$env:TMP = $env:TEMP
.venv/Scripts/python.exe verify_optimizers.py --variant adamw --output results/phase25_optimizer_adamw.json
.venv/Scripts/python.exe verify_optimizers.py --variant muon --output results/phase25_optimizer_muon.json
.venv/Scripts/python.exe verify_optimizers.py --variant root --output results/phase25_optimizer_root.json
```
