# Interfacce testo — fase 2.2

Codice: `fly_interfaces.py`. Calcolo: GPU FP32 + CUDA Graph. Tokenizer fisso; embedding casuale, nessun preaddestramento.

## Percorso

`ID[B] → E[4096,256] → LayerNorm → ingresso sparso → core → pooling(v,rate) → LayerNorm → proiezione256 → LayerNorm → E.T → logits[B,4096]`

- E condivisa: un parametro, lookup e uscita → stesso gradiente.
- PAD non blocca riga E: esclusione da loss/avanzamento in fase2.4.
- No somma diretta embedding→logits. Porte lettura disgiunte da porte ingresso.
- Stato zero + archi azzerati → nessun segnale token raggiunge lettura. Controllo non prova da solo utilità topologia biologica: ablazioni fase6.

## Ingresso

- Tutti nodi con superclass contenente `sensory`, stesso ordine compatto del core.
- Ogni porta: 8 componenti distinte su 256; carico globale per componente differisce max di1.
- Permutazione fissa, seme17; pesi `Normal(0,1/sqrt(8))`, trainabili.
- `I_s = bias_s + 0.6*tanh(sum_k(w_sk*e_channel(sk)))`; bias iniziale0.6, trainabile.
- Corrente iniziale [0,1.2]; bias può uscire dall'intervallo in training. Gain/bias artificiali, da calibrare in fase3.
- Zero sugli altri neuroni. Temporaneo `[B,N_sensory,8]`; corrente `[B,N]`. No adattatore denso256×N.

## Lettura

- Popolazioni: descending_neuron, descending_neuron_tbc, vnc_motor, cb_motor, vnc_efferent, cb_efferent, efferent_descending, efferent_ascending.
- Raggruppamento `superclass × rootSide`; dentro gruppo, ordine bodyId e blocchi max32.
- Blocchi NON regioni anatomiche spaziali; classificazione anatomica + suddivisione deterministica artificiale.
- Ogni blocco: due medie separate: potenziale finale e rate fase. Concatenazione `[B,2G]`.
- `LayerNorm(2G) → Linear(2G,256,bias=False)`; normalizzazione256 prima della testa condivisa.
- Nessun nodo rimosso dal core. Popolazioni lette ristrette: scelta iniziale, copertura/raggiungibilità e alternative da valutare.

## Rientro attenzione, pronto per 2.3

- `recalled[B,256] → adattatore sparso separato → gate → corrente[B,N]`.
- Stesse porte sensoriali; mappa seme18, pesi indipendenti, no bias/tanh.
- Gate scalare trainabile iniziale0.1. Vettore recuperato zero → corrente zero.
- Fase2.2 implementa solo la porta. Update2.3: Q/K/V, RoPE, cache e attenzione separati; vedere `ATTENZIONE_API.md`. Collegamento LM completato2.4: `LM_API.md`.

## API

- `anatomical_ports(body_ids, annotations, pool_size=32)` costruisce mappe da annotazioni; conserva ordine/ID del core.
- `TextInterfaces(...).cuda()` trasferisce parametri e mappe; `input_current(ids)`, `feedback_current(recalled)`, `representation(state,rate)`, `logits(state,rate)` sono operazioni tensoriali catturabili.
- `describe()` esporta configurazione, conteggi, hash mappe; chiamare fuori dalla cattura.
- `state_dict` comprende mappe e parametri; ricostruire stessa configurazione prima del caricamento.
- Impostazioni speciali futuro LM: reset/PAD/TBPTT e split pre/post → fasi2.3–2.5.

## Esecuzione richiesta

- Modello, loss, gradienti, Adam: GPU. No offload CPU, no fallback silenzioso.
- CPU: file, annotazioni, inizializzazione, controllo processo. Eager GPU solo riferimento correttezza e warmup.
- CUDA Graph: buffer statici, contenuti aggiornabili; cattura forward+backward+Adam (`capturable=True`, `foreach=False`).
- Eliminare riferimenti ai grafi autograd del riferimento prima della cattura; ripristinare parametri e stati Adam dopo warmup/cattura.
- Verificare più replay con input/target/feedback diversi: loss, parametri, gradienti, momenti e step Adam.
- Supervisore RAM/VRAM, timeout e processo posseduto. Cleanup include workspace cuBLAS della build locale; nessuna modifica a PyTorch.

Riferimento utente: [PR llama.cpp #27721](https://github.com/ggml-org/llama.cpp/pull/27721). Abilita CUDA Graph per Pascal abbassando il gate architetturale; patch specifica llama.cpp. PyTorch locale supporta già la cattura: no trasferimento automatico patch o speedup.

## Riprodurre le verifiche

```powershell
$env:TEMP = (Resolve-Path '.runtime-tmp').Path
$env:TMP = $env:TEMP
.venv\Scripts\python.exe verify_phase22.py --toy --output results/phase22_toy.json
.venv\Scripts\python.exe verify_phase22.py --output results/phase22_cns_t10.json
```

Eseguire in sequenza. `test_interfaces.py` contiene test numerici GPU e preparazione mappe su host; avviare tramite supervisore. Smoke sintetico di3 update; non benchmark LM, pretraining o misura qualità linguistica.
