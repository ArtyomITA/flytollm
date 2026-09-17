# LM integrato — fase2.4

Scelte approvate: **4 sottopassi pre +4 post; corrente post = token + feedback**. Moduli e parametri invariati2.1–2.3; totale CNS soglia10:4.410.893.

## Percorso esatto

1. ID[B]; PAD inattivo; BOS o reset esplicito azzera core e cache per slot.
2. Embedding normalizzato → corrente sensoriale.
3. Core4 sottopassi → lettura potenziale/rate fase pre → rappresentazione256.
4. Attenzione legge cache stati finali precedenti → adattatore feedback e gate.
5. Core4 sottopassi corrente token + feedback → lettura potenziale/rate fase post → rappresentazione256.
6. Rappresentazione finale × E.T → logits. Stessa rappresentazione → K/V → append cache.

Stato core continua fra token. Corrente token calcolata una volta per token, riutilizzata post. Magnitudini core condivise entro un forward, ricalcolate dopo ogni update.

## API funzionale, autograd

- `FlyLM.step(ids[B], state=None, active=None, reset=None)` → logits[B,V], LMState.
- `FlyLM.forward(ids[L,B], ...)` → logits[L,B,V], stato finale. Loop causale, nessuna corrente permanente[L,B,N].
- `LMState(voltage,spike,cache)`: nessuna mutazione input. Ogni stato appartiene a una storia per slot.
- `loss(logits,targets,ids,active)` → CE media su token validi e conteggio. Esclude PAD input/target e slot inattivi; batch tutto mascherato →loss0. Chiamante deve saltare optimizer.step senza target validi: Adam può aggiornare da momenti precedenti anche con gradiente zero.
- PAD conserva core/cache e restituisce logits0. Reset esplicito vale anche su slot inattivi.
- `lm_io.story_batch`: finestre in singole storie; input[t]→target[t+1], EOS previsto, nessun target oltre fine storia. CPU solo lettura/staging; tensori modello su GPU.

## TBPTT

- Finestra diagnostica L4. Nessun detach fra token della stessa finestra.
- Dopo backward: `next_state=model.detach(final_state)` prima del forward successivo. Taglia gradiente core e K/V, conserva valori. Mai `retain_graph=True` fra update optimizer.
- Stati conservati dopo update rappresentano pesi precedenti: approssimazione TBPTT esplicita. Smoke reale usa tre finestre indipendenti con BOS; non valida ancora training prolungato con carry dopo update.
- Cache128 = memoria consultabile; non estende automaticamente backprop a128.
- Nessun checkpoint attivazioni nella baseline2.4; confronto segmenti puri previsto2.5/4.

## CUDA Graph e generazione

- `verify_phase24.py`: cattura forward completo + CE + backward + Adam FP32, `capturable=True`, `foreach=False`, decay0. Gruppi core/interfacce+attenzione distinti, LR diagnostico1e-4.
- `CUDATokenStep(model,batch,temperature)` cattura inferenza+selezione token con stato GPU persistente. Pesi fissi durante vita decoder; ricrearlo dopo training/caricamento/modifica parametri.
- `replay`: input e maschere copiati in buffer statici; restituisce logits e token in buffer riusati. Clonare se devono sopravvivere al replay successivo.
- Temperatura0: greedy. Temperatura>0: softmax e campionamento categorico via CDF, uniforme generata su GPU o fornita per test riproducibile.
- PAD/BOS esclusi da generazione; EOS termina continuazione slot. Filtro riguarda decoding, non modifica CE sul vocabolario completo.
- `generate(prompt[L,B],max_new_tokens)`: prompt già tokenizzato, preferibilmente BOS iniziale e padding in coda; prefill e generazione con lo stesso step catturato. Restituisce continuazione a lunghezza fissa: EOS poi PAD; slot finiti congelati.
- Loop host orchestra replay; nessun forward/backward CPU. Nessun fallback. `close()` rilascia grafo; eliminare decoder e output trattenuti per rilasciare anche tensori.

## Checkpoint e limiti

`lm_io.save_model/load_model`: configurazioni, stato completo moduli inclusi indici e segni, chunk core e metadati. Ricostruzione verificata su GPU; embedding resta condiviso con uscita.

File `.smoke.pt`: tre update diagnostici, **non pretraining**. Non contiene optimizer/RNG/stato ricorrente per ripresa esatta del training; checkpoint riprendibili completi previsti fase5. Tokenizer esterno identificato via hash nei metadati del checkpoint CNS.

```powershell
$env:TEMP = (Resolve-Path '.runtime-tmp').Path
$env:TMP = $env:TEMP
.venv\Scripts\python.exe verify_phase24.py --toy
.venv\Scripts\python.exe verify_phase24.py --output results/phase24_cns_t10.json
```

Avviare in sequenza sotto supervisore. Risultati diagnostici non misurano qualità linguistica o throughput sostenuto del pretraining.

## Verifica2.5

Carry TBPTT verificato per6 update consecutivi B2×L4 su CNS: `verify_phase25.py --case carry_training --output results/phase25_cns_carry.json`. Copia stato detached dopo backward/update, valori del passato conservati; stato prodotto dai pesi precedenti dichiarato. Reset BOS elimina anche gradienti verso lo stato precedente.

`lm_checkpointing.checkpoint_forward(model,ids,state=None,active=None,reset=None,segment=2)` restituisce logits/stato come forward. Checkpoint non reentrant su segmenti funzionali; segmenti2/4/8 verificati per valori e gradienti su GPU eager B2×L8. Opzionale, non attivo nel training di riferimento, capture CUDA Graph del checkpoint da valutare in fase4.

Risorse, limiti e telemetria: [RESOCONTO_FASE_2_5.md](RESOCONTO_FASE_2_5.md).
