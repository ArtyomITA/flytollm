# API nucleo — fase2.1

`fly_core.Core`: parametri/topologia invariati; API funzionale per futuro LM.

```python
state = core.initial_state(batch)                 # (v, spike), [B,N], zeri
w = core.weights()                                # riutilizzo entro stesso forward
state, rate_pre = core.advance(current, 4, state, weights=w,
                               reset=new_story, active=valid_slot)
# Fase2.3 futura: leggere stato → attenzione → corrente recuperata.
state, rate_post = core.advance(current + recalled_current, 4, state,
                                weights=w, active=valid_slot)
# Fase2.2 futura: leggere (v, spike, rate) → rappresentazione compatta.
state = core.detach_state(state)                   # SOLO confine TBPTT scelto
```

- `current`: `[B,N]`, dtype/device core, costante nei sottopassi chiamata.
- `steps`: intero positivo. Tpre+Tpost=T; rate globale = media pesata rate fase.
- Stato: `(v,s)`; s binario; input finiti. `initial_state` alloca tensori nuovi.
- `reset`: bool[B]; azzera stato pre-fase, taglia gradiente stato precedente per quegli slot.
- `active`: bool[B]; slot False conserva stato dopo reset, dà rate0, nessun gradiente verso corrente slot.
- `advance`: dà nuovo stato + rate[B,N], tiene autograd; nessuna loss/attenzione/cache interna.
- `reset_state`: funzionale. `detach_state`: conserva valori, condivide storage, vietate modifiche in-place dopo.
- `weights`: opzionali; se passati derivano da core corrente e tengono gradiente verso raw. Mai riusare dopo optimizer.step.
- `reference=True`: gather completo solo test piccoli; mai CNS batch×archi integrale.
- `window`: wrapper storico loss sintetica; ora usa `advance`, conserva contratto e checkpoint.
- Letture compresse/anatomiche, embedding, attenzione **non incluse** in 2.1.

Verifiche: `test_fly_core.py` (6 regressioni) + `test_core_api.py` (4 test contratto/equazioni/gradienti). Smoke CUDA via supervisore esistente, nessun test GPU non protetto.
