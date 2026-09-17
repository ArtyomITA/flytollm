# Attenzione e memoria — fase2.3

Implementazione: `fly_attention.py`. Solo GPU FP32 percorso verificato; CUDA Graph training+inferenza. Modulo verificato separatamente in2.3; ora collegato in LM2.4: `LM_API.md`.

## Configurazione

- MHA softmax: d256,4 teste×64, finestra128, dropout0.
- LayerNorm256 condivisa query/memoria; Q/K/V/O256×256 no bias.
- **262.656 parametri aggiunti**. Totale core soglia10+interfacce+attenzione:4.410.893; no nuovo neurone/arco biologico.
- RoPE base10.000, coppie componenti adiacenti; su Q e K, no V.
- Posizioni assolute int64 storia; FP32 per trigonometria. No modulo128: eviction non riavvia posizioni. Contesti enormi non validati.
- Init deterministica seme23; tokenizer/embedding fase2.2 invariati.

## Ordine obbligatorio per il chiamante

```python
cache = attention.reset(cache, reset_mask)
# Core: fase pre -> rappresentazione provvisoria [B,256].
recalled = attention.read(provisional, cache, active)
# Interfaccia feedback -> core fase post -> rappresentazione finale [B,256].
cache = attention.append(final, cache, active)
```

Read separata da append: query token corrente, K/V solo stati finali precedenti. Classe gestisce attenzione; chiamante rispetta ordine pre/post. No scorciatoia output attenzione→logits.

## Cache funzionale

`AttentionCache(k,v,positions,valid,next_position)`:

| Campo | Forma | Contenuto |
|---|---|---|
| k,v | B×4×128×64 | K già ruotate; V non ruotate |
| positions | B×128 | posizione assoluta di ogni slot |
| valid | B×128 | slot realmente scritto |
| next_position | B | posizione del token corrente |

- `empty(B)`: cache zero su device/dtype modulo. KV:256KiB/esempio FP32; metadati1.160byte/esempio, attivazioni/autograd escluse.
- `append`: nuovi tensori, scorre di uno, aggiunge ultimo K/V, espelle più vecchio. Forme fisse; no update in-place cache training.
- Slot inattivo: output zero, cache/posizioni invariate. No PAD, no avanzamento posizione.
- `reset`: azzera solo slot richiesti, inclusi inattivi; interrompe gradiente verso vecchia storia.
- `detach`: confine TBPTT esplicito; conserva valori e storage, interrompe autograd. Non mutare storage condiviso finché serve a backward.
- K/V mantengono gradienti dentro finestra backprop. Memoria128 non implica backprop128.
- Update optimizer rende vecchia cache costruita con pesi precedenti: politica TBPTT da esplicitare in2.4; no riuso grafo autograd attraverso update.

## Maschere e backend sostituibile

- Maschera: slot valido, posizione<corrente, esempio attivo. Bool True=leggibile.
- SDPA matematico forzato via `sdpa_kernel(SDPBackend.MATH)`; `is_causal=False`, maschera già contiene passato per query singola.
- Memoria vuota: slot fittizio zero permesso per evitare softmax tutta mascherata; uscita finale forzata zero. No contributo spurio al primo token.
- K/V invalidi azzerati prima di SDPA; anche NaN in slot invalidi non contaminano output.
- Backend separato `MathSDPA.forward(q,k,v,mask)`. Sostituzione numericamente equivalente riusa pesi/cache; diversa matematica/teste richiede nuova verifica.

Semantica verificata su [SDPA PyTorch2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) e [selezione backend](https://docs.pytorch.org/docs/2.14/generated/torch.nn.attention.sdpa_kernel.html).

## CUDA Graph

- Training: finestra statica4 rappresentazioni, batch2; reset/active restano tensori modificabili fra replay. Cattura include forward, backward, Adam.
- Inferenza: cache persistente, aggiornata in-place dal solo harness sotto `no_grad` dopo read/append funzionali. Vietato riusare nel training con autograd vivo.
- `verify_phase23.py` dimostra entrambe catture. No fallback CPU; eager GPU solo riferimento numerico/warmup.
- Percorso completo core+interfacce+attenzione catturato e verificato separatamente in2.4. Smoke2.3 riguarda solo attenzione.

```powershell
$env:TEMP = (Resolve-Path '.runtime-tmp').Path
$env:TMP = $env:TEMP
.venv\Scripts\python.exe verify_phase23.py
```

Output: `results/phase23_attention.json`, log JSONL, hash sorgenti. Test numerici in `test_attention.py`; avviare via supervisore RAM/VRAM.
