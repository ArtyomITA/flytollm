# Riassunto della notte 18 settembre 2026 (aggiornato a ogni fine run)

Utente a dormire dalle 02:17. Catena automatica: 8b2 (25 run a 8+8 sulla base default candidato + Muon 1e-3, riferimento L4 4,187) → 8d (E1-E4: far imparare il nucleo) → 8e (P0-P4b: canali d'ingresso per modalità) → 8c (L8, H5, H6, H7 corte; L9, C17, L10, C18 a 8000). Ogni run registrata in `SUITE_TEST_FASE_6B.md` con considerazioni; tabella completa in `PIANO_TEST_RIMASTI.md`.

## Risultati della notte (ordine cronologico; Δ negativo = meglio del riferimento)

| ora fine | run | CE 2000 | riferimento | Δ | verdetto |
|---|---|---|---|---|---|
| 02:03 | C16 gradiente allargato (soft_gw) a 8+8 | 4,183 | L4 4,187 | −0,004 | rumore; archi mossi 10,5% vs 2,5%: copertura ×4, CE ferma (3ª conferma dopo M0 e C14) |
| 02:29 | C16 + passo sinaptico 1e-3 a 8+8 | 4,185 | L4 4,187 | −0,002 | rumore; pesi mossi 3,2% (×18), archi mossi 20,6% (×8): le sinapsi si muovono molto, CE identica = loss piatta lungo le sinapsi (regime lazy confermato) |
| 03:04 | C16 + passo sinaptico 1e-2 a 8+8 | 4,200 | L4 4,187 | +0,013 | rumore; pesi mossi 27,8% (×160), archi 30,4%, spike −5%: CE indipendente dal movimento sinaptico su 3 ordini di grandezza (serie 1e-4/1e-3/1e-2: 4,183/4,185/4,200). Famiglia C14/C16 chiusa: passo e copertura non curano |
| 03:33 | N1 due letture dell'attention per token (sottopassi 4 e 12) a 8+8 | 4,196 | L4 4,187 | +0,009 | rumore; +6% di costo. Una seconda rilettura del contesto dentro il giro non aggiunge nulla |
| 04:25 | N2 attention a ogni sottopasso (16 letture, cache per lettura, step id, concat) a 8+8 | 4,178 | L4 4,187 | −0,009 | rumore; costo ×1,8 (1.216 ms/update), 5,6 M parametri; parte peggio e chiude appena sotto. Rileggere il contesto dentro il giro non aggiunge informazione |
| 04:51 | N5 token iniettato solo al primo sottopasso a 8+8 | 4,914 | L4 4,187 | **+0,727** | senza corrente tonica la rete tace (spike ×0,01): l'input injection sostenuta è il supporto vitale della dinamica. Confondente: toglie anche il bias tonico 0,6; proposta N5b (bias sempre, token solo al primo) |
| 05:20 | M4 porte a conduttanza a 8+8 | 4,188 | L4 4,187 | +0,001 | nulla; +11% costo. La conduttanza pagava sulle sinapsi ricorrenti, non sugli ingressi |
| 06:07 | A2.7.2 guadagno esterno per superclasse (lr 1e-2) a 8+8 | 4,198 | L4 4,187 | +0,011 | rumore; costo ×1,9. Il modello alza la corrente ×2 su cb/vnc/ol intrinsic e cb_sensory, la inverte su sensory_ascending; spike +35% ma CE ferma: più eccitazione da sola non basta |
| 06:33 | B2.5.3a nucleo identità (sinapsi spente e congelate, neuroni isolati) a 8+8 | 5,864 | L4 4,187 | **+1,677** | pavimento senza grafo: il cablaggio vale 1,68 nat come reservoir fisso (C15: allenarlo vale 0,0006). Il grafo espande l'ingresso, le interfacce leggono |
| 06:56 | B10 reservoir (sinapsi reali congelate all'init) a 8+8 | 4,182 | L4 4,187 | −0,005 | identico al nucleo allenato, −6% costo. Quadro completo: grafo assente 5,864 · grafo fisso 4,182 · grafo allenato 4,187 |
| 07:22 | N4 due letture senza identificativo di passo a 8+8 | 4,198 | N1 4,196 / L4 4,187 | +0,002 / +0,011 | rumore: lo step id vale zero perché le letture ripetute non aggiungono nulla |
| 08:03 | N3 attention a ogni sottopasso con cache della prima lettura a 8+8 | 4,186 | N2 4,178 / L4 4,187 | +0,008 / −0,001 | rumore; costo ×1,66. Famiglia N chiusa: attention nel giro neutra (N1-N4 entro ±0,011), conta solo la corrente d'ingresso sostenuta (N5) |

## Diagnostiche della notte

- **E0 pannello delle cinque cause** (K8, 4+4, 8000; 02:29, 46 s): neuroni muti 81%; distanza dalla soglia xi mediana 9,1 (73% sopra 3, solo 7% nella finestra [1, 3]); sigma_U mediana 0,11 (solo 11% in [1/3, 1]); 61,5% dei neuroni non entra mai nella finestra del gradiente. Passo effettivo sigmoid(raw) medio 0,029 = taglio ×34. Gradiente sulle sinapsi: solo 16,6% ne riceve, SNR mediano 0,22 = rumore puro (0,25), SNR > 1 solo 0,4%. Il 10% di sorgenti più attive porta il 59% del movimento. Lettura: init quiescente + gradiente senza direzione (rumore) + softplus ×34, in serie; C16+1e-3 conferma il regime lazy. Cura in coda: E1 (8d). Proposta nuova: E5 accumulo del gradiente sinaptico su 16 update sopra E1 (da approvare).
- **C13 caricatore universale delle varianti** (`phase8_load_variant.py`; 02:30, 0,9 min): K5 ricostruito con nucleo non fuso, CE DEV 4,1529 = 4,152 di fine allenamento. Suite di inferenza ora possibile su ogni variante.

## Cose aggiunte stanotte (nessuna run programmata modificata)

- `phase8_modality_lesion.py`: lesione per senso a inferenza (modo info / silence, controllo random), in coda 8e dopo P0/P1 (~15 min l'una).
- `phase8_e0_panel.py`: pannello E0 (eccitabilità xi/sigma per neurone, passo effettivo sigmoid(raw), SNR del gradiente su 16 campioni, dove si muovono i pesi).
- `phase8_load_variant.py`: C13, carica qualsiasi checkpoint di variante per inferenza (VariantSuite).
- Pannello E0 esteso alle varianti (`--run`, init a fluttuazione riconosciuta) e aggiunto in fondo alla coda 8d su L4 e su tutti i checkpoint 8d (E1, E2, E3, E4, E5; ~1,5 min l'uno): misura se ogni regime cambia eccitabilità, SNR del gradiente e dove si muovono i pesi.
- `--core-accumulate N` in `pretrain_control.py` / `HybridMuon`: Adam sulle sinapsi ogni N update sul gradiente sommato (E5), capture-safe, self-test 10.
- Pagina Pages: slide 7 corretta (+0,18 / +0,12); da pushare.

## Proposte in attesa di decisione (non applicate)

1. C17/C18 (8000, 100 + 140 min) valgono solo se E4 (8d) dice che il nucleo può imparare; altrimenti sostituirle col regime vincente di 8d.
2. B10 reservoir: tenere (controllo pulito, 27 min).
3. H6/H7 wake_learn: leggere come test di copertura; se vince E2 o E3 aggiungere H6' = wake + regime vincente.
4. Corti da aggiungere dopo 8d: E2+E3 insieme; E1 + bilanciamento E/I se E1 instabile.
5. E5 (nuova, da E0): accumulo del gradiente di core.raw su 16 update (SNR ×4), passo 3e-3, 8+8: AGGIUNTA in fondo alla coda 8d (`phase8_E5_accum16_T8_2000`, 27 min, self-test superato); E5 + E1 da decidere dopo il risultato di E1.
6. N5b (nuova, da N5): bias tonico dell'iniettore a ogni sottopasso, parte dipendente dal token solo al primo sottopasso; 8+8, 27 min: separa 'informazione ripetuta' da 'corrente vitale'.

## Problemi della notte

- 03:05: tre smoke di 8b2 FALLITI (shock, homeo, arousal) → la coda ha SALTATO N6d (shock di profondità), H1 (omeostasi 8+8), H2/H3 (arousal), H4 (omeostasi 12+12). Cause trovate: (1) `current_depth()` indicizzava con un tensore 0-dim = sync host, vietato dentro la cattura del CUDA Graph; (2) `FairCapture.restore()` ripristinava i parametri ma non i buffer che si muovono in allenamento (scarti omeostatici, contatori di arousal e di schedule): passo eager e replay partivano da stati diversi, controllo di equivalenza fallito. Fix 03:10: indice con tensore a 1 elemento (`fly_lm_variants.py`), ripristino anche dei buffer (`phase3_t45.py`, nessun effetto sui modelli con buffer costanti = tutte le run precedenti; il file è nel fingerprint dei checkpoint, quindi i checkpoint salvati prima non sono più ripristinabili con --resume: nessuna coda lo usa). Self-test CPU ok. Smoke rifatti nella pausa di coda dopo N1 (03:34-03:39): shock, homeo, arousal tutti OK (135 / 90 / 84 s). Le 5 run saltate vengono rieseguite in fondo alla catena (aggiunto in `phase8c_queue.py`: rilancio di 8b2 sulle run non fatte). Costo dell'incidente: ~8 min di GPU ferma per le pause, 5 run spostate in fondo (~2,5 h in più alla catena).

