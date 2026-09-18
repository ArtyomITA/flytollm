# Riassunto della notte 18 settembre 2026 (aggiornato a ogni fine run)

Utente a dormire dalle 02:17. Catena automatica: 8b2 (25 run a 8+8 sulla base default candidato + Muon 1e-3, riferimento L4 4,187) → 8d (E1-E4: far imparare il nucleo) → 8e (P0-P4b: canali d'ingresso per modalità) → 8c (L8, H5, H6, H7 corte; L9, C17, L10, C18 a 8000). Ogni run registrata in `SUITE_TEST_FASE_6B.md` con considerazioni; tabella completa in `PIANO_TEST_RIMASTI.md`.

## Risultati della notte (ordine cronologico; Δ negativo = meglio del riferimento)

| ora fine | run | CE 2000 | riferimento | Δ | verdetto |
|---|---|---|---|---|---|
| 02:03 | C16 gradiente allargato (soft_gw) a 8+8 | 4,183 | L4 4,187 | −0,004 | rumore; archi mossi 10,5% vs 2,5%: copertura ×4, CE ferma (3ª conferma dopo M0 e C14) |
| 02:29 | C16 + passo sinaptico 1e-3 a 8+8 | 4,185 | L4 4,187 | −0,002 | rumore; pesi mossi 3,2% (×18), archi mossi 20,6% (×8): le sinapsi si muovono molto, CE identica = loss piatta lungo le sinapsi (regime lazy confermato) |

## Diagnostiche della notte

- **E0 pannello delle cinque cause** (K8, 4+4, 8000; 02:29, 46 s): neuroni muti 81%; distanza dalla soglia xi mediana 9,1 (73% sopra 3, solo 7% nella finestra [1, 3]); sigma_U mediana 0,11 (solo 11% in [1/3, 1]); 61,5% dei neuroni non entra mai nella finestra del gradiente. Passo effettivo sigmoid(raw) medio 0,029 = taglio ×34. Gradiente sulle sinapsi: solo 16,6% ne riceve, SNR mediano 0,22 = rumore puro (0,25), SNR > 1 solo 0,4%. Il 10% di sorgenti più attive porta il 59% del movimento. Lettura: init quiescente + gradiente senza direzione (rumore) + softplus ×34, in serie; C16+1e-3 conferma il regime lazy. Cura in coda: E1 (8d). Proposta nuova: E5 accumulo del gradiente sinaptico su 16 update sopra E1 (da approvare).
- **C13 caricatore universale delle varianti** (`phase8_load_variant.py`; 02:30, 0,9 min): K5 ricostruito con nucleo non fuso, CE DEV 4,1529 = 4,152 di fine allenamento. Suite di inferenza ora possibile su ogni variante.

## Cose aggiunte stanotte (nessuna run programmata modificata)

- `phase8_modality_lesion.py`: lesione per senso a inferenza (modo info / silence, controllo random), in coda 8e dopo P0/P1 (~15 min l'una).
- `phase8_e0_panel.py`: pannello E0 (eccitabilità xi/sigma per neurone, passo effettivo sigmoid(raw), SNR del gradiente su 16 campioni, dove si muovono i pesi).
- `phase8_load_variant.py`: C13, carica qualsiasi checkpoint di variante per inferenza (VariantSuite).
- Pagina Pages: slide 7 corretta (+0,18 / +0,12); da pushare.

## Proposte in attesa di decisione (non applicate)

1. C17/C18 (8000, 100 + 140 min) valgono solo se E4 (8d) dice che il nucleo può imparare; altrimenti sostituirle col regime vincente di 8d.
2. B10 reservoir: tenere (controllo pulito, 27 min).
3. H6/H7 wake_learn: leggere come test di copertura; se vince E2 o E3 aggiungere H6' = wake + regime vincente.
4. Corti da aggiungere dopo 8d: E2+E3 insieme; E1 + bilanciamento E/I se E1 instabile.
5. E5 (nuova, da E0): accumulo del gradiente di core.raw su 16 update (SNR ×4) sopra E1, 8+8, 2000 update, 27 min.

## Problemi della notte

(nessuno finora)
