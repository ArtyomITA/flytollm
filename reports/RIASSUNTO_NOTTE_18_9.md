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
| 08:32 | B8 porte random seme 23 a 8+8 | 4,185 | L4 (seme 17) 4,187 | −0,002 | il campione di porte non conta |
| 08:54 | B9a 2.639 porte random (÷7) a 8+8 | 4,214 | L4 4,187 | +0,027 | sette volte meno porte costano 0,03; spike ÷7,5. Il costo delle porte anatomiche (0,10-0,25 a pari numero, 0,82 per gli occhi) è anatomia, non numero |
| 09:16 | B9b 1.733 porte random (÷10) a 8+8 | 4,221 | L4 4,187 | +0,034 | curva porte→CE: 17.937 4,187 · 2.639 4,214 · 1.733 4,221; spike ÷13. Capacità d'ingresso abbondante |
| 09:43 | B2.5.2a grafo reale soglia 10, attention accesa, 8+8 | 4,184 | L4 (relthr) 4,187 | −0,003 | soglia 10 = relthr; riferimento della fattoriale grafo × attention |
| 17:17 | H1 omeostasi di risveglio per tipo a 8+8 | 5,608 | L4 4,187 | **+1,42 (diverge)** | gradiente esplode da update ~700 (3,6 → 1e10): scarti di soglia saturano (mediana 0,66, tetto 0,9), due terzi del cervello a ridosso della soglia = backward supercritico. Idea valida, parametri no: proposta H1b (tetto 0,3-0,4, bersaglio 0,005). Non usare in run lunghe (C18) |
| 17:44 | H2 impulsi di arousal (soglia −0,3 per 100 update ogni 500) a 8+8 | 4,245 | L4 4,187 | +0,058 | stabile; spike +48% negli impulsi, archi mossi ×2,6; leggermente peggio (allenato a tratti in un regime diverso da quello di valutazione). Non candidata |
| 18:10 | H3 arousal a onda liscia a 8+8 | 4,218 | L4 4,187 | +0,031 | stabile, archi mossi ×2,1, metà del costo di H2 |
| 18:16 | C15 (sinapsi all'init) su L4 / H2 / H3 | contributo sinapsi +0,0044 / −0,0107 / +0,0017 | | | l'arousal allarga la copertura ma le sinapsi in più non lavorano (in H2 sono leggermente dannose a arousal spento): l'indizio non regge a 2000 update con porte random |
| 18:57 | H4 omeostasi di risveglio a 12+12 (parametri di H1) | 5,870 | L5 4,162 | **+1,71 (diverge)** | esplode prima e più forte che a 8+8 (gradiente 1e16): la profondità peggiora. Conferma il rinvio di H6/H7/C18; si aspetta H1b |
| 19:24 | B2.5.2b grafo reale, attention SPENTA, 8+8 | 4,282 | B252a (attention accesa) 4,184 | +0,098 | il cervello senza attention arriva a 4,28: a 2000 update l'attention vale 0,10 nat |
| 19:51 | B2.5.2c grafo RIMESCOLATO (gradi conservati), attention accesa, 8+8 | 4,243 | B252a (grafo reale) 4,184 | +0,059 | **il grafo reale è avanti di 0,06**: prima volta che batte un null a gradi conservati (prima: null meglio o pari). Un solo seme: da confermare (secondo seme + 8000) |
| 20:22 | B2.5.2d grafo rimescolato, attention spenta, 8+8 | 4,321 | fattoriale: reale 4,184 / 4,282 · rimescolato 4,243 / 4,321 | interazione +0,020 | attention vale 0,098 (reale) e 0,078 (rimescolato): additivi, nessun mascheramento. Grafo reale avanti in entrambe le coppie (+0,059 / +0,039). Coda 8b2 FINITA 20:22, parte 8d |

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
7. DECISO 18:25: porte anatomiche nel default attuale a 8000 update (P0L) + gemella con arousal a onda (P0A), in fondo a 8e, con C15 finale.
6. N5b (nuova, da N5): bias tonico dell'iniettore a ogni sottopasso, parte dipendente dal token solo al primo sottopasso; 8+8, 27 min: separa 'informazione ripetuta' da 'corrente vitale'.

## Correzione del pomeriggio (16:40, dopo l'obiezione dell'utente)

C15 rifatto sui checkpoint del vecchio standard (porte anatomiche, Adam): contributo delle sinapsi allenate 0,008 a 2000 update, 0,089 a 8000, **0,243 a 26.247**: cresce con l'allenamento. Nel nuovo standard (porte random) a 8000: 0,0006. Quindi: (1) "allenare le sinapsi vale zero" vale solo a pochi update e con porte random; (2) l'SNR del gradiente è uguale al rumore anche a 26k (0,235), eppure le sinapsi imparano: il segnale si accumula col tempo; (3) con porte random le interfacce scavalcano il cervello (1 salto fino alla lettura), con porte anatomiche il modello è costretto a usarlo. Dettagli: suite, voce C15b.

Coda ripartita alle 16:39 (ok dell'utente): N6d, H1-H4, B252b/c/d, poi 8d (con N5b aggiunta, sì dell'utente) → 8e → 8c.

## Stato alle 09:48

Tutto in pausa su tua richiesta: marker STOP_PHASE7, coda 8b2 fermata al confine dopo lo smoke off (09:47:38), 8c non lanciata, GPU libera, nessuna run invalidata. Mancano in 8b2: B252b/c/d (fattoriale grafo × attention) e le 5 saltate (N6d, H1-H4). Comandi per riprendere nel handoff (voce 09:48).

## Problemi della notte

- 03:05: tre smoke di 8b2 FALLITI (shock, homeo, arousal) → la coda ha SALTATO N6d (shock di profondità), H1 (omeostasi 8+8), H2/H3 (arousal), H4 (omeostasi 12+12). Cause trovate: (1) `current_depth()` indicizzava con un tensore 0-dim = sync host, vietato dentro la cattura del CUDA Graph; (2) `FairCapture.restore()` ripristinava i parametri ma non i buffer che si muovono in allenamento (scarti omeostatici, contatori di arousal e di schedule): passo eager e replay partivano da stati diversi, controllo di equivalenza fallito. Fix 03:10: indice con tensore a 1 elemento (`fly_lm_variants.py`), ripristino anche dei buffer (`phase3_t45.py`, nessun effetto sui modelli con buffer costanti = tutte le run precedenti; il file è nel fingerprint dei checkpoint, quindi i checkpoint salvati prima non sono più ripristinabili con --resume: nessuna coda lo usa). Self-test CPU ok. Smoke rifatti nella pausa di coda dopo N1 (03:34-03:39): shock, homeo, arousal tutti OK (135 / 90 / 84 s). Le 5 run saltate vengono rieseguite in fondo alla catena (aggiunto in `phase8c_queue.py`: rilancio di 8b2 sulle run non fatte). Costo dell'incidente: ~8 min di GPU ferma per le pause, 5 run spostate in fondo (~2,5 h in più alla catena).
- 16:50: N6d (shock di profondità) ABORTITA all'update 544 dalla guardia della RAM di sistema (`guard_abort: host_ram`): RAM del PC già all'83% prima della run (browser, app), picco 99% su 13,9 GB. Non è un difetto del modello (smoke ok, 544 update regolari). Persi 11 min. N6d verrà rifatta in fondo alla catena (rilancio automatico di 8b2 sulle run non fatte). Rischio per le run successive finché la RAM resta sopra ~85%.
