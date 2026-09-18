"""Rebuild PIANO_TEST_RIMASTI.md: index, best values and notable conclusions first, then the numbered checklist, then
every test per category (short / long) with a Result column and a terse technical-notes column (caveman ultra, written
for the assistant), then the queue of next tests. The M0 micro-sweep table is read from the result files at run time, so
the script can be re-run as the sweep progresses. Source of the per-category rows: piano_sez2_v3.md in the scratchpad
(copied next to this script as PIANO_SEZ2_BASE.md the first time)."""
import io, json, re, shutil
from pathlib import Path
from bench_runtime import ROOT

BASE = ROOT / 'PIANO_SEZ2_BASE.md'
SCRATCH = Path('C:/Users/ADMINI~1/AppData/Local/Temp/claude/E--claudecode-pesante/94ed36ed-ac4d-4e37-a50f-881f49cc7e50/scratchpad/piano_sez2_v3.md')
if not BASE.exists():
    shutil.copy(SCRATCH, BASE)

# ---- result updates on top of the base rows (test id -> new Result cell)
RESULT = {
    'L7': 'FATTO: 4,132 (+0,020 su K5: dentro il rumore). Non candidata.',
    'N0': 'FATTO: nel blocco pre rileggere dà lo stesso richiamo (coseno ≥ 0,965); dopo il feedback lo stato si allontana: al sottopasso 8 una rilettura guarderebbe altrove nel 39% dei token. Dinamica = ciclo di periodo 2.',
    'N6c': 'FATTO: fra 8 e 12 il cambio costa +0,04 / +0,07; tutto ciò che tocca 4+4 costa 0,12-0,50; shock 4 → 12 = +0,34.',
    'C15': "FATTO: sinapsi rimesse all'init su K8: 3,5407 → 3,5401 (0,0006). Il nucleo è un reservoir fisso; l'apprendimento sta nelle interfacce.",
    'C14': 'FATTO: passo sinaptico 1e-3 / 1e-2 a 4+4: 4,275 / 4,276 (= 4,275); a 8+8 1e-2: 4,202 vs L4 4,187 (rumore). Pesi mossi 1,4% / 14,6%: si muovono, la CE non cambia.',
    'N1': 'FATTO: due letture (sottopassi 4 e 12, cache per lettura, step id) a 8+8: 4,196 vs L4 4,187 (+0,009, rumore), +6% costo.',
    'N2': 'FATTO: attention a ogni sottopasso (cache per lettura, step id, concat) a 8+8: 4,178 vs 4,187 (−0,009, rumore), costo ×1,8.',
    'N3': 'FATTO: ogni sottopasso con cache della prima lettura: 4,186 vs N2 4,178 (rumore), costo ×1,66.',
    'N4': 'FATTO: due letture senza step id: 4,198 vs N1 4,196 (rumore): lo step id vale zero.',
    'N5': 'FATTO: token iniettato solo al primo sottopasso: 4,914 (+0,727): senza corrente tonica la rete tace (spike ×0,01). Proposta N5b (bias sempre, token solo al primo).',
    'M4': 'FATTO: porte a conduttanza a 8+8: 4,188 vs 4,187 (nulla), +11% costo.',
    'A2.7.2': 'FATTO: guadagno esterno per superclasse (lr 1e-2) a 8+8: 4,198 (+0,011, rumore), costo ×1,9; guadagni ×2 su intrinsic/cb_sensory, invertiti su sensory_ascending; spike +35%, CE ferma.',
    'B2.5.3a': 'FATTO: nucleo identità (sinapsi spente e congelate) a 8+8: 5,864 (+1,677): il cablaggio vale 1,68 nat come reservoir fisso.',
    'B10': "FATTO: reservoir (sinapsi reali congelate all'init) a 8+8: 4,182 vs 4,187: identico al nucleo allenato, −6% costo.",
    'C13': 'FATTO: caricatore universale delle varianti (phase8_load_variant.py): K5 ricostruito, CE DEV 4,1529 = 4,152. Suite di inferenza su ogni variante.',
    'C16': 'FATTO: gradiente allargato a 8+8 con passo 1e-4 / 1e-3 / 1e-2: 4,183 / 4,185 / 4,200 vs L4 4,187 (tutto rumore) con pesi mossi 0,4% / 3,2% / 27,8% e archi mossi 10,5% / 20,6% / 30,4%: la CE non dipende dal movimento sinaptico. Famiglia chiusa.',
}
# ---- terse technical notes (caveman ultra, for the assistant)
NOTES = {
    'C12': 'power iter oscilla periodo 2 (0,7194/0,3475, prod 0,25). rho=0,500 da norm Σ|w_in|=0,5. trained = init a 4 cifre. attivi-only 0,405 (T8 0,412). null signed 0,198/0,234, abs 0,493/0,464. fix: geo-mean 2 iter + seed fisso in phase6c_inference.spectral. C11/G0i/J3 void.',
    'C13w': 'dw/w0: 1k 0,11% · 4k 0,21% · 8k 0,30% · Muon8k 0,26 · T8-8k 0,32. moved>1e-4 2,9% (from-active 14,9%), max 0,016. causa: gw = g_dst·s_src, 81% muti, 19% archi da sorgenti attive; softplus raw≈−3,5 → step_w ≈ lr·σ(raw) ≈ lr·0,03.',
    'D2b': 'reach lineare con segno, init thr10: random→chunks hop1 reach 0,85 coh 0,26; visual→chunks hop3 reach 0,37 coh 0,07; visual→fru hop2 reach 0,03; visual→hub coh 0,96 net<0; anatomical hop1 coh≈1 reach 0,41.',
    'L5': '952 ms/upd, 34,6 min. gap vs L4 0,064→0,025 lungo run. = K5 (4,152) senza cond/tau. scala profondità 4,275/4,187/4,162. pari tempo: default ~5k upd = 3,79.',
    'L7': 'curva −0,02/−0,04 costante vs K5. dw 0,09% moved 0,5% (K5 0,12/1,9): pesi init piccoli → Δ assoluto minore. 1 seed non basta (rumore 0,021).',
    'F9': 'shortpath 14.957 porte (5.756 mecc + 9.201 vis proj). gap vs L4 0,29→0,15 lungo run (può chiudersi a 8k). 641 ms. porte meno → B9 separa capacità.',
    'K7': 'Muon lr: 1e-4 4,648 · 3e-4 4,411 · 1e-3 4,276 · 3e-3 4,250. gap 3e-3−1e-3: 0,225→0,078→0,048→0,026. grad 5,07. 3e-3 non adottato; parziale 8k a 3e-3 interrotta a 2000 (5,190/4,786/4,387).',
    'L4': 'gain 8+8 vs ottimizzatore: Adam +0,181 · Muon3e-4 +0,139 · Muon1e-3 +0,088. 680 ms (×1,8). pari tempo: default ~3,6k upd = 3,95.',
    'L3': 'seed 23 vs 17: Δ 0,021, curve sovrapposte (500 5,141/5,158). dw 0,120/0,125%, leak 0,40-0,996 entrambi. soglia pratica famiglia K5: <0,04 indistinguibile.',
    'L2': 'K8@7000 = 3,615 vs K5@2000 = 4,152, stesso wall (46 min). confonde dati visti (7k vs 2k upd).',
    'K0': '= config model 3,648; replica G0@4000 4,000 vs 3,999; 365 ms. riferimento di tutte le promozioni.',
    'K6': 'gap vs base: 2000 +0,138 · 4000 +0,096 · 6000 +0,070 · 8000 +0,081. 379 ms.',
    'K8': 'gap vs base: 2000 +0,288 · 4000 +0,155 · 6000 +0,114 · 8000 +0,112 (si stabilizza). vs 3e-4: +0,149→+0,031. grad med 5,9 max 25,8. replica K1 4,274. < Transformer 3,563 (Adam, non pari ottimizzatore).',
    'N0': 'cos rep vs real s1-8: .869 .934 .977 1 .978 .929 .854 .760. recall cos: .965 .980 .992 1 .993 .967 .902 .799. argmax agree: .851 .886 .936 1 .925 .849 .740 .614. JS max .096. entropia 1,10→1,25. spike Jaccard lag1 .308 lag2 .656 lag3 .354 lag4 .537. CE nel JSON (7,36) invalida: logit con embedding tied invece di model.head; fix fatto, rerun in 8c.',
    'N6c': 'matrice train×eval (4/8/12): 4: 4,276 4,396 4,620 · 8: 4,442 4,185 4,223 · 12: 4,666 4,235 4,162. diagonale = eval di fine run. no test-time scaling (8→12 peggiora 0,038).',
    'C15': 'K8 4+4 8k: trained 3,5407 · all-init 3,5401 · moved-init 3,5395 · unmoved-init 3,5494 (Δ 0,009: i pesi NON mossi contano più dei mossi = rumore di init). ol_intrinsic 89k neuroni 17,6% attivi, 0,24% archi mossi. bug closure fixato (record.add_).',
    'C14': '4+4: 1e-3 4,275 · 1e-2 4,276 (dw 1,4% / 14,6%). T8 1e-2: 4,202 vs 4,187 (+0,015 rumore). movimento ∝ lr, CE piatta: loss piatta lungo direzioni sinaptiche.',
    'N1': 'reads 4,12 per_read + step id: 4,196; 720 ms; grad 6,7; dw/moved = L4. curva sopra L4 a ogni ckpt di ~0,01.',
    'N2': 'every + per_read + step id + concat: 4,178; 1216 ms; 5,60 M param; VRAM 1,2 GB; curva parte +0,015 e chiude −0,009 (trend). spike 0,0526.',
    'N3': 'every + first kv: 4,186; 1127 ms (−7% vs N2). = N2 → qualità cache irrilevante.',
    'N4': 'reads 4,12 senza id: 4,198 = N1. id vale 0 (letteratura −0,107) perché letture ripetute non aggiungono nulla.',
    'N5': "inject first only: 4,914; spike 0,00057 (÷100); grad 26,9 (clip); 621 ms. confondente: toglie anche bias tonico 0,6 dell'iniettore. N5b = bias sempre, token al primo.",
    'M4': 'cond_ports E_e 3 E_i −1: 4,188; spike 0,0515 (−5%); 757 ms (+11%). conduttanza pagava su sinapsi ricorrenti (H3), non su ingressi.',
    'A2.7.2': 'gain_group superclasse lr 1e-2: 4,198; 1290 ms (×1,9, no kernel fuso per scalatura); spike 0,0733 (+35%). gains: cb_sensory 2,34 vnc_tbc 2,29 vis_centrif 2,14 cb_intr 2,12 vnc_intr 2,02 ol_sens 1,78 ol_intr 1,46; sens_asc −0,22 vnc_sens_tbc −0,15; gruppi senza porte = 1,000.',
    'B2.5.3a': 'identity (w×1e-6 frozen): 5,864; curva piatta 5,93/5,93/5,85/5,86; grad 20,8 clip; spike 0,0515; 589 ms. grafo = +1,68 nat come reservoir.',
    'B10': 'reservoir frozen init: 4,182; curva = L4 a ogni ckpt; spike 0,0536; 637 ms (−6%). C15 confermato da zero. assente 5,864 · fisso 4,182 · allenato 4,187.',
    'C13': 'VariantSuite = Suite con base.load_payload patchato; build_control_cns fast_mode off + control da worker.json; src32/dst32 droppati; strict load. 0,9 min.',
    'C16': 'T8 soft_gw lr 1e-4/1e-3/1e-2: CE 4,183/4,185/4,200; dw 0,38/3,24/27,8%; moved 10,5/20,6/30,4%; max 0,01/0,09/1,65; spike 0,0542/0,0538/0,0515; grad 6,4/6,4/6,2. 708 ms (+4%). E0: SNR grad 0,22 = rumore → Adam random walk ∝ lr. copertura e passo ≠ cura.',
}


def micro_table():
    from phase8_coverage import describe
    rows = [('default (1e-4)', 'phase8_micro_default_300')]
    rows += [(f'{lr}', f'phase8_micro_corelr{tag}_300') for lr, tag in (('1e-3', '1e3'), ('3e-3', '3e3'), ('1e-2', '1e2'), ('3e-2', '3e2'))]
    rows += [(f'{lr} + gradiente allargato', f'phase8_micro_soft_corelr{tag}_300') for lr, tag in (('1e-3', '1e3'), ('3e-3', '3e3'), ('1e-2', '1e2'), ('3e-2', '3e2'))]
    out = ['| lr sinapsi | CE a 100 / 200 / 300 | pesi mossi | archi mossi > 1e-4 | Δ massimo | norma grad | spike |', '|---|---|---|---|---|---|---|']
    done = 0
    for label, name in rows:
        d = describe(name)
        if 'curve' not in d:
            out.append(f'| {label} | in corso / in coda | | | | | |'); continue
        done += 1
        c = d['curve']; f = lambda x: f'{x:.3f}'.replace('.', ',')
        out.append(f"| {label} | {f(c['100'])} / {f(c['200'])} / {f(c['300'])} | {d['weights_relative_change'] * 100:.2f}% | {d['synapses_moved_fraction'] * 100:.1f}% | "
                   f"{d['largest_change']:.3f} | {d['grad_norm_last_fifth_median']:.2f} | {d['spike_sample_mean_last']:.3f} |".replace('.', ','))
    return '\n'.join(out), done, len(rows)


def section2():
    text = io.open(BASE, encoding='utf-8').read()
    lines = []
    for line in text.splitlines():
        if line.startswith('|'):
            cells = line.strip().strip('|').split('|')
            if set(line.replace('|', '').strip()) <= set('-'):
                line = line.rstrip() + '---|'
            elif cells[0].strip() == 'N.':
                line = line.rstrip() + ' Osservazioni tecniche (caveman ultra, per l\'assistente) |'
            else:
                key = re.sub(r'\s*\(.*?\)\s*', '', cells[0]).strip()
                if key in RESULT:
                    cells[-1] = ' ' + RESULT[key] + ' '
                    line = '|' + '|'.join(cells) + '|'
                line = line.rstrip() + ' ' + NOTES.get(key, '—') + ' |'
        lines.append(line)
    return '\n'.join(lines)


def main():
    P = ROOT / 'PIANO_TEST_RIMASTI.md'
    old = io.open(P, encoding='utf-8').read()
    checklist = old[old.index('## 1. Numerazione originale'):old.index('## 2. Tutti i test')]
    decisions = old[old.index('## 4. Decisioni'):] if '## 4. Decisioni' in old else ''
    micro, done, total = micro_table()
    head = f"""# Piano dei test FlyToLLM — aggiornato il 18 settembre 2026

## Indice

0. [In breve: miglior valore e conclusioni](#0-in-breve-miglior-valore-e-conclusioni)
1. [Numerazione originale, stato voce per voce](#1-numerazione-originale-stato-voce-per-voce)
2. [Tutti i test per categoria, corti e lunghi, con risultato e osservazioni tecniche](#2-tutti-i-test-per-categoria-corti-e-lunghi-con-risultato)
   A canale esterno · B null e controlli · C sonde · C-bis quanto cervello impara · D parametrizzazione · E sistemi · F regioni · K/L vincitrici · M attention anatomica · N looped e profondità
3. [Micro-sweep del passo sinaptico (M0), tabella](#3-micro-sweep-del-passo-sinaptico-m0)
4. [Fila dei prossimi test](#4-fila-dei-prossimi-test)
5. [Decisioni dell'utente pendenti](#5-decisioni-dellutente-pendenti)

## 0. In breve: miglior valore e conclusioni

**Miglior valore.** A 8000 update: configurazione standard + Muon 1e-3 sulle matrici dense = **3,541** (baseline standard 3,652; configuration model 3,648; Transformer piccolo 3,563 con Adam; GRU 3,429): è il candidato sicuro per il default. A 2000 update: tutte le leve insieme (8+8 + conduttanza + tau 1e-2 + Muon 1e-3) = **4,152** (replica con altro seed 4,173), 3,5 volte più lenta; 12+12 da solo 4,162; default candidato 4,275.

**Conclusioni degne di nota.**
1. **Le sinapsi quasi non imparano**: 0,30% di variazione in 8000 update, 2,9% degli archi mossi. Il nucleo è stato finora una scatola quasi fissa; l'apprendimento sta nelle interfacce. I confronti reale contro null fatti finora confrontavano reservoir.
2. **Il problema non è il passo ma quanti neuroni sparano** (M0): con lr sinaptico fino a 300 volte più grande i pesi si muovono fino al 15,8%, tutto stabile, ma la CE a 300 update non cambia e gli archi che ricevono gradiente salgono solo da 1,2% a 6,8%. Col gradiente allargato gli archi mossi passano da 3,3% a 14,6% a pari lr. Le leve giuste sono quelle che accendono neuroni: gradiente allargato, omeostasi, arousal, shock di profondità (in coda).
3. **Muon è l'unica leva che resta sopra soglia a 8000** (+0,112, costo zero). Le leve di dinamica sono acceleratori: 8+8 da +0,16 a +0,06, conduttanza da +0,10 a +0,03.
4. **A pari tempo GPU vince il default semplice**: nel tempo in cui K5 fa 2000 update (4,152) il default ne fa 7000 (3,615).
5. **Profondità**: satura a 8+8 (12+12 +0,025); fra 8 e 12 si può cambiare profondità quasi gratis (+0,04 / +0,07), tutto ciò che tocca 4+4 costa (shock 4 → 12: +0,34 subito).
6. **Attention**: nel blocco pre rileggere non aggiunge nulla; dopo il feedback lo stato cambia e a fine token una rilettura guarderebbe altrove nel 39% dei casi; la dinamica degli spike è un ciclo di periodo 2.
7. **Ingressi anatomici**: costano ancora (−0,145) ma la metà di prima; da porte random il segnale arriva all'85% della lettura in un salto, dagli occhi in 2-3 salti con segni che si cancellano.
8. **Rumore**: 0,02 fra run e fra semi; sotto 0,04 due configurazioni non si distinguono con una run.
9. **Errata**: il raggio spettrale riportato in C11/G0i/J3 era un artefatto (iterazione che oscilla); vale 0,500 per costruzione e non cambia con l'allenamento.
10. **Bug evitati**: la testa separata del trainer avrebbe scartato in silenzio le varianti looped (corretto prima di ogni run); regola di scelta automatica nelle code = proposta dell'assistente, da approvare; trappola del recorder ("run `nome`" nelle note).

"""
    queue = """## 4. Fila dei prossimi test

In esecuzione, coda 8b (riprendibile, log `results/phase7_queue.console.log`, stato `results/phase8b_live.json`), nell'ordine:
1. M0 micro-sweep del passo sinaptico (9 run da 3,5 min).
2. C14 lr sinaptico 1e-3 e 1e-2 a 2000; C16 gradiente allargato da solo e con 1e-3 / 1e-2 (5 × 15 min).
3. N6d shock di profondità 12 → 8 → 4 → 12 (35 min).
4. H1o omeostasi per tipo (15 min); H2a arousal a impulsi e a onda (2 × 15 min); omeostasi a 12+12 (35 min).
5. N1, N2, N5 attention nel giro; M4 porte a conduttanza; A2.7.2 guadagno per superclasse.
6. B2.5.3a identità, B10 reservoir, N4, N3, B8 seed porte, B9 pari capacità (2), B2.5.2 fattoriale (4).

Coda 8c (da lanciare dopo la 8b): C15 rifatta, N0 con la CE corretta, e i seguiti decisi sui risultati: conferma a 2000 e a 8000 del regime in cui più sinapsi imparano (C17), shock + passo sinaptico (N6e), scala dei null rifatta con sinapsi che imparano (B13), L1 (K5 a 8000).

Da implementare: C6b, C10b, C13, A5b, D3b, N6b, N7, N12, 2.5.3b, M1/M2, medi e lunghi delle categorie D, M, N.

"""
    body = head + checklist + section2() + '\n\n## 3. Micro-sweep del passo sinaptico (M0)\n\n' + \
        f'Run da 300 update sul default candidato ({done} di {total} finite). "Pesi mossi" = ‖w − w₀‖/‖w₀‖; "archi mossi" = frazione con |Δw| > 1e-4.\n\n' + micro + '\n\n' + queue + \
        ('## 5. Decisioni dell\'utente pendenti' + decisions.split('pendenti', 1)[1] if decisions else '')
    io.open(P, 'w', encoding='utf-8').write(body)
    print('piano rebuilt:', len(body), 'chars; micro-sweep', done, '/', total)


if __name__ == '__main__':
    main()
