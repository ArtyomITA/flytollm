"""Record finished control runs into SUITE_TEST_FASE_6B.md: for every known run whose results JSON exists, replace the
'Risultato: in coda.' / 'Risultato: in corso.' line of its test with the DEV curve every 500 updates, the final CE and
accuracy, the delta against the reference and the pre-registered verdict, plus the greedy generation.
Idempotent: a test already recorded (marker 'Risultato (run ') is left alone unless --force."""
import argparse, json, re
from bench_runtime import ROOT

REAL_2000 = 5.087
BASE7_2000 = 4.580  # phase7_baseline_4000 at 2000 updates (standard configuration)
REAL_8000 = 3.959
RUNS = {
    # name: (heading marker in the .md, reference CE, reference label)
    'phase6c_permweights_2000': ('### B3 =', REAL_2000, 'reale 5,087'),
    'phase6c_permsigns_2000': ('### B4 =', REAL_2000, 'reale 5,087'),
    'phase6c_erdosrenyi_2000': ('### B5 =', 4.533, 'configuration model 4,533'),
    'phase6c_withinsuperclass_2000': ('### B6 =', 4.931, 'gradi conservati 4,931'),
    'phase6c_ports_random_2000': ('### B7 =', REAL_2000, 'reale 5,087'),
    'phase6c_ports_olfactory_2000': ('### B7 =', REAL_2000, 'reale 5,087'),
    'phase6c_readout_anatomical_2000': ('### D3 =', REAL_2000, 'reale 5,087'),
    'phase6c_core_apl_2000': ('### D4 =', REAL_2000, 'reale 5,087'),
    'phase6c_core_graded_ol_2000': ('### D5 =', REAL_2000, 'reale 5,087'),
    'phase6c_core_tau_type_2000': ('### D6 =', REAL_2000, 'reale 5,087'),
    'phase6c_core_reversal_2000': ('### D7 =', REAL_2000, 'reale 5,087'),
    'phase6c_relthr_2000': ('### D8 =', REAL_2000, 'reale 5,087'),
    'phase6c_softsign_2000': ('### D9 =', REAL_2000, 'reale 5,087'),
    'phase6d_ports_visual_2000': ('### F1 ', REAL_2000, 'reale 5,087'),
    'phase6d_readout_fru_2000': ('### F2 ', REAL_2000, 'reale 5,087'),
    'phase6d_visual_fru_2000': ('### F3 ', REAL_2000, 'reale 5,087'),
    'phase6d_readout_hub_2000': ('### F4 ', REAL_2000, 'reale 5,087'),
    'phase6d_readout_cx_2000': ('### F5 ', REAL_2000, 'reale 5,087'),
    'phase6d_ports_auditory_2000': ('### F6 ', REAL_2000, 'reale 5,087'),
    'phase6d_visual_hub_2000': ('### F7 ', REAL_2000, 'reale 5,087'),
    'phase6s_S8_fused_reorder_2000': ('S8 (2000 update, kernel fuso + riordino', REAL_2000, 'pilota 5,087'),
    'phase6s_S0c_reference_2000': ('S8 (2000 update, kernel fuso + riordino', REAL_2000, 'pilota 5,087'),
    'phase6s_S8b_fused_2000': ('S8 (2000 update, kernel fuso + riordino', REAL_2000, 'pilota 5,087'),
    # phase 6f promotions: fresh 8000-update runs, compared with the real fly at 8000
    'phase6b_configmodel_s41_8000': ('### B2 =', REAL_8000, 'reale 8000 3,959'),
    'phase6c_erdosrenyi_8000': ('### B5 =', REAL_8000, 'reale 8000 3,959'),
    'phase6c_permsigns_8000': ('### B4 =', REAL_8000, 'reale 8000 3,959'),
    'phase6c_ports_random_8000': ('### B7 =', REAL_8000, 'reale 8000 3,959'),
    'phase6c_readout_anatomical_8000': ('### D3 =', REAL_8000, 'reale 8000 3,959'),
    'phase6c_relthr_8000': ('### D8 =', REAL_8000, 'reale 8000 3,959'),
    'phase6c_softsign_8000': ('### D9 =', REAL_8000, 'reale 8000 3,959'),
    # phase 7: standard baseline and 'tengo' levers (SUITE section G); tengo runs compared with the baseline at 2000 once known
    'phase7_baseline_4000': ('### G0 Baseline', 4.070, 'porte random 4000 4,070'),
    'phase7_tengo_T8_2000': ('### G1 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_tengo_shortpath_2000': ('### G2 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_tengo_relthr5_2000': ('### G3 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_tengo_all_2000': ('### G4 ', BASE7_2000, 'baseline standard 2000 4,580'),
    # phase 7c levers (SUITE section H), compared with the standard baseline at 2000
    'phase7_lever_bias_2000': ('### H1 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_bias_lr1e4_2000': ('### H1 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_bias_lr3e4_2000': ('### H1 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_tau_2000': ('### H2 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_reversal_2000': ('### H3 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_excit_2000': ('### H4 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_mono_2000': ('### H5 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_scale05_2000': ('### H6 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_scale2_2000': ('### H6 ', BASE7_2000, 'baseline standard 2000 4,580'),
    # phase 7e: Muon on the dense matrices (SUITE section I)
    'phase7_lever_muon_2000': ('### I1 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_muon3e4_2000': ('### I2 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_T8muon_2000': ('### I3 ', 4.399, '8+8 sottopassi (G1) 4,399'),
    'phase7_lever_muonhead_2000': ('### I4 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_T8rev_2000': ('### G5 ', 4.399, '8+8 sottopassi (G1) 4,399'),
    # phase 7f (SUITE section J): advised tests and promotions
    'phase7_lever_tau3e3_2000': ('### J1 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_revtau_2000': ('### J2 ', 4.459, 'conduttanza (H3) 4,459'),
    'phase7_promote_T8_8000': ('### J4 ', 3.689, 'porte random 8000 3,689'),
    'phase7_promote_reversal_8000': ('### J4 ', 3.689, 'porte random 8000 3,689'),
    'phase7_promote_T8rev_8000': ('### J4 ', 3.689, 'porte random 8000 3,689'),
}


def fmt(x):
    return f'{x:.3f}'.replace('.', ',')


def block(name, data, ref, ref_label):
    r = data['result']
    curve = ' / '.join(f"{e['update']}: {fmt(e['dev']['ce'])} ({e['dev']['accuracy']:.3f})" for e in r.get('curve', []))
    final = r['curve'][-1]['dev']['ce'] if r.get('curve') else float('nan')
    delta = ref - final
    if abs(delta) < 0.10:
        verdict = f'entro 0,10 dal riferimento ({ref_label}): nessun effetto misurabile'
    elif delta > 0:
        verdict = (f'migliore del riferimento ({ref_label}) di {fmt(delta)} nat: effetto positivo, vantaggio confermato a 8000' if ref == REAL_8000                   else f'migliore del riferimento ({ref_label}) di {fmt(delta)} nat: effetto positivo, candidata alla promozione a 8000')
    else:
        verdict = f'peggiore del riferimento ({ref_label}) di {fmt(-delta)} nat: effetto negativo'
    gen = next((g['text'] for g in r.get('generation', []) if g.get('temperature') == 0 and g.get('prompt') == 'Once upon a time'), '')
    ctrl = r.get('control', {})
    extra = []
    for key in ('ports', 'readout', 'core'):
        v = ctrl.get('stats', {}).get(key)
        if isinstance(v, dict):
            extra.append(f"{key}: " + ', '.join(f"{k}={val}" for k, val in v.items() if k != 'labels'))
    return (f"Risultato (run `{name}`, {r['updates']} update, {r['targets']} target, {r['mean_step_s']*1000:.0f} ms/update, core {ctrl.get('fast_mode', 'off')}): "
            f"curva CE DEV (accuracy) ogni 500: {curve}. Finale {fmt(final)}; Δ = {fmt(delta)} rispetto a {ref_label}. Lettura pre-registrata: {verdict}. "
            f"Generazione greedy: \"{' '.join(gen.split())}\". " + (' '.join(extra) if extra else ''))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--force', action='store_true'); a = p.parse_args()
    path = ROOT / 'SUITE_TEST_FASE_6B.md'; text = path.read_text(encoding='utf8'); changed = []
    for name, (marker, ref, ref_label) in RUNS.items():
        jp = ROOT / 'results' / f'{name}.json'
        if not jp.exists():
            continue
        data = json.loads(jp.read_text())
        if not data.get('ok') or not isinstance(data.get('result'), dict):
            continue
        start = text.find(marker)
        if start < 0:
            print('marker not found', name, marker); continue
        ends = [e for e in (text.find('\n### ', start + 1), text.find('\n## ', start + 1), text.find('\n---', start + 1)) if e >= 0]
        section_end = min(ends) if ends else len(text)
        section = text[start:section_end]
        if f'run `{name}`' in section and not a.force:
            continue
        new_block = block(name, data, ref, ref_label)
        if re.search(r'Risultato: in (coda|corso)\.', section):
            section_new = re.sub(r'Risultato: in (coda|corso)\.', new_block, section, count=1)
        elif f'run `{name}`' in section:
            section_new = re.sub(r'Risultato \(run `' + re.escape(name) + r'`.*?(?=\n|$)', new_block, section, count=1)
        else:
            section_new = section.rstrip('\n') + '\n' + new_block + '\n'
        text = text[:start] + section_new + text[section_end:]
        changed.append(name)
    path.write_text(text, encoding='utf8')
    print('recorded:', changed)


if __name__ == '__main__':
    main()
