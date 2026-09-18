"""Render the two animated decks of docs/index.html (pipeline with the brain, test results) to an MP4, offscreen and
frame by frame: a headless Chromium draws the page with a mocked clock (Playwright page.clock), so every output frame
advances the page time by exactly the same amount; nothing is captured from the screen and the result is smooth whatever
the load of the machine. Frames go straight to ffmpeg through a pipe (no temporary images).

    python tools/render_video.py            # -> media/flytollm_decks.mp4 (1920x1080, 30 fps, page time x2.5)

Requires: playwright (with an installed Chrome: channel='chrome'), ffmpeg in PATH."""
import argparse, subprocess, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BG = '0x070A0E'


def encoder(path, fps):
    return subprocess.Popen(
        ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', str(fps), '-c:v', 'mjpeg', '-i', '-',
         '-vf', f'scale=1920:-2:flags=lanczos,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color={BG},format=yuv420p',
         '-c:v', 'libx264', '-crf', '17', '-preset', 'slow', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)


def render(page, selector, first_button, stop, out, fps, step_ms, max_seconds):
    """stop(page) -> True when the deck has come back to its first slide after a full tour."""
    rig = page.locator(selector).first
    rig.scroll_into_view_if_needed()
    page.evaluate("(s) => document.querySelector(s).scrollIntoView({block: 'center'})", selector)
    page.clock.run_for(200)
    page.locator(first_button).first.click()
    page.clock.run_for(50)
    ff = encoder(out, fps); frames = 0; started = time.time(); left = False
    while frames < max_seconds * fps:
        ff.stdin.write(rig.screenshot(type='jpeg', quality=93))
        frames += 1
        page.clock.run_for(step_ms)
        state = stop(page)
        if state == 'away':
            left = True
        elif state == 'first' and left:
            break
        if frames % 150 == 0:
            print(f'  {out.name}: {frames} frames, {time.time() - started:.0f} s', flush=True)
    ff.stdin.close(); ff.wait()
    print(f'{out.name}: {frames} frames = {frames / fps:.1f} s of video, rendered in {time.time() - started:.0f} s', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--speed', type=float, default=2.5, help='page time per second of video')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--scale', type=float, default=1.7205)  # 1116 px rig -> 1920 px, no resampling
    p.add_argument('--only', choices=['pipeline', 'tests'], default=None)
    a = p.parse_args()
    media = ROOT / 'media'; media.mkdir(exist_ok=True)
    step_ms = int(round(1000 * a.speed / a.fps))
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel='chrome', headless=True, args=['--disable-gpu', '--force-color-profile=srgb'])
        context = browser.new_context(viewport=dict(width=a.width, height=900), device_scale_factor=a.scale, reduced_motion='no-preference')
        page = context.new_page()
        page.clock.install(time=0)
        page.goto((ROOT / 'docs' / 'index.html').as_uri())
        page.evaluate('document.fonts.ready')
        # install() lets the fake time flow with the real one: pause it, from here on the page time moves only with
        # run_for(), by the same amount for every output frame
        page.clock.pause_at(60_000)
        page.clock.run_for(500)
        parts = []
        if a.only in (None, 'pipeline'):
            out = media / 'part1_pipeline.mp4'; parts.append(out)
            stage = "() => { const t = document.getElementById('hudNum').textContent; return /FASE 01/.test(t) ? 'first' : 'away'; }"
            render(page, '.rig:not(.rig-t)', '.rig:not(.rig-t) .rail-btn', lambda pg: pg.evaluate(stage), out, a.fps, step_ms, 150)
        if a.only in (None, 'tests'):
            out = media / 'part2_tests.mp4'; parts.append(out)
            slide = "() => { const t = document.getElementById('tNum').textContent; return /TEST 01/.test(t) ? 'first' : 'away'; }"
            render(page, '.rig-t', '#tRail .rail-btn', lambda pg: pg.evaluate(slide), out, a.fps, step_ms, 200)
        browser.close()
    if len(parts) == 2:
        listing = media / 'parts.txt'
        listing.write_text(''.join(f"file '{x.name}'\n" for x in parts))
        final = media / 'flytollm_decks.mp4'
        subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(listing), '-c', 'copy', '-movflags', '+faststart', str(final)])
        print('written', final, f'{final.stat().st_size / 2 ** 20:.1f} MB', flush=True)


if __name__ == '__main__':
    sys.exit(main())
