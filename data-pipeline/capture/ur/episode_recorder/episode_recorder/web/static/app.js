// Minimal client for the episode-recorder GUI trigger.
// Polls /api/state twice per second, toggles via /api/toggle.

const statusEl = document.getElementById('status');
const stateText = document.getElementById('state-text');
const btn = document.getElementById('toggle');
const liveBadges = document.querySelectorAll('.camera-frame .live-badge');

const statEpisodes = document.getElementById('stat-episodes');
const statFrames = document.getElementById('stat-frames');
const statFps = document.getElementById('stat-fps');
const statSaving = document.getElementById('stat-saving');
const statSession = document.getElementById('stat-session');

let lastActive = null;

async function refresh() {
  try {
    const r = await fetch('/api/state', { cache: 'no-store' });
    if (!r.ok) return;
    const j = await r.json();
    applyState(j.active);
    applyCameraState(j.cameras || []);
    applyRobotState(j.robots || []);
    applyRecorderStats(j.stats, j.stats_age_seconds);
  } catch (e) {
    // Network blip; UI stays stale but recovers on next tick.
  }
}

function applyState(active) {
  if (active === lastActive) return;
  lastActive = active;
  if (active) {
    statusEl.classList.add('on');
    statusEl.classList.remove('off');
    stateText.textContent = 'RECORDING';
    btn.textContent = 'Stop';
    btn.classList.add('recording');
    document.body.classList.add('recording');
  } else {
    statusEl.classList.add('off');
    statusEl.classList.remove('on');
    stateText.textContent = 'IDLE';
    btn.textContent = 'Record';
    btn.classList.remove('recording');
    document.body.classList.remove('recording');
  }
}

function applyCameraState(cams) {
  for (const cam of cams) {
    const badge = document.querySelector(
      `.camera[data-index="${cam.index}"] .live-badge`);
    if (!badge) continue;
    if (cam.live) {
      badge.dataset.state = 'live';
      badge.textContent = 'live';
    } else if (cam.age_seconds != null) {
      badge.dataset.state = 'stale';
      badge.textContent = `stale ${cam.age_seconds.toFixed(1)}s`;
    } else {
      badge.dataset.state = 'unknown';
      badge.textContent = 'offline';
    }
  }
}

function fmtJoints(joints) {
  if (!joints || joints.length === 0) return '—';
  return joints.map(j => j.toFixed(3)).join(', ');
}

function fmtGripper(pos, closed) {
  if (pos == null && closed == null) return '—';
  const p = (pos == null) ? '?' : pos.toFixed(2);
  const c = (closed == null) ? '?' : (closed ? 'closed' : 'open');
  return `${p} (${c})`;
}

function applyRobotState(robots) {
  for (const r of robots) {
    const tile = document.querySelector(
      `.robot-tile[data-ns="${r.namespace}"]`);
    if (!tile) continue;
    tile.querySelector('.joints').textContent = fmtJoints(r.joints);
    tile.querySelector('.gripper').textContent =
      fmtGripper(r.gripper_position, r.gripper_closed);
    const badge = tile.querySelector('.live-badge');
    if (r.joints_live || r.gripper_live) {
      badge.dataset.state = 'live'; badge.textContent = 'live';
    } else if (r.joints_age_seconds != null || r.gripper_age_seconds != null) {
      const age = Math.max(r.joints_age_seconds || 0, r.gripper_age_seconds || 0);
      badge.dataset.state = 'stale';
      badge.textContent = `stale ${age.toFixed(1)}s`;
    } else {
      badge.dataset.state = 'unknown'; badge.textContent = 'offline';
    }
  }
}

function applyRecorderStats(stats, ageSec) {
  if (!statEpisodes) return;
  if (!stats || (ageSec != null && ageSec > 5)) {
    statEpisodes.textContent = '—';
    statFrames.textContent = '—';
    statFps.textContent = '—';
    statSaving.textContent = '—';
    statSession.textContent = 'session: —';
    return;
  }
  statEpisodes.textContent = String(stats.episode_count ?? '—');
  statFrames.textContent = String(stats.frames_in_episode ?? '—');
  statFps.textContent = String(stats.fps ?? '—');
  statSaving.textContent = stats.is_saving ? 'yes' : 'no';
  statSession.textContent =
    'session: ' + (stats.session_dir || stats.dataset_root || '—');
}

btn.addEventListener('click', async () => {
  btn.disabled = true;
  try {
    await fetch('/api/toggle', { method: 'POST' });
  } finally {
    setTimeout(() => { btn.disabled = false; refresh(); }, 150);
  }
});

setInterval(refresh, 500);
refresh();
