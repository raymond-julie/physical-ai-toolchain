/* ═══════════════════════════════════════════════════════════════
   UR Recorder — Client-side JavaScript
   ═══════════════════════════════════════════════════════════════ */

// ── WebSocket connection ────────────────────────────────────────
const socket = io();
let currentState = {};
let selectedRecording = null;

// ── DOM references ──────────────────────────────────────────────
const $id = (id) => document.getElementById(id);

const connBadge      = $id('connection-badge');
const connSource     = $id('conn-source');
const connDest       = $id('conn-dest');
const connSrcGrip    = $id('conn-src-grip');
const connDstGrip    = $id('conn-dst-grip');
const connCamera     = $id('conn-camera');
const connCamera2    = $id('conn-camera2');
const connCameraLabel  = $id('conn-camera-label');
const connCamera2Label = $id('conn-camera2-label');
const sourceJoints   = $id('source-joints');
const destJoints     = $id('dest-joints');
const srcGripVal     = $id('src-grip-val');
const dstGripVal     = $id('dst-grip-val');
const di0Val         = $id('di0-val');
const di1Val         = $id('di1-val');
const btnStart       = $id('btn-start');
const btnStop        = $id('btn-stop');
const recInfo        = $id('recording-info');
const recName        = $id('rec-name');
const recElapsed     = $id('rec-elapsed');
const recBody        = $id('recordings-body');
const recDetail      = $id('recording-detail');
const detailName     = $id('detail-name');
const detailTopics   = $id('detail-topics');
const playbackViewer = $id('playback-viewer');
const playbackName   = $id('playback-name');
const playbackFeed   = $id('playback-feed');
const cameraFeed     = $id('camera-feed');
const settingsOverlay = $id('settings-overlay');

const pills = {
    WAITING:   $id('pill-waiting'),
    ALIGNING:  $id('pill-aligning'),
    IDLE:      $id('pill-idle'),
    MIRRORING: $id('pill-mirroring'),
    RETURNING: $id('pill-returning'),
};


// ══════════════════════════════════════════════════════════════════
//  WebSocket state updates (10 Hz)
// ══════════════════════════════════════════════════════════════════

socket.on('connect', () => {
    connBadge.className = 'badge badge-on';
    connBadge.textContent = '● Connected';
});

socket.on('disconnect', () => {
    connBadge.className = 'badge badge-off';
    connBadge.textContent = '● Disconnected';
});

socket.on('state_update', (data) => {
    currentState = data;
    updateUI(data);
});


function updateUI(s) {
    // ── Connections ──
    setDot(connSource, s.source.connected);
    setDot(connDest, s.dest.connected);
    // Gripper connectivity approximated from main connection
    setDot(connSrcGrip, s.source.connected);
    setDot(connDstGrip, s.dest.connected);
    setDot(connCamera, s.camera_connected);
    setDot(connCamera2, s.camera2_connected);

    if (connCameraLabel) {
        connCameraLabel.textContent =
            'Camera 1' + (s.camera_model ? ` (${s.camera_model})` : '');
    }
    if (connCamera2Label) {
        connCamera2Label.textContent =
            'Camera 2' + (s.camera2_model ? ` (${s.camera2_model})` : '');
    }

    // ── Joints ──
    sourceJoints.textContent = formatJoints(s.source.joints);
    destJoints.textContent = formatJoints(s.dest.joints);

    // ── Grippers ──
    srcGripVal.textContent = s.source.gripper_closed
        ? `${(s.source.gripper_pos * 255).toFixed(0)}/255 (closed)`
        : `${(s.source.gripper_pos * 255).toFixed(0)}/255 (open)`;
    dstGripVal.textContent = (s.dest.gripper_pos * 255).toFixed(0) + '/255';

    // ── DI0/DI1 ──
    setIO(di0Val, s.source.di0);
    setIO(di1Val, s.source.di1);

    // ── State pills ──
    for (const [name, el] of Object.entries(pills)) {
        el.classList.toggle('active', s.state === name);
    }

    // ── Buttons ──
    if (s.state === 'IDLE') {
        btnStart.disabled = false;
        btnStop.disabled = true;
    } else if (s.state === 'MIRRORING') {
        btnStart.disabled = true;
        btnStop.disabled = false;
    } else {
        // WAITING, ALIGNING or RETURNING — both buttons disabled
        btnStart.disabled = true;
        btnStop.disabled = true;
    }

    // ── Motion confirmation overlay ──
    const confirmOverlay = $id('motion-confirm-overlay');
    if (confirmOverlay) {
        if (s.state === 'WAITING' && !s.motion_confirmed) {
            confirmOverlay.classList.remove('hidden');
        } else {
            confirmOverlay.classList.add('hidden');
        }
    }

    // ── Recording info ──
    if (s.recorder_active && s.current_bag) {
        recInfo.classList.remove('hidden');
        recName.textContent = s.current_bag;
        recElapsed.textContent = s.recording_elapsed ?? '0.0';
    } else {
        recInfo.classList.add('hidden');
    }
}

function setDot(el, on) {
    el.className = on ? 'dot dot-on' : 'dot dot-off';
}

function setIO(el, on) {
    el.textContent = on ? 'ON' : 'OFF';
    el.className = on ? 'io-badge io-on' : 'io-badge io-off';
}

function formatJoints(joints) {
    return '[' + joints.map(j => j.toFixed(3)).join(', ') + ']';
}


// ══════════════════════════════════════════════════════════════════
//  Controls — Start / Stop
// ══════════════════════════════════════════════════════════════════

btnStart.addEventListener('click', () => {
    socket.emit('request_toggle');
    btnStart.disabled = true;
});

btnStop.addEventListener('click', () => {
    socket.emit('request_toggle');
    btnStop.disabled = true;
});


// ══════════════════════════════════════════════════════════════════
//  Camera toggle (RGB / Depth)
// ══════════════════════════════════════════════════════════════════

document.querySelectorAll('.cam-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const targetId = btn.dataset.target || 'camera-feed';
        // Deactivate sibling buttons (same target)
        document.querySelectorAll(`.cam-btn[data-target="${targetId}"]`)
            .forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const target = $id(targetId);
        if (target) target.src = btn.dataset.src;
    });
});


// ══════════════════════════════════════════════════════════════════
//  Recordings
// ══════════════════════════════════════════════════════════════════

function loadRecordings() {
    fetch('/api/recordings')
        .then(r => r.json())
        .then(recs => renderRecordings(recs))
        .catch(() => {
            recBody.innerHTML = '<tr><td colspan="6" class="empty-row">Failed to load</td></tr>';
        });
}

function renderRecordings(recs) {
    if (!recs.length) {
        recBody.innerHTML = '<tr><td colspan="6" class="empty-row">No recordings yet</td></tr>';
        return;
    }

    recBody.innerHTML = recs.map((r, i) => `
        <tr data-name="${r.name}" class="${selectedRecording === r.name ? 'selected' : ''}">
            <td>${i + 1}</td>
            <td>${r.name}</td>
            <td>${r.duration}s</td>
            <td>${r.messages.toLocaleString()}</td>
            <td>${r.topic_count}/5</td>
            <td>
                <button class="icon-btn btn-row-play" data-name="${r.name}" title="Play">▶</button>
                <button class="icon-btn btn-row-delete" data-name="${r.name}" title="Delete">🗑</button>
            </td>
        </tr>
    `).join('');

    // Row click → show detail
    recBody.querySelectorAll('tr').forEach(tr => {
        tr.addEventListener('click', (e) => {
            if (e.target.closest('button')) return;  // ignore button clicks
            const name = tr.dataset.name;
            selectRecording(name, recs.find(r => r.name === name));
        });
    });

    // Play button
    recBody.querySelectorAll('.btn-row-play').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            startPlayback(btn.dataset.name);
        });
    });

    // Delete button
    recBody.querySelectorAll('.btn-row-delete').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteRecording(btn.dataset.name);
        });
    });
}

function selectRecording(name, rec) {
    selectedRecording = name;

    // Highlight row
    recBody.querySelectorAll('tr').forEach(tr => {
        tr.classList.toggle('selected', tr.dataset.name === name);
    });

    // Show detail
    recDetail.classList.remove('hidden');
    detailName.textContent = name;

    if (rec && rec.topics) {
        detailTopics.innerHTML = rec.topics.map(t => `
            <div class="topic-card">
                <div class="topic-name">${t.name}</div>
                <div class="topic-info">${t.type} — ${t.count.toLocaleString()} msgs</div>
            </div>
        `).join('');
    }
}

function startPlayback(name) {
    playbackViewer.classList.remove('hidden');
    playbackName.textContent = name;
    playbackFeed.src = `/api/recordings/${name}/playback`;
}

function deleteRecording(name) {
    if (!confirm(`Delete recording "${name}"?`)) return;

    fetch(`/api/recordings/${name}`, { method: 'DELETE' })
        .then(r => r.json())
        .then(res => {
            if (res.ok) {
                if (selectedRecording === name) {
                    recDetail.classList.add('hidden');
                    selectedRecording = null;
                }
                loadRecordings();
            } else {
                alert('Delete failed: ' + res.msg);
            }
        });
}

// Playback detail buttons
$id('btn-playback')?.addEventListener('click', () => {
    if (selectedRecording) startPlayback(selectedRecording);
});

$id('btn-delete')?.addEventListener('click', () => {
    if (selectedRecording) deleteRecording(selectedRecording);
});

$id('btn-close-playback')?.addEventListener('click', () => {
    playbackViewer.classList.add('hidden');
    playbackFeed.src = '';
});

$id('btn-refresh')?.addEventListener('click', loadRecordings);


// ══════════════════════════════════════════════════════════════════
//  Settings modal
// ══════════════════════════════════════════════════════════════════

$id('btn-settings')?.addEventListener('click', openSettings);
$id('btn-cancel-settings')?.addEventListener('click', closeSettings);
$id('btn-save-settings')?.addEventListener('click', saveSettings);

// Close on overlay click
settingsOverlay?.addEventListener('click', (e) => {
    if (e.target === settingsOverlay) closeSettings();
});

function openSettings() {
    fetch('/api/settings')
        .then(r => r.json())
        .then(s => {
            $id('set-source-ip').value    = s.source_ip;
            $id('set-dest-ip').value      = s.dest_ip;
            $id('set-gripper-port').value  = s.gripper_port;
            $id('set-gap-mult').value      = s.gap_multiplier;
            $id('set-min-gap').value       = s.min_gap_ms;
            $id('set-grace').value         = s.grace_period;
            $id('set-catchup').value       = s.catch_up_speed;
            $id('set-mode').value          = s.mode;
            settingsOverlay.classList.remove('hidden');
        });
}

function closeSettings() {
    settingsOverlay.classList.add('hidden');
}

function saveSettings() {
    const data = {
        source_ip:      $id('set-source-ip').value,
        dest_ip:        $id('set-dest-ip').value,
        gripper_port:   parseInt($id('set-gripper-port').value),
        gap_multiplier: parseFloat($id('set-gap-mult').value),
        min_gap_ms:     parseFloat($id('set-min-gap').value),
        grace_period:   parseFloat($id('set-grace').value),
        catch_up_speed: parseFloat($id('set-catchup').value),
        mode:           $id('set-mode').value,
    };

    fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    })
    .then(r => r.json())
    .then(res => {
        if (res.ok) {
            closeSettings();
        } else {
            alert('Save failed: ' + res.msg);
        }
    });
}


// ══════════════════════════════════════════════════════════════════//  Motion Confirmation
// ════════════════════════════════════════════════════════════════

(function setupMotionConfirmation() {
    const chkClear  = $id('chk-clear');
    const chkEstop  = $id('chk-estop');
    const chkBrakes = $id('chk-brakes');
    const chkUseHome = $id('chk-use-home');
    const useHomeHint = $id('use-home-hint');
    const chkUseDepth = $id('chk-use-depth');
    const useDepthHint = $id('use-depth-hint');
    const btnConfirm = $id('btn-confirm-motion');

    if (!btnConfirm) return;

    function updateConfirmBtn() {
        btnConfirm.disabled = !(chkClear.checked && chkEstop.checked && chkBrakes.checked);
    }

    function updateHomeHint() {
        if (chkUseHome.checked) {
            useHomeHint.textContent = 'Robot will align to home, then wait for Start.';
        } else {
            useHomeHint.textContent = 'Robot will stay in place and mirror directly on Start.';
        }
    }

    function updateDepthHint() {
        if (chkUseDepth.checked) {
            useDepthHint.textContent = 'Depth previews shown. Recording follows the launch flag (--depth / --no-depth).';
        } else {
            useDepthHint.textContent = 'Depth previews hidden. Recording follows the launch flag (--depth / --no-depth).';
        }
    }

    function applyDepthVisibility() {
        // Show/hide the Depth tab buttons + feeds based on the slider.
        const show = chkUseDepth.checked;
        document.querySelectorAll('.cam-btn[data-src*="/depth"]').forEach((btn) => {
            btn.style.display = show ? '' : 'none';
            // If a Depth tab was active and we just hid it, fall back to RGB.
            if (!show && btn.classList.contains('active')) {
                const target = btn.dataset.target;
                btn.classList.remove('active');
                const rgb = document.querySelector(
                    `.cam-btn[data-target="${target}"][data-src*="/color"]`);
                if (rgb) {
                    rgb.classList.add('active');
                    const img = $id(target);
                    if (img) img.src = rgb.dataset.src;
                }
            }
        });
    }

    chkClear.addEventListener('change', updateConfirmBtn);
    chkEstop.addEventListener('change', updateConfirmBtn);
    chkBrakes.addEventListener('change', updateConfirmBtn);
    chkUseHome.addEventListener('change', updateHomeHint);
    chkUseDepth.addEventListener('change', () => {
        updateDepthHint();
        applyDepthVisibility();
    });

    // Sync hints + depth visibility to the (unchecked) initial state.
    updateHomeHint();
    updateDepthHint();
    applyDepthVisibility();

    btnConfirm.addEventListener('click', () => {
        btnConfirm.disabled = true;
        btnConfirm.textContent = '⏳ Connecting...';
        socket.emit('request_confirm_motion', {
            use_home: chkUseHome.checked,
            use_depth: chkUseDepth.checked,
        });
    });
})();


// ════════════════════════════════════════════════════════════════//  Init
// ══════════════════════════════════════════════════════════════════

// Load recordings on page load
loadRecordings();

// Auto-refresh recordings every 10 seconds
setInterval(loadRecordings, 10000);
