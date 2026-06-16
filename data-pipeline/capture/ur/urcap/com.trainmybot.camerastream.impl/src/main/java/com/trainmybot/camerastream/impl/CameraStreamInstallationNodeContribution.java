package com.trainmybot.camerastream.impl;

import com.ur.urcap.api.contribution.InstallationNodeContribution;
import com.ur.urcap.api.contribution.installation.InstallationAPIProvider;
import com.ur.urcap.api.domain.data.DataModel;
import com.ur.urcap.api.domain.script.ScriptWriter;
import com.ur.urcap.api.domain.userinteraction.keyboard.KeyboardInputCallback;
import com.ur.urcap.api.domain.userinteraction.keyboard.KeyboardInputFactory;
import com.ur.urcap.api.domain.userinteraction.keyboard.KeyboardTextInput;

/**
 * Installation-scoped contribution that persists the stream connection details
 * (base URL + camera id) and drives the live preview in the Swing view.
 *
 * <p>The configured values point at a running UrCameraStreamer instance, whose
 * MJPEG endpoint is {@code http://<host>:<port>/stream/<cameraId>}.</p>
 */
public class CameraStreamInstallationNodeContribution implements InstallationNodeContribution {

    private static final String BASE_URL_KEY = "baseUrl";
    private static final String CAMERA_ID_KEY = "cameraId";

    private static final String DEFAULT_BASE_URL = "http://192.168.1.20:8000";
    private static final String DEFAULT_CAMERA_ID = "";

    private final CameraStreamInstallationNodeView view;
    private final DataModel model;
    private final KeyboardInputFactory keyboardInputFactory;

    public CameraStreamInstallationNodeContribution(
            InstallationAPIProvider apiProvider,
            CameraStreamInstallationNodeView view,
            DataModel model) {
        this.view = view;
        this.model = model;
        this.keyboardInputFactory = apiProvider.getUserInterfaceAPI()
                .getUserInteraction()
                .getKeyboardInputFactory();
    }

    @Override
    public void openView() {
        view.setBaseUrl(getBaseUrl());
        view.setCameraId(getCameraId());
        view.startStream(getStreamUrl());
    }

    @Override
    public void closeView() {
        view.stopStream();
    }

    @Override
    public void generateScript(ScriptWriter writer) {
        // The camera preview is a pendant-only UI feature; it contributes no
        // robot program script.
    }

    // --- Persisted configuration -------------------------------------------------

    public String getBaseUrl() {
        return model.get(BASE_URL_KEY, DEFAULT_BASE_URL);
    }

    public String getCameraId() {
        return model.get(CAMERA_ID_KEY, DEFAULT_CAMERA_ID);
    }

    /**
     * Builds the full MJPEG endpoint from the stored base URL and camera id.
     * When no camera id is configured the base URL is used verbatim, allowing a
     * full stream URL to be pasted directly.
     */
    public String getStreamUrl() {
        String base = normalizeBaseUrl(getBaseUrl());
        String cameraId = getCameraId();
        if (cameraId == null || cameraId.trim().isEmpty()) {
            return base;
        }
        return base + "/stream/" + cameraId.trim();
    }

    /** Restarts the preview using the currently stored connection details. */
    public void reconnect() {
        view.startStream(getStreamUrl());
    }

    // --- On-screen keyboard wiring ----------------------------------------------

    public KeyboardTextInput getKeyboardForBaseUrl() {
        KeyboardTextInput keyboard = keyboardInputFactory.createStringKeyboardInput();
        keyboard.setInitialValue(getBaseUrl());
        return keyboard;
    }

    public KeyboardInputCallback<String> getCallbackForBaseUrl() {
        return new KeyboardInputCallback<String>() {
            @Override
            public void onOk(String value) {
                model.set(BASE_URL_KEY, normalizeBaseUrl(value));
                view.setBaseUrl(getBaseUrl());
                view.startStream(getStreamUrl());
            }
        };
    }

    public KeyboardTextInput getKeyboardForCameraId() {
        KeyboardTextInput keyboard = keyboardInputFactory.createStringKeyboardInput();
        keyboard.setInitialValue(getCameraId());
        return keyboard;
    }

    public KeyboardInputCallback<String> getCallbackForCameraId() {
        return new KeyboardInputCallback<String>() {
            @Override
            public void onOk(String value) {
                model.set(CAMERA_ID_KEY, value == null ? "" : value.trim());
                view.setCameraId(getCameraId());
                view.startStream(getStreamUrl());
            }
        };
    }

    private static String normalizeBaseUrl(String url) {
        if (url == null) {
            return DEFAULT_BASE_URL;
        }
        String trimmed = url.trim();
        if (trimmed.isEmpty()) {
            return DEFAULT_BASE_URL;
        }
        while (trimmed.endsWith("/")) {
            trimmed = trimmed.substring(0, trimmed.length() - 1);
        }
        return trimmed;
    }
}
