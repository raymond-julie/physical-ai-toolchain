package com.trainmybot.camerastream.impl;

import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;

import com.ur.urcap.api.contribution.installation.swing.SwingInstallationNodeService;
import com.ur.urcap.api.contribution.toolbar.swing.SwingToolbarService;

/**
 * OSGi entry point. Registers the Camera Stream installation node so the live
 * MJPEG feed appears under Installation &rarr; URCaps, plus a toolbar button in
 * the PolyScope header bar that opens the same live feed from any screen
 * (independent of Remote Control mode).
 */
public class Activator implements BundleActivator {

    @Override
    public void start(BundleContext context) {
        context.registerService(
                SwingInstallationNodeService.class,
                new CameraStreamInstallationNodeService(),
                null);

        context.registerService(
                SwingToolbarService.class,
                new CameraStreamToolbarService(),
                null);
    }

    @Override
    public void stop(BundleContext context) {
        // Services registered with the bundle context are unregistered automatically.
    }
}
