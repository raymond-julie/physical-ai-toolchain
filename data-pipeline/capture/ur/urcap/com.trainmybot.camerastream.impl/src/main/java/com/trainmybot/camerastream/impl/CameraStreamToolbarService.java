package com.trainmybot.camerastream.impl;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;

import javax.swing.Icon;
import javax.swing.ImageIcon;

import com.ur.urcap.api.contribution.toolbar.ToolbarConfiguration;
import com.ur.urcap.api.contribution.toolbar.ToolbarContext;
import com.ur.urcap.api.contribution.toolbar.swing.SwingToolbarContribution;
import com.ur.urcap.api.contribution.toolbar.swing.SwingToolbarService;

/**
 * Toolbar service that places a Camera Stream button in the PolyScope header
 * bar. Unlike the installation node, the toolbar popup is reachable from any
 * screen and is independent of Remote Control mode, so an operator can glance
 * at the live feed while the robot is driven remotely.
 *
 * <p>The popup reuses the connection details configured on the installation
 * node, so the streamer URL and camera id only need to be set once.</p>
 */
public class CameraStreamToolbarService implements SwingToolbarService {

    private static final int TOOLBAR_HEIGHT = 440;

    @Override
    public Icon getIcon() {
        return createCameraIcon();
    }

    @Override
    public void configureContribution(ToolbarConfiguration configuration) {
        configuration.setToolbarHeight(TOOLBAR_HEIGHT);
    }

    @Override
    public SwingToolbarContribution createToolbar(ToolbarContext context) {
        return new CameraStreamToolbarContribution(context);
    }

    /**
     * Draws a small camera glyph programmatically so the bundle needs no binary
     * icon asset.
     */
    private static Icon createCameraIcon() {
        // PolyScope requires toolbar icons to be 30 px or smaller.
        int size = 30;
        BufferedImage image = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = image.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.setColor(new Color(0x33, 0x33, 0x33));
            g.fillRoundRect(2, 8, 26, 18, 4, 4);
            g.fillRect(9, 4, 9, 6);
            g.setColor(Color.WHITE);
            g.fillOval(10, 11, 12, 12);
            g.setColor(new Color(0x33, 0x33, 0x33));
            g.fillOval(13, 14, 6, 6);
            g.setColor(Color.WHITE);
            g.setStroke(new BasicStroke(1f));
            g.drawOval(10, 11, 12, 12);
        } finally {
            g.dispose();
        }
        return new ImageIcon(image);
    }
}
