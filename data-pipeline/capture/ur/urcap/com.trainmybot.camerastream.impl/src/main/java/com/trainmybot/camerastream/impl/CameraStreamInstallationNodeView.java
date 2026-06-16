package com.trainmybot.camerastream.impl;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;

import com.ur.urcap.api.contribution.installation.swing.SwingInstallationNodeView;

/**
 * Swing view for the Camera Stream installation node. Shows the live MJPEG
 * preview and lets the operator point the URCap at a UrCameraStreamer host and
 * camera id. Text entry uses PolyScope's on-screen keyboard, wired through the
 * contribution.
 */
public class CameraStreamInstallationNodeView
        implements SwingInstallationNodeView<CameraStreamInstallationNodeContribution> {

    private static final Dimension PREVIEW_SIZE = new Dimension(640, 480);

    private final MjpegStreamPanel previewPanel = new MjpegStreamPanel();
    private final JTextField baseUrlField = new JTextField(28);
    private final JTextField cameraIdField = new JTextField(18);
    private final JLabel statusLabel = new JLabel(" ");

    @Override
    public void buildUI(JPanel panel, final CameraStreamInstallationNodeContribution contribution) {
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

        panel.add(createConnectionRow(contribution));
        panel.add(Box.createVerticalStrut(12));
        panel.add(createPreviewRow());
        panel.add(Box.createVerticalStrut(8));
        panel.add(createStatusRow());

        previewPanel.setStatusListener(new MjpegStreamPanel.StatusListener() {
            @Override
            public void onStatus(final String message) {
                SwingUtilities.invokeLater(new Runnable() {
                    @Override
                    public void run() {
                        statusLabel.setText(message);
                    }
                });
            }
        });
    }

    private JPanel createConnectionRow(final CameraStreamInstallationNodeContribution contribution) {
        JPanel row = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 4));
        row.setAlignmentX(Component.LEFT_ALIGNMENT);

        row.add(new JLabel("Streamer URL:"));
        baseUrlField.setEditable(false);
        baseUrlField.setFocusable(false);
        baseUrlField.setToolTipText("Streamer base URL (e.g. http://192.168.1.20:8000)");
        baseUrlField.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                contribution.getKeyboardForBaseUrl()
                        .show(baseUrlField, contribution.getCallbackForBaseUrl());
            }
        });
        row.add(baseUrlField);

        row.add(new JLabel("Camera id:"));
        cameraIdField.setEditable(false);
        cameraIdField.setFocusable(false);
        cameraIdField.setToolTipText("Camera id / serial (blank = use full URL above)");
        cameraIdField.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                contribution.getKeyboardForCameraId()
                        .show(cameraIdField, contribution.getCallbackForCameraId());
            }
        });
        row.add(cameraIdField);

        JButton reconnectButton = new JButton("Reconnect");
        reconnectButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                contribution.reconnect();
            }
        });
        row.add(reconnectButton);

        return row;
    }

    private JPanel createPreviewRow() {
        JPanel row = new JPanel(new BorderLayout());
        row.setAlignmentX(Component.LEFT_ALIGNMENT);
        previewPanel.setPreferredSize(PREVIEW_SIZE);
        previewPanel.setMinimumSize(PREVIEW_SIZE);
        previewPanel.setBorder(BorderFactory.createLineBorder(Color.GRAY));
        row.add(previewPanel, BorderLayout.WEST);
        return row;
    }

    private JPanel createStatusRow() {
        JPanel row = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 0));
        row.setAlignmentX(Component.LEFT_ALIGNMENT);
        row.add(new JLabel("Status:"));
        row.add(statusLabel);
        return row;
    }

    // --- Called by the contribution ---------------------------------------------

    public void setBaseUrl(String baseUrl) {
        baseUrlField.setText(baseUrl);
    }

    public void setCameraId(String cameraId) {
        cameraIdField.setText(cameraId);
    }

    public void startStream(String streamUrl) {
        previewPanel.start(streamUrl);
    }

    public void stopStream() {
        previewPanel.stop();
    }
}
