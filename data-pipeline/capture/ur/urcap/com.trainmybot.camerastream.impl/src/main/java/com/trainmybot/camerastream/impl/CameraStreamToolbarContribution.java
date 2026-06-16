package com.trainmybot.camerastream.impl;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

import com.ur.urcap.api.contribution.toolbar.ToolbarContext;
import com.ur.urcap.api.contribution.toolbar.swing.SwingToolbarContribution;

/**
 * Header-bar popup that shows the live MJPEG preview. Connection details are
 * read from the {@link CameraStreamInstallationNodeContribution}, so the popup
 * mirrors whatever streamer URL and camera id were configured on the
 * installation node.
 */
class CameraStreamToolbarContribution implements SwingToolbarContribution {

    private static final Dimension PREVIEW_SIZE = new Dimension(480, 360);
    private static final int HEADER_FONT_SIZE = 18;

    private final ToolbarContext context;
    private final MjpegStreamPanel previewPanel = new MjpegStreamPanel();
    private final JLabel statusLabel = new JLabel(" ");

    CameraStreamToolbarContribution(ToolbarContext context) {
        this.context = context;
    }

    @Override
    public void buildUI(JPanel panel) {
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

        panel.add(createHeader());
        panel.add(Box.createVerticalStrut(8));
        panel.add(createPreviewRow());
        panel.add(Box.createVerticalStrut(6));
        panel.add(createControlsRow());

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

    @Override
    public void openView() {
        previewPanel.start(resolveStreamUrl());
    }

    @Override
    public void closeView() {
        previewPanel.stop();
    }

    // --- UI -----------------------------------------------------------------

    private Component createHeader() {
        Box headerBox = Box.createHorizontalBox();
        headerBox.setAlignmentX(Component.CENTER_ALIGNMENT);
        JLabel header = new JLabel("Camera Stream");
        header.setFont(header.getFont().deriveFont(Font.BOLD, HEADER_FONT_SIZE));
        headerBox.add(header);
        return headerBox;
    }

    private JPanel createPreviewRow() {
        JPanel row = new JPanel(new BorderLayout());
        row.setAlignmentX(Component.CENTER_ALIGNMENT);
        previewPanel.setPreferredSize(PREVIEW_SIZE);
        previewPanel.setMinimumSize(PREVIEW_SIZE);
        previewPanel.setMaximumSize(PREVIEW_SIZE);
        previewPanel.setBorder(BorderFactory.createLineBorder(Color.GRAY));
        row.add(previewPanel, BorderLayout.CENTER);
        return row;
    }

    private JPanel createControlsRow() {
        JPanel row = new JPanel(new FlowLayout(FlowLayout.CENTER, 8, 0));
        row.setAlignmentX(Component.CENTER_ALIGNMENT);

        JButton reconnectButton = new JButton("Reconnect");
        reconnectButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                previewPanel.start(resolveStreamUrl());
            }
        });
        row.add(reconnectButton);
        row.add(new JLabel("Status:"));
        row.add(statusLabel);
        return row;
    }

    // --- Configuration ------------------------------------------------------

    /**
     * Reads the stream URL from the installation node. Returns an empty string
     * if the node is unavailable, which leaves the preview idle rather than
     * throwing.
     */
    private String resolveStreamUrl() {
        try {
            CameraStreamInstallationNodeContribution installation =
                    context.getAPIProvider().getApplicationAPI()
                            .getInstallationNode(CameraStreamInstallationNodeContribution.class);
            if (installation != null) {
                return installation.getStreamUrl();
            }
        } catch (Exception e) {
            statusLabel.setText("Configure the streamer under Installation -> URCaps.");
        }
        return "";
    }
}
