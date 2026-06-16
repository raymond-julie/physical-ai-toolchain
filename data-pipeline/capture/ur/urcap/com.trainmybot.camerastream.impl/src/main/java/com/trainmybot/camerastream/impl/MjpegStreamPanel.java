package com.trainmybot.camerastream.impl;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

import javax.imageio.ImageIO;
import javax.swing.JPanel;

/**
 * Swing panel that renders a live MJPEG ({@code multipart/x-mixed-replace})
 * stream. A single daemon thread reads the HTTP body, slices out each JPEG
 * frame by its SOI ({@code 0xFFD8}) / EOI ({@code 0xFFD9}) markers, decodes it,
 * and repaints. The connection auto-reconnects with backoff on any error.
 */
public class MjpegStreamPanel extends JPanel {

    /** Receives human-readable connection status updates. */
    public interface StatusListener {
        void onStatus(String message);
    }

    private static final int CONNECT_TIMEOUT_MS = 5000;
    private static final int READ_TIMEOUT_MS = 8000;
    private static final int MAX_FRAME_BYTES = 8 * 1024 * 1024;
    private static final long RECONNECT_MIN_MS = 500L;
    private static final long RECONNECT_MAX_MS = 5000L;

    private volatile BufferedImage currentImage;
    private volatile Thread worker;
    private volatile boolean running;
    private volatile String streamUrl;
    private StatusListener statusListener;

    public MjpegStreamPanel() {
        setBackground(Color.BLACK);
    }

    public void setStatusListener(StatusListener listener) {
        this.statusListener = listener;
    }

    /**
     * Stops any current stream and begins reading the given MJPEG URL.
     */
    public synchronized void start(String url) {
        stop();
        this.streamUrl = url;
        this.running = true;
        Thread t = new Thread(new Runnable() {
            @Override
            public void run() {
                runLoop();
            }
        }, "mjpeg-reader");
        t.setDaemon(true);
        this.worker = t;
        t.start();
    }

    /**
     * Stops the reader thread. Safe to call from the EDT; does not block.
     */
    public synchronized void stop() {
        this.running = false;
        Thread t = this.worker;
        this.worker = null;
        if (t != null) {
            t.interrupt();
        }
        this.currentImage = null;
        repaint();
    }

    private void runLoop() {
        long backoff = RECONNECT_MIN_MS;
        final Thread self = Thread.currentThread();
        while (running && worker == self) {
            HttpURLConnection connection = null;
            try {
                publishStatus("Connecting to " + streamUrl);
                URL url = new URL(streamUrl);
                connection = (HttpURLConnection) url.openConnection();
                connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
                connection.setReadTimeout(READ_TIMEOUT_MS);
                connection.setUseCaches(false);
                connection.connect();

                int status = connection.getResponseCode();
                if (status != HttpURLConnection.HTTP_OK) {
                    throw new IllegalStateException("HTTP " + status);
                }

                publishStatus("Streaming");
                backoff = RECONNECT_MIN_MS;
                readStream(connection.getInputStream(), self);
            } catch (Exception e) {
                if (running) {
                    publishStatus("Reconnecting (" + describe(e) + ")");
                }
            } finally {
                if (connection != null) {
                    connection.disconnect();
                }
            }

            if (!running || worker != self) {
                break;
            }
            sleepQuietly(backoff);
            backoff = Math.min(backoff * 2, RECONNECT_MAX_MS);
        }
    }

    /**
     * Reads JPEG frames from the multipart body by scanning for SOI/EOI markers.
     */
    private void readStream(InputStream rawStream, Thread self) throws Exception {
        BufferedInputStream in = new BufferedInputStream(rawStream, 64 * 1024);
        ByteArrayOutputStream frame = null;
        int prev = -1;
        int cur;
        while (running && worker == self && (cur = in.read()) != -1) {
            if (frame == null) {
                if (prev == 0xFF && cur == 0xD8) {
                    frame = new ByteArrayOutputStream(128 * 1024);
                    frame.write(0xFF);
                    frame.write(0xD8);
                }
            } else {
                frame.write(cur);
                if (prev == 0xFF && cur == 0xD9) {
                    decodeAndShow(frame.toByteArray());
                    frame = null;
                } else if (frame.size() > MAX_FRAME_BYTES) {
                    // Corrupt stream / never found EOI; drop and resync.
                    frame = null;
                }
            }
            prev = cur;
        }
    }

    private void decodeAndShow(byte[] jpegBytes) {
        try {
            BufferedImage image = ImageIO.read(new ByteArrayInputStream(jpegBytes));
            if (image != null) {
                currentImage = image;
                repaint();
            }
        } catch (Exception ignored) {
            // Skip a single bad frame.
        }
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        BufferedImage image = currentImage;
        int panelWidth = getWidth();
        int panelHeight = getHeight();

        if (image == null) {
            g.setColor(Color.LIGHT_GRAY);
            g.setFont(getFont().deriveFont(Font.PLAIN, 16f));
            String text = running ? "Connecting…" : "Stream stopped";
            int textWidth = g.getFontMetrics().stringWidth(text);
            g.drawString(text, (panelWidth - textWidth) / 2, panelHeight / 2);
            return;
        }

        double scale = Math.min(
                (double) panelWidth / image.getWidth(),
                (double) panelHeight / image.getHeight());
        int drawWidth = (int) Math.round(image.getWidth() * scale);
        int drawHeight = (int) Math.round(image.getHeight() * scale);
        int x = (panelWidth - drawWidth) / 2;
        int y = (panelHeight - drawHeight) / 2;

        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2.drawImage(image, x, y, drawWidth, drawHeight, null);
    }

    private void publishStatus(String message) {
        StatusListener listener = statusListener;
        if (listener != null) {
            listener.onStatus(message);
        }
    }

    private static String describe(Exception e) {
        String message = e.getMessage();
        if (message == null || message.isEmpty()) {
            return e.getClass().getSimpleName();
        }
        return message;
    }

    private static void sleepQuietly(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
